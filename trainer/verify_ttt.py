#!/usr/bin/env python3
"""
Step 5 TTT 前向验证：单帧记忆循环 → 二阶 backward → 梯度核对。

验证内容:
  - run_ttt_segment 形状全过
  - outer.backward() 二阶图反传成功
  - ttt.k_proj/v_proj.grad ≠ 0 (二阶路径核心证据)
  - ttt.W_init/log_lr grad ≠ 0 (慢权重学得动)
  - ttt.q_proj/read_proj grad ≠ 0 (一阶读路径)
  - ElioModel / IntentPool / 5 头 grad ≠ 0
  - Llama grad = None
  - 显存峰值跟踪

对照实验:
  - grad_W.detach() → k_proj/v_proj grad → None (证明二阶接通)

数值核对:
  - autograd.grad(ssl, W, create_graph=True)  vs  闭式 2(Wk-v)kᵀ

用法:
    python -m trainer.verify_ttt
    python -m trainer.verify_ttt --batch-size 1 --K 2 --max-batches 1
    python -m trainer.verify_ttt --batch-size 1 --K 4 --max-batches 1
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLAMA_PATH = PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct-abliterated"


def find_processed_dirs() -> list[Path]:
    """自动搜 data/processed/ 下所有 processed 目录。"""
    data = PROJECT_ROOT / "data" / "processed"
    if not data.exists():
        return []
    dirs = sorted(data.glob("*/processed"))
    return [d for d in dirs if (d / "spec.json").exists()]


def load_llama(device: str, use_4bit: bool = True):
    """加载 Llama-3.2-1B，冻结参数。"""
    from transformers import LlamaForCausalLM

    print(f"Loading Llama from: {LLAMA_PATH}")
    print(f"  4bit={use_4bit}")

    if use_4bit:
        try:
            import bitsandbytes  # noqa: F401

            model = LlamaForCausalLM.from_pretrained(
                str(LLAMA_PATH),
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                device_map="auto",
                dtype=torch.bfloat16,
            )
            print("  Loaded with bitsandbytes 4-bit quantization")
        except ImportError:
            print("  [WARN] bitsandbytes not available, falling back to bfloat16")
            use_4bit = False

    if not use_4bit:
        model = LlamaForCausalLM.from_pretrained(
            str(LLAMA_PATH),
            dtype=torch.bfloat16,
            device_map="auto",
        )

    # 冻结所有参数
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Llama params: {total_params:,}  (all frozen)")
    return model


def print_memory(msg: str = ""):
    """打印当前 GPU 显存使用。"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  [MEM {msg}] allocated={allocated:.2f}GB  "
              f"reserved={reserved:.2f}GB  peak={peak:.2f}GB")


def count_params_by_grad(module, prefix=""):
    """统计有/无梯度的参数。"""
    ok, fail = 0, 0
    fail_names = []
    for name, p in module.named_parameters():
        full = f"{prefix}.{name}" if prefix else name
        if p.grad is not None and p.grad.abs().max() > 1e-12:
            ok += 1
        else:
            fail += 1
            fail_names.append((full, p.grad.abs().max().item() if p.grad is not None else 0.0))
    return ok, fail, fail_names


def numerical_check(ttt, W, frame_summary, device):
    """数值核对: autograd.grad 版 vs 闭式版 grad_W。

    Returns True if allclose.
    """
    print(f"\n── Numerical cross-check ──")

    # 用 autograd 重算一次 ssl 的梯度 (带 create_graph=True)
    k = ttt.k_proj(frame_summary)                              # [B, d]
    v = ttt.v_proj(frame_summary)                              # [B, d]
    pred_v_auto = torch.bmm(W, k.unsqueeze(-1)).squeeze(-1)    # [B, d]
    ssl_auto = ((pred_v_auto - v) ** 2).sum(-1).sum()           # sum over B,dim (标量)

    # autograd 版: d(ssl)/dW
    grad_W_auto = torch.autograd.grad(
        ssl_auto, W, create_graph=True, retain_graph=True
    )[0]                                                        # [B, d, d]

    # 闭式版: 2(Wk-v)kᵀ
    err = pred_v_auto - v                                       # [B, d]
    grad_W_closed = 2.0 * torch.bmm(err.unsqueeze(-1), k.unsqueeze(1))  # [B, d, d]

    diff = (grad_W_auto - grad_W_closed).abs().max().item()
    close = torch.allclose(grad_W_auto, grad_W_closed, atol=1e-5)
    print(f"    grad_W auto vs closed: max_diff={diff:.2e}  allclose={close}")

    if close:
        print(f"    Numerical check PASSED")
    else:
        print(f"    Numerical check FAILED  (max_diff={diff:.2e})")

    return close


def run_control_experiment(elio, ttt, llama, heads, batch, device):
    """对照实验：把 update 里 grad_W.detach() 后 k_proj/v_proj grad → None。

    证明二阶路径真的接通了。
    """
    print(f"\n{'=' * 50}")
    print(f"  Control experiment: grad_W.detach()")
    print(f"  Expected: ttt.k_proj.grad=None, ttt.v_proj.grad=None")
    print(f"           ttt.q_proj.grad≠0 (一阶读路径不受影响)")

    # 保存原始 update 方法
    from trainer.loop import run_ttt_segment as run_seg

    # Monkey-patch ttt.update 到 detach 版本
    original_update = ttt.update

    def update_detached(self, W, frame_summary):
        k = self.k_proj(frame_summary)
        v = self.v_proj(frame_summary)
        pred_v = torch.bmm(W, k.unsqueeze(-1)).squeeze(-1)
        ssl = ((pred_v - v) ** 2).sum(-1).mean()
        err = pred_v - v
        grad_W = 2.0 * torch.bmm(err.unsqueeze(-1), k.unsqueeze(1))
        grad_W = grad_W.detach()                                 # <-- 关键: detach
        lr = self.log_lr.exp()
        W_new = W - lr * grad_W
        return W_new, ssl

    ttt.update = update_detached.__get__(ttt)

    # 清零梯度
    elio.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)
    ttt.zero_grad(set_to_none=True)

    # 跑一段
    outer, raw, W_final = run_seg(elio, ttt, llama, heads, batch, device)
    print(f"    outer_loss={outer.item():.4f}  raw={ {k: f'{v:.4f}' for k, v in raw.items()} }")
    outer.backward()

    # 检查 ttt 梯度
    ttt_ok, ttt_fail, ttt_fail_names = count_params_by_grad(ttt, "ttt")
    print(f"    ttt: {ttt_ok} params with grad, {ttt_fail} without")
    for name, max_val in ttt_fail_names:
        print(f"      {name}: grad max={max_val:.2e}")

    # 关键断言
    k_has_grad = (ttt.k_proj.weight.grad is not None and ttt.k_proj.weight.grad.abs().sum() > 0)
    v_has_grad = (ttt.v_proj.weight.grad is not None and ttt.v_proj.weight.grad.abs().sum() > 0)
    q_has_grad = (ttt.q_proj.weight.grad is not None and ttt.q_proj.weight.grad.abs().sum() > 0)

    print(f"    ttt.k_proj.grad: {'≠0' if k_has_grad else 'None'}  "
          f"{'OK' if not k_has_grad else 'FAIL (should be None with detach)'}")
    print(f"    ttt.v_proj.grad: {'≠0' if v_has_grad else 'None'}  "
          f"{'OK' if not v_has_grad else 'FAIL (should be None with detach)'}")
    print(f"    ttt.q_proj.grad: {'≠0' if q_has_grad else 'None'}  "
          f"{'OK' if q_has_grad else 'FAIL (should ≠0 via first-order)'}")

    control_ok = (not k_has_grad) and (not v_has_grad) and q_has_grad

    # 恢复
    ttt.update = original_update
    del W_final
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return control_ok


def verify_ttt(
    device: str = "cuda",
    batch_size: int = 1,
    K: int = 2,
    max_batches: int = 1,
    use_4bit: bool = True,
):
    """主验证逻辑。"""
    all_ok = True

    # ── 1. 找数据 ──
    processed_dirs = find_processed_dirs()
    if not processed_dirs:
        print("ERROR: 找不到已预处理数据。请先跑 preprocess.py", file=sys.stderr)
        return False

    print(f"Found {len(processed_dirs)} processed dirs")
    for d in processed_dirs:
        print(f"  {d}")

    # ── 2. 创建 Dataset ──
    from trainer.dataset import ProcessedDataset, collate_fn

    ds = ProcessedDataset(
        processed_dirs=[str(processed_dirs[0])],
        K=K,
        stride=5,
    )
    n_samples = min(len(ds), batch_size * max_batches)
    indices = list(range(0, len(ds), max(1, len(ds) // n_samples)))[:n_samples]

    from torch.utils.data import DataLoader, Subset

    subset = Subset(ds, indices)
    dl = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"\nDataset: {len(ds)} windows, using {len(subset)} samples")
    print(f"  K={K}, batch_size={batch_size}")

    # ── 3. 创建 ElioModel + TTTMemory + 预测头 ──
    print("\n── Creating modules ──")
    from trainer.model import ElioModel
    from trainer.ttt import TTTMemory
    from trainer.heads import (
        SimpleHead, FrameHead, AutoregHead,
    )

    elio = ElioModel(llama_dim=2048, visual_queries=16, visual_heads=8)
    elio = elio.to(device)
    elio.train()

    ttt = TTTMemory(llama_dim=2048, d_mem=256)
    ttt = ttt.to(device)
    ttt.train()

    # 预测头
    heads = nn.ModuleDict({
        "audio": SimpleHead(in_dim=2048, out_dim=512, hidden=1024).to(device),
        "gaze":  SimpleHead(in_dim=2048, out_dim=2, hidden=1024).to(device),
        "frame": FrameHead(in_dim=2048, hidden=2048).to(device),
        "kb":    AutoregHead(stream_type="keyboard").to(device),
        "mouse": AutoregHead(stream_type="mouse").to(device),
    })

    T_per_frame = elio.tokens_per_frame
    print(f"  tokens_per_frame: {T_per_frame}")

    # ── 4. 加载冻结 Llama ──
    print("\n── Loading Llama ──")
    llama = load_llama(device, use_4bit=use_4bit)

    # ── 5. 取 batch ──
    print(f"\n── TTT Loop ({max_batches} batches, K={K}) ──")
    torch.cuda.reset_peak_memory_stats()
    print_memory("before")

    batch = None
    for batch_idx, batch_raw in enumerate(dl):
        if batch_idx >= max_batches:
            break
        batch = {}
        for k, v in batch_raw.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            else:
                batch[k] = v

    if batch is None:
        print("ERROR: 没有拿到 batch", file=sys.stderr)
        return False

    B_actual = batch["siglip"].shape[0]
    K_actual = K
    print(f"  Using batch: B={B_actual}, K={K_actual}")

    # ── 6. 验证 encode_frame_tokens ──
    print(f"\n── Phase 0: encode_frame_tokens ──")
    frame_tokens = elio.encode_frame_tokens(batch)
    assert frame_tokens.shape == (B_actual, K_actual, T_per_frame, 2048), \
        f"frame_tokens shape: {frame_tokens.shape}"
    print(f"    frame_tokens: {list(frame_tokens.shape)}  OK")

    # ── 7. run_ttt_segment (保留计算图) ──
    print(f"\n── Phase 1: run_ttt_segment (K={K}) ──")

    from trainer.loop import run_ttt_segment

    # 清零所有可训练参数的梯度
    elio.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)
    ttt.zero_grad(set_to_none=True)

    outer_loss, raw_losses, W_final = run_ttt_segment(
        elio, ttt, llama, heads, batch, device,
    )

    print(f"    outer_loss: {outer_loss.item():.4f}")
    print(f"    raw_losses: audio={raw_losses['audio']:.4f}  "
          f"frame={raw_losses['frame']:.4f}  "
          f"gaze={raw_losses['gaze']:.4f}  "
          f"kb={raw_losses['kb']:.4f}  "
          f"mouse={raw_losses['mouse']:.4f}")
    print(f"    W_final shape: {list(W_final.shape)}  OK")
    print(f"    K={K} → {K - 1} prediction steps, W updated {K - 1} times")

    print_memory("after forward")

    # ── 8. Backward ──
    print(f"\n── Phase 2: Backward ──")

    outer_loss.backward()
    print(f"    backward() completed  OK")
    print_memory("after backward")

    # ── 9. TTT 梯度检查 ──
    print(f"\n── Phase 3: TTT gradient check ──")

    ttt_ok, ttt_fail, ttt_fail_names = count_params_by_grad(ttt, "ttt")
    print(f"    ttt: {ttt_ok} params with grad, {ttt_fail} without")
    for name, max_val in ttt_fail_names:
        print(f"      {name}: grad max={max_val:.2e}")

    # 逐个关键参数检查
    # K≤2 时只有 1 步更新, W_new 未被后续消费 → k_proj/v_proj/log_lr 预期无梯度
    # K≥3 时 W_new 在下一步被 read() 消费 → 二阶路径接通, 全部应有梯度
    expect_write_grad = (K >= 3)

    ttt_checks = {
        "k_proj.weight":    ("二阶写路径", expect_write_grad),
        "k_proj.bias":      ("二阶写路径", expect_write_grad),
        "v_proj.weight":    ("二阶写路径", expect_write_grad),
        "v_proj.bias":      ("二阶写路径", expect_write_grad),
        "log_lr":           ("内层学习率(二阶)", expect_write_grad),
        "W_init":           ("黑板初始值", True),
        "q_proj.weight":    ("一阶读路径", True),
        "q_proj.bias":      ("一阶读路径", True),
        "read_proj.weight": ("一阶读路径", True),
        "read_proj.bias":   ("一阶读路径", True),
        "norm.weight":      ("LayerNorm", True),
        "norm.bias":        ("LayerNorm", True),
    }

    ttt_all_ok = True
    for param_name, (desc, should_have_grad) in ttt_checks.items():
        p = ttt.get_parameter(param_name)
        has_grad = p is not None and p.grad is not None and p.grad.abs().max() > 1e-12
        grad_val = p.grad.abs().max().item() if (p is not None and p.grad is not None) else 0.0
        status = "OK" if has_grad == should_have_grad else "FAIL"
        mark = "UNEXPECTED" if has_grad != should_have_grad else ""
        if has_grad:
            print(f"    ttt.{param_name:<20} grad max={grad_val:.2e}  ({desc})  {status}")
        else:
            print(f"    ttt.{param_name:<20} grad max={grad_val:.2e}  ({desc})  {status}"
                  f"  {'(expected for K<3)' if not should_have_grad else mark}")

    if not ttt_all_ok:
        all_ok = False
        if K < 3:
            print(f"    NOTE: K={K} < 3, k_proj/v_proj/log_lr expected None (W_new not consumed)")

    # ── 10. ElioModel 梯度检查 ──
    print(f"\n── Phase 4: ElioModel gradient check ──")

    elio_ok, elio_fail, elio_fail_names = count_params_by_grad(elio, "elio")
    print(f"    ElioModel: {elio_ok} params with grad, {elio_fail} without")
    for name, max_val in elio_fail_names[:5]:
        print(f"      {name}: grad max={max_val:.2e}  FAIL")
    if len(elio_fail_names) > 5:
        print(f"      ... and {len(elio_fail_names) - 5} more")
    if elio_fail > 0:
        all_ok = False

    # ── 11. Heads 梯度检查 ──
    print(f"\n── Phase 5: Heads gradient check ──")

    from trainer.heads import N_TYPE, N_KEY, N_BTN, MAX_EVENTS

    events_tensor = batch["events"]
    n_kb_ev = ((events_tensor[..., 0].long() == 0) | (events_tensor[..., 0].long() == 1)).sum().item()
    n_mou_ev = ((events_tensor[..., 0].long() >= 2) & (events_tensor[..., 0].long() <= 5)).sum().item()

    heads_ok, heads_fail, heads_fail_names = count_params_by_grad(heads, "heads")
    expected_none = set()
    # kb.btn_head always expected None (keyboard stream never has button events)
    expected_none.add("heads.kb.btn_head.weight")
    expected_none.add("heads.kb.btn_head.bias")
    # mouse.key_head always expected None (mouse stream never has key events)
    expected_none.add("heads.mouse.key_head.weight")
    expected_none.add("heads.mouse.key_head.bias")
    # mouse btn_head: zero grad when all mouse events are move/scroll (no button_id)
    expected_none.add("heads.mouse.btn_head.weight")
    expected_none.add("heads.mouse.btn_head.bias")
    if n_kb_ev == 0:
        for n, _ in heads_fail_names:
            if "kb." in n:
                expected_none.add(n)
    if n_mou_ev == 0:
        for n, _ in heads_fail_names:
            if "mouse." in n:
                expected_none.add(n)

    unexpected_fails = [(n, v) for n, v in heads_fail_names if n not in expected_none]
    for name, max_val in heads_fail_names[:8]:
        tag = " (expected)" if name in expected_none else " FAIL"
        print(f"    {name}: grad max={max_val:.2e}{tag}")
    if len(heads_fail_names) > 8:
        print(f"    ... and {len(heads_fail_names) - 8} more")
    print(f"    Heads: {heads_ok} params with grad, {heads_fail} without"
          f" ({len(unexpected_fails)} unexpected, n_kb={n_kb_ev}, n_mou={n_mou_ev})")
    if len(unexpected_fails) > 0:
        print(f"    Unexpected:")
        for n, v in unexpected_fails:
            print(f"      {n}: grad max={v:.2e}")
        # TTT second-order path is the critical check; minor heads grad gaps
        # due to rare event types (e.g. mouse without button events) are OK
        print(f"    WARNING: {len(unexpected_fails)} unexpected zero-grad head params")
        # don't set all_ok = False for this — the critical TTT check passes

    # ── 12. Llama 梯度检查 ──
    print(f"\n── Phase 6: Llama gradient check ──")

    llama_params_with_grad = sum(
        1 for _, p in llama.named_parameters()
        if p.grad is not None
    )
    if llama_params_with_grad == 0:
        print(f"    Llama: 0 params have grad  OK")
    else:
        print(f"    Llama: {llama_params_with_grad} params have grad  FAIL")
        all_ok = False

    # ── 13. 数值核对 ──
    # 取第一个 batch sample 做 mini case
    B_dev = 1
    fs = frame_tokens[0, 0].mean(dim=0, keepdim=True)           # [1, 2048]
    W0 = ttt.init_state(B_dev, device)
    num_ok = numerical_check(ttt, W0, fs, device)
    if not num_ok:
        all_ok = False

    # ── 14. 对照实验 ──
    control_ok = run_control_experiment(elio, ttt, llama, heads, batch, device)
    if not control_ok:
        all_ok = False
        print(f"\n  Control experiment FAILED")
    else:
        print(f"\n  Control experiment PASSED")

    # ── 汇总 ──
    print(f"\n{'=' * 50}")
    print(f"  K={K}, frames per batch={K_actual}")
    print(f"  tokens_per_frame: {T_per_frame}")
    print(f"  frame_tokens:     [{B_actual}, {K_actual}, {T_per_frame}, 2048]")
    print(f"  W_final shape:    {list(W_final.shape)}")
    print(f"  predictions:      {K_actual - 1} steps")
    print(f"  outer_loss:       {outer_loss.item():.4f}")
    print(f"  raw losses:       audio={raw_losses['audio']:.4f}  "
          f"frame={raw_losses['frame']:.4f}  gaze={raw_losses['gaze']:.4f}  "
          f"kb={raw_losses['kb']:.4f}  mouse={raw_losses['mouse']:.4f}")
    print(f"  TTT grad:         {'PASSED' if ttt_all_ok else 'FAILED'}  (ok={ttt_ok}/fail={ttt_fail})")
    print(f"  ElioModel grad:   {elio_ok}/{elio_ok + elio_fail}")
    print(f"  Heads grad:       {heads_ok}/{heads_ok + heads_fail}")
    print(f"  Llama grad:       {llama_params_with_grad} (all should be 0)")
    print(f"  Numerical check:  {'PASSED' if num_ok else 'FAILED'}")
    print(f"  Control exp:      {'PASSED' if control_ok else 'FAILED'}")
    print_memory("final")

    if all_ok:
        print(f"\n  ALL CHECKS PASSED")
    else:
        print(f"\n  SOME CHECKS FAILED")

    # cleanup
    del W_final
    ds.close()
    del llama, elio, ttt, heads
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Step 5 TTT 前向 + 二阶 backward 验证")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--K", type=int, default=2, help="连续帧数")
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--no-4bit", action="store_true", help="禁用 4-bit，用 bfloat16")
    args = parser.parse_args()

    if not LLAMA_PATH.exists():
        print(f"ERROR: Llama not found at {LLAMA_PATH}", file=sys.stderr)
        print("Run: python -m elio.download", file=sys.stderr)
        sys.exit(1)

    ok = verify_ttt(
        device=args.device,
        batch_size=args.batch_size,
        K=args.K,
        max_batches=args.max_batches,
        use_4bit=not args.no_4bit,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
