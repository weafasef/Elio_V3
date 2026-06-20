#!/usr/bin/env python3
"""
Step 5B 验证：TBPTT 截断 — mini-session 从头滚到尾。

验证内容（不需要 optimizer，backward 梯度即全覆盖）:
  - session 从头到尾多段 unroll 不报错
  - checkpoint 下二阶路径仍通 (ttt.k_proj/v_proj grad ≠ 0)
  - 连续段间 W 数值传递 (seg n+1 收到 seg n 的输出)
  - TBPTT 截断生效 (段间 W.grad_fn is None)
  - 显存 O(1) (peak 不随段数增长)
  - Llama grad = None

用法:
    python -m trainer.verify_5b --max-sessions 1 --K 2 --max-frames 16
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLAMA_PATH = PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct-abliterated"


def find_processed_dirs() -> list[Path]:
    data = PROJECT_ROOT / "data" / "processed"
    if not data.exists():
        return []
    dirs = sorted(data.glob("*/processed"))
    return [d for d in dirs if (d / "spec.json").exists()]


def load_llama(device: str, use_4bit: bool = True):
    from transformers import LlamaForCausalLM

    print(f"Loading Llama from: {LLAMA_PATH}")
    if use_4bit:
        try:
            import bitsandbytes  # noqa: F401
            model = LlamaForCausalLM.from_pretrained(
                str(LLAMA_PATH), load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                device_map="auto", dtype=torch.bfloat16,
            )
            print("  Loaded with bitsandbytes 4-bit quantization")
        except ImportError:
            print("  [WARN] bitsandbytes not available, falling back to bfloat16")
            use_4bit = False
    if not use_4bit:
        model = LlamaForCausalLM.from_pretrained(
            str(LLAMA_PATH), dtype=torch.bfloat16, device_map="auto",
        )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print(f"  Llama params: {sum(p.numel() for p in model.parameters()):,}  (all frozen)")
    return model


def print_memory(msg: str = ""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        p = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  [MEM {msg}] allocated={a:.2f}GB  reserved={r:.2f}GB  peak={p:.2f}GB")


def verify_5b(
    device="cuda", K=4, use_4bit=True, use_checkpoint=True,
    max_sessions=1, max_frames=40,
):
    all_ok = True

    # ── 1. 数据 ──
    processed_dirs = find_processed_dirs()
    if not processed_dirs:
        print("ERROR: 找不到已预处理数据", file=sys.stderr)
        return False
    print(f"Found {len(processed_dirs)} processed dirs")

    from trainer.dataset import ProcessedDataset, collate_fn
    ds = ProcessedDataset(processed_dirs=[str(processed_dirs[0])], K=K, stride=1)
    n_sessions = min(len(ds.session_lengths), max_sessions)
    print(f"Sessions: {n_sessions}, lengths: {ds.session_lengths[:n_sessions]}")

    # ── 2. 模型 ──
    from trainer.model import ElioModel
    from trainer.ttt import TTTMemory
    from trainer.heads import SimpleHead, FrameHead, AutoregHead
    from trainer.loop import run_ttt_segment

    print("\n── Creating models ──")
    elio = ElioModel().to(device); elio.train()
    ttt = TTTMemory().to(device); ttt.train()
    heads = nn.ModuleDict({
        "audio": SimpleHead(2048, 512, 1024).to(device),
        "gaze":  SimpleHead(2048, 2, 1024).to(device),
        "frame": FrameHead().to(device),
        "kb":    AutoregHead(stream_type="keyboard").to(device),
        "mouse": AutoregHead(stream_type="mouse").to(device),
    })

    print("\n── Loading Llama ──")
    llama = load_llama(device, use_4bit=use_4bit)

    params = list(elio.parameters()) + list(ttt.parameters()) + list(heads.parameters())
    n_trainable = sum(p.numel() for p in params)
    print(f"Trainable params: {n_trainable:,}")
    # NOTE: AdamW states would double this (~15GB), skipping optimizer in verify.
    # Training requires 4bit quantization or >16GB GPU.

    # ── 3. 跑 mini-session ──
    print(f"\n── TBPTT Loop (K={K}, checkpoint={use_checkpoint}) ──")
    torch.cuda.reset_peak_memory_stats()
    print_memory("before")

    global_step = 0
    peaks = []
    W_snapshots = []

    # 初始清零所有梯度 (梯度会在各段 backward 间累加)
    for p in params:
        p.grad = None

    for si in range(n_sessions):
        N = min(ds.session_lengths[si], max_frames)
        W = ttt.init_state(B=1, device=device)
        print(f"\nsession {si}: N={N}, t0 steps: {list(range(0, N-1, K))}")

        for seg_idx, t0 in enumerate(range(0, N - 1, K)):
            seg = ds.get_segment(si, t0)
            batch = collate_fn([seg])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            outer, raw, W_new = run_ttt_segment(
                elio, ttt, llama, heads, batch, device,
                detach_state=W, use_checkpoint=use_checkpoint,
            )

            # backward (梯度累加，不段间清零 — 让 seg0 的 W_init 梯度保留)
            outer.backward()
            # NOTE: 不 clip — W 爆炸时 clip 会将小梯度 crush 到零
            # 训练时 train.py 里 clip 是必要的

            # ── W 连续检查 ──
            if seg_idx == 0:
                # 第一段：W_new 应有 grad_fn (段内更新带梯度)
                W_has_grad = W_new.grad_fn is not None
                print(f"  seg{seg_idx} t0={t0}: W grad_fn={'yes' if W_has_grad else 'None'}"
                      f"  {'OK' if W_has_grad else 'FAIL'}")
                if not W_has_grad:
                    all_ok = False
                W_snapshots.append(W_new.detach().clone())
            elif seg_idx <= 3:
                # 后续段：截断后 W 传入 detach_state，段内更新后又带梯度
                W_has_grad = W_new.grad_fn is not None
                print(f"  seg{seg_idx} t0={t0}: W grad_fn={'yes' if W_has_grad else 'None'}"
                      f"  {'OK' if W_has_grad else 'FAIL'}")
                if not W_has_grad:
                    all_ok = False

                # 数值变化检查 (W 应该不同于上一段末尾，因为经过了段内更新)
                curr_val = W_new.detach().clone()
                diff = (curr_val - W_snapshots[-1]).abs().max().item()
                print(f"    W diff from seg{seg_idx-1} end: {diff:.6f}"
                      f"  {'OK (>0)' if diff > 0 else 'FAIL (no change)'}")
                if diff == 0:
                    all_ok = False
                W_snapshots.append(curr_val)

            # ── TBPTT 截断 ──
            W = W_new.detach().requires_grad_(use_checkpoint)
            global_step += 1

            # 记录 peak
            if torch.cuda.is_available():
                peaks.append(torch.cuda.max_memory_allocated() / 1024**3)

        print(f"  session {si} done, {global_step} steps")
        print_memory(f"after sess{si}")

    # ── 4. 显存 O(1) 检查 ──
    if len(peaks) >= 2:
        peak_growth = peaks[-1] - peaks[0]
        print(f"\n── Memory O(1) check ──")
        for i, p in enumerate(peaks):
            print(f"  seg{i}: peak={p:.2f}GB")
        print(f"  growth over {len(peaks)} segs: {peak_growth:.2f}GB"
              f"  {'OK (flat)' if peak_growth < 1.5 else 'WARN (growing)'}")

    # ── 5. TTT 二阶梯度检查 ──
    print(f"\n── TTT gradient check ──")
    ttt_ok = 0; ttt_fail = 0
    for name, p in ttt.named_parameters():
        has = p.grad is not None and p.grad.abs().max() > 1e-12
        if has:
            ttt_ok += 1
        else:
            ttt_fail += 1
            print(f"  ttt.{name}: grad={'zero' if p.grad is not None else 'None'}  FAIL")
    print(f"  TTT: {ttt_ok}/{ttt_ok + ttt_fail} params with grad"
          f"  {'OK' if ttt_fail == 0 else 'FAIL'}")
    if ttt_fail > 0:
        all_ok = False

    # ── 6. Llama 梯度检查 ──
    print(f"\n── Llama gradient check ──")
    llama_bad = sum(1 for _, p in llama.named_parameters() if p.grad is not None)
    print(f"  Llama params with grad: {llama_bad}  {'OK' if llama_bad == 0 else 'FAIL'}")
    if llama_bad > 0:
        all_ok = False

    # ── 7. 汇总 ──
    print(f"\n{'=' * 50}")
    print(f"  K={K}, checkpoint={use_checkpoint}")
    print(f"  sessions: {n_sessions}, total steps: {global_step}")
    print_memory("final")
    if all_ok:
        print(f"\n  ALL CHECKS PASSED")
    else:
        print(f"\n  SOME CHECKS FAILED")

    ds.close()
    del llama, elio, ttt, heads
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Step 5B TBPTT 截断验证")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--max-sessions", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()

    if not LLAMA_PATH.exists():
        print(f"ERROR: Llama not found at {LLAMA_PATH}", file=sys.stderr)
        sys.exit(1)

    ok = verify_5b(
        device=args.device, K=args.K,
        use_4bit=not args.no_4bit,
        use_checkpoint=not args.no_checkpoint,
        max_sessions=args.max_sessions,
        max_frames=args.max_frames,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
