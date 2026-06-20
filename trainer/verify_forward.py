#!/usr/bin/env python3
"""
最小前向验证：Dataset → merge+proj → frozen Llama → h_t → IntentPool → 5 heads → backward。

Step 4 扩展:
  - IntentPool 输出 5 意图 [B,K,5,2048]
  - audio_head / gaze_head / frame_head 形状 + loss + backward
  - AutoregHead × 2 (键盘/鼠标) teacher forcing + field_mask + 空帧
  - detach 归一化总 loss，5 头一起 backward

用法:
    python -m trainer.verify_forward
    python -m trainer.verify_forward --batch-size 1 --K 4 --max-batches 2
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


def verify_forward(
    device: str = "cuda",
    batch_size: int = 2,
    K: int = 4,
    max_batches: int = 2,
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

    # ── 3. 创建 ElioModel + 预测头 ──
    print("\n── Creating ElioModel + Heads ──")
    from trainer.model import ElioModel
    from trainer.heads import (
        IntentPool, SimpleHead, FrameHead, AutoregHead,
        frame_loss, audio_loss, gaze_loss, total_loss,
    )

    elio = ElioModel(llama_dim=2048, visual_queries=16, visual_heads=8)
    elio = elio.to(device)
    elio.train()

    # 预测头
    audio_head = SimpleHead(in_dim=2048, out_dim=512, hidden=1024).to(device)
    gaze_head = SimpleHead(in_dim=2048, out_dim=2, hidden=1024).to(device)
    frame_head = FrameHead().to(device)
    kb_head = AutoregHead(stream_type="keyboard").to(device)
    mouse_head = AutoregHead(stream_type="mouse").to(device)

    heads = nn.ModuleDict({
        "audio": audio_head,
        "gaze": gaze_head,
        "frame": frame_head,
        "kb": kb_head,
        "mouse": mouse_head,
    })

    T_per_frame = elio.tokens_per_frame
    print(f"  tokens_per_frame: {T_per_frame}  (1 audio + 16 full + 16 fovea + 1 action)")

    # ── 4. 加载冻结 Llama ──
    print("\n── Loading Llama ──")
    llama = load_llama(device, use_4bit=use_4bit)

    # ── 5. 取一个 batch，前向 ──
    print(f"\n── Forward + Heads ({max_batches} batches) ──")
    torch.cuda.reset_peak_memory_stats()
    print_memory("before")

    # 收集所有 batch
    batches = []
    for batch_idx, batch in enumerate(dl):
        if batch_idx >= max_batches:
            break
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.to(device)
            else:
                batch_gpu[k] = v
        batches.append(batch_gpu)

    batch = batches[-1]  # 用最后一个做详细验证
    B_actual = batch["siglip"].shape[0]
    print(f"\n  Using batch: B={B_actual}, K={K}")

    # ── ElioModel + Llama 前向 → intents ──
    elio_out = elio(batch, llama=llama)
    inputs_embeds = elio_out["inputs_embeds"]
    h_t = elio_out["h_t"]
    intents = elio_out["intents"]                                    # [B, K, 5, 2048]

    expected_seq_len = K * T_per_frame
    assert inputs_embeds.shape == (B_actual, expected_seq_len, 2048), \
        f"inputs_embeds shape: {inputs_embeds.shape}"
    assert h_t.shape == (B_actual, expected_seq_len, 2048), \
        f"h_t shape: {h_t.shape}"
    print(f"    inputs_embeds: {list(inputs_embeds.shape)}  OK")
    print(f"    h_t:           {list(h_t.shape)}  OK")

    # ── 验证 IntentPool ──
    assert intents.shape == (B_actual, K, 5, 2048), \
        f"intents shape: {intents.shape}, expected ({B_actual}, {K}, 5, 2048)"
    print(f"    intents:       {list(intents.shape)}  OK")

    for i, name in enumerate(["audio", "gaze", "frame", "kb", "mouse"]):
        intent_i = elio_out[f"intent_{name[0]}"]  # intent_a, intent_g, etc.
        assert intent_i.shape == (B_actual, K, 2048), \
            f"intent_{name[0]} shape: {intent_i.shape}"
    print(f"    5 intents:     each [{B_actual}, {K}, 2048]  OK")

    print_memory("after IntentPool")

    # ══════════════════════════════════════════════════════
    #  阶段 A: audio / gaze / frame 三头
    # ══════════════════════════════════════════════════════

    print(f"\n── Phase A: Simple heads ──")

    # 目标错位: pred[t] 预测 target[t+1]
    # intent 有 K 帧，用前 K-1 帧预测后 K-1 帧
    intent_a = elio_out["intent_a"][:, :-1]                         # [B, K-1, 2048]
    intent_g = elio_out["intent_g"][:, :-1]                         # [B, K-1, 2048]
    intent_f = elio_out["intent_f"][:, :-1]                         # [B, K-1, 2048]

    # ── audio head ──
    pred_audio = audio_head(intent_a)                                # [B, K-1, 512]
    tgt_audio = batch["audio"][:, 1:]                                # [B, K-1, 512]
    assert pred_audio.shape == tgt_audio.shape, \
        f"audio pred: {pred_audio.shape}, target: {tgt_audio.shape}"
    L_audio = audio_loss(pred_audio, tgt_audio)
    print(f"    audio:  pred {list(pred_audio.shape)}  loss={L_audio.item():.6f}  OK")

    # ── gaze head ──
    pred_gaze = torch.sigmoid(gaze_head(intent_g))                   # [B, K-1, 2]
    tgt_gaze = batch["gaze"][:, 1:]                                  # [B, K-1, 2]
    assert pred_gaze.shape == tgt_gaze.shape, \
        f"gaze pred: {pred_gaze.shape}, target: {tgt_gaze.shape}"
    L_gaze = gaze_loss(pred_gaze, tgt_gaze)
    print(f"    gaze:   pred {list(pred_gaze.shape)}  loss={L_gaze.item():.6f}  OK")

    # ── frame head ──
    pred_siglip_dz, pred_dino_dz = frame_head(intent_f)              # [B,K-1,196,768], [B,K-1,257,768]
    true_siglip_dz = batch["siglip"][:, 1:] - batch["siglip"][:, :-1]
    true_dino_dz = batch["dinov2"][:, 1:] - batch["dinov2"][:, :-1]
    assert pred_siglip_dz.shape == true_siglip_dz.shape, \
        f"siglip_dz pred: {pred_siglip_dz.shape}, target: {true_siglip_dz.shape}"
    assert pred_dino_dz.shape == true_dino_dz.shape, \
        f"dino_dz pred: {pred_dino_dz.shape}, target: {true_dino_dz.shape}"
    L_frame_s = frame_loss(pred_siglip_dz, true_siglip_dz)
    L_frame_d = frame_loss(pred_dino_dz, true_dino_dz)
    L_frame = L_frame_s + L_frame_d
    print(f"    frame:  siglip_dz {list(pred_siglip_dz.shape)}  "
          f"dino_dz {list(pred_dino_dz.shape)}  loss={L_frame.item():.6f}  OK")

    print_memory("after simple heads")

    # ══════════════════════════════════════════════════════
    #  阶段 B: 自回归头 (键盘 + 鼠标)
    # ══════════════════════════════════════════════════════

    print(f"\n── Phase B: Autoregressive heads ──")
    from trainer.heads import N_TYPE, N_KEY, N_BTN, MAX_EVENTS

    events_tensor = batch["events"]                                   # [B, K, max_ev, 10]
    events_mask = batch["events_mask"]                                 # [B, K, max_ev]
    K_actual = K
    max_ev_actual = events_tensor.shape[2]

    # ── 键盘 AR 头 ──
    intent_k = elio_out["intent_k"]                                   # [B, K, 2048]
    intent_k_flat = intent_k.reshape(B_actual * K_actual, 2048)      # [B*K, 2048]
    ev_flat = events_tensor.reshape(B_actual * K_actual, max_ev_actual, 10)
    mask_flat = events_mask.reshape(B_actual * K_actual, max_ev_actual)

    kb_preds = kb_head(intent_k_flat, ev_flat, mask_flat)
    assert kb_preds["type_logits"].shape == (B_actual * K_actual, max_ev_actual + 1, kb_head.n_type_stream), \
        f"kb type_logits: {kb_preds['type_logits'].shape}"
    print(f"    kb:     type_logits {list(kb_preds['type_logits'].shape)}  OK")

    L_kb = kb_head.loss(kb_preds, ev_flat, mask_flat)
    print(f"    kb:     loss={L_kb.item():.6f}  OK")

    # ── 鼠标 AR 头 ──
    intent_m = elio_out["intent_m"]                                   # [B, K, 2048]
    intent_m_flat = intent_m.reshape(B_actual * K_actual, 2048)

    mou_preds = mouse_head(intent_m_flat, ev_flat, mask_flat)
    assert mou_preds["type_logits"].shape == (B_actual * K_actual, max_ev_actual + 1, mouse_head.n_type_stream), \
        f"mouse type_logits: {mou_preds['type_logits'].shape}"
    print(f"    mouse:  type_logits {list(mou_preds['type_logits'].shape)}  OK")

    L_mouse = mouse_head.loss(mou_preds, ev_flat, mask_flat)
    print(f"    mouse:  loss={L_mouse.item():.6f}  OK")

    # ── 空事件帧测试 ──
    n_real_per_frame = mask_flat.sum(dim=1).long()                   # [B*K]
    n_zero_frames = (n_real_per_frame == 0).sum().item()
    print(f"    frames with 0 events: {n_zero_frames}/{B_actual * K_actual}  "
          f"{'OK' if n_zero_frames > 0 else '(no empty frames in this batch)'}")

    print_memory("after AR heads")

    # ══════════════════════════════════════════════════════
    #  阶段 C: 合龙 — total_loss + backward
    # ══════════════════════════════════════════════════════

    print(f"\n── Phase C: Total loss + backward ──")

    # 清零所有慢权重梯度
    elio.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)

    # 重新前向 (保留计算图)
    elio_out2 = elio(batch, llama=llama)

    # 简单头
    pred_a = audio_head(elio_out2["intent_a"][:, :-1])
    L_a = audio_loss(pred_a, batch["audio"][:, 1:])

    pred_g = torch.sigmoid(gaze_head(elio_out2["intent_g"][:, :-1]))
    L_g = gaze_loss(pred_g, batch["gaze"][:, 1:])

    pred_s_dz, pred_d_dz = frame_head(elio_out2["intent_f"][:, :-1])
    L_f = frame_loss(pred_s_dz, batch["siglip"][:, 1:] - batch["siglip"][:, :-1]) \
        + frame_loss(pred_d_dz, batch["dinov2"][:, 1:] - batch["dinov2"][:, :-1])

    # AR 头
    i_k = elio_out2["intent_k"].reshape(B_actual * K_actual, 2048)
    i_m = elio_out2["intent_m"].reshape(B_actual * K_actual, 2048)
    k_preds = kb_head(i_k, ev_flat, mask_flat)
    m_preds = mouse_head(i_m, ev_flat, mask_flat)
    L_k = kb_head.loss(k_preds, ev_flat, mask_flat)
    L_m = mouse_head.loss(m_preds, ev_flat, mask_flat)

    # ── detach 归一化总 loss ──
    losses = {"audio": L_a, "frame": L_f, "gaze": L_g, "kb": L_k, "mouse": L_m}
    print(f"    raw losses: audio={L_a.item():.4f}  frame={L_f.item():.4f}  "
          f"gaze={L_g.item():.4f}  kb={L_k.item():.4f}  mouse={L_m.item():.4f}")

    L_total = total_loss(losses)
    print(f"    total_loss (normalized): {L_total.item():.4f}")

    L_total.backward()

    print_memory("after backward")

    # ── 验证梯度 ──
    print(f"\n── Gradient check ──")

    elio_grad_ok = 0
    elio_grad_fail = 0
    for name, p in elio.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            elio_grad_ok += 1
        else:
            if elio_grad_fail < 3:
                print(f"    elio.{name}: grad={'None' if p.grad is None else 'zero'}  FAIL")
            elio_grad_fail += 1
    print(f"    ElioModel: {elio_grad_ok} params with grad, {elio_grad_fail} without")

    heads_grad_ok = 0
    heads_grad_fail = 0
    heads_fail_names: list[str] = []
    for name, p in heads.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            heads_grad_ok += 1
        else:
            heads_fail_names.append(name)
            heads_grad_fail += 1

    # 哪些是"预期无梯度":
    # - kb.btn_head: 键盘流 type 0/1 永远不触发 button (field_mask[0/1][1]=0)
    # - 若批内无键盘/鼠标事件,对应流的所有头都可能无梯度
    n_kb_ev = ((events_tensor[..., 0].long() == 0) | (events_tensor[..., 0].long() == 1)).sum().item()
    n_mou_ev = ((events_tensor[..., 0].long() >= 2) & (events_tensor[..., 0].long() <= 5)).sum().item()

    expected_none: set[str] = set()
    expected_none.add("kb.btn_head.weight")
    expected_none.add("kb.btn_head.bias")
    if n_kb_ev == 0:
        for n in heads_fail_names:
            if n.startswith("kb."):
                expected_none.add(n)
    if n_mou_ev == 0:
        for n in heads_fail_names:
            if n.startswith("mouse."):
                expected_none.add(n)

    unexpected_fails = [n for n in heads_fail_names if n not in expected_none]
    for name in heads_fail_names[:8]:  # 最多打印 8 个
        tag = " (expected)" if name in expected_none else " FAIL"
        print(f"    heads.{name}: grad=None{tag}")
    if len(heads_fail_names) > 8:
        print(f"    ... and {len(heads_fail_names) - 8} more")
    print(f"    Heads:      {heads_grad_ok} params with grad, {heads_grad_fail} without"
          f" ({len(unexpected_fails)} unexpected, n_kb={n_kb_ev}, n_mou={n_mou_ev})")

    # Llama 参数应该无梯度
    llama_params_with_grad = 0
    for name, p in llama.named_parameters():
        if p.grad is not None:
            llama_params_with_grad += 1

    if llama_params_with_grad == 0:
        print(f"    Llama:      0 params have grad  OK")
    else:
        print(f"    Llama:      {llama_params_with_grad} params have grad  FAIL")
        all_ok = False

    if elio_grad_fail > 0 or len(unexpected_fails) > 0:
        all_ok = False

    # ── 汇总 ──
    print(f"\n{'=' * 50}")
    print(f"  K × T_per_frame: {K} × {T_per_frame} = {K * T_per_frame}")
    print(f"  inputs_embeds:   [{B_actual}, {K * T_per_frame}, 2048]")
    print(f"  intents:         [{B_actual}, {K}, 5, 2048]")
    print(f"  audio loss:      {L_a.item():.4f}")
    print(f"  gaze loss:       {L_g.item():.4f}")
    print(f"  frame loss:      {L_f.item():.4f}")
    print(f"  kb loss:         {L_k.item():.4f}")
    print(f"  mouse loss:      {L_m.item():.4f}")
    print(f"  total loss:      {L_total.item():.4f}")
    print(f"  ElioModel grad:  {elio_grad_ok}/{elio_grad_ok + elio_grad_fail}")
    print(f"  Heads grad:      {heads_grad_ok}/{heads_grad_ok + heads_grad_fail}")
    print(f"  Llama grad:      {llama_params_with_grad}/{sum(1 for _ in llama.parameters())}")
    print_memory("final")

    if all_ok:
        print(f"\n  ALL CHECKS PASSED")
    else:
        print(f"\n  SOME CHECKS FAILED")

    # cleanup
    ds.close()
    del llama, elio, heads
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="验证前向骨架：数据→merge→proj→Llama→5heads→backward")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--K", type=int, default=4, help="连续帧数")
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--no-4bit", action="store_true", help="禁用 4-bit，用 bfloat16")
    args = parser.parse_args()

    if not LLAMA_PATH.exists():
        print(f"ERROR: Llama not found at {LLAMA_PATH}", file=sys.stderr)
        print("Run: python -m elio.download", file=sys.stderr)
        sys.exit(1)

    ok = verify_forward(
        device=args.device,
        batch_size=args.batch_size,
        K=args.K,
        max_batches=args.max_batches,
        use_4bit=not args.no_4bit,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
