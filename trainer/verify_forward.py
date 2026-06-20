#!/usr/bin/env python3
"""
最小前向验证：Dataset → merge+proj → frozen Llama → h_t → backward。

验证点:
  1. inputs_embeds 形状 [B, K*T, 2048] 正确
  2. h_t 形状 [B, K*T, 2048] 正确
  3. 前向不 OOM，打印峰值显存
  4. 反向后 proj.grad 非零、Llama.grad 为 None

用法:
    python -m trainer.verify_forward
    python -m trainer.verify_forward --batch-size 1 --K 4 --max-batches 3
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
    """加载 Llama-3.2-1B，冻结参数。

    Returns:
        model, tokenizer (tokenizer may be None if not needed)
    """
    from transformers import LlamaForCausalLM, LlamaTokenizer

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

    # 测试用：只取第一个目录的前 100 帧
    ds = ProcessedDataset(
        processed_dirs=[str(processed_dirs[0])],
        K=K,
        stride=5,  # 跳着取，覆盖更多时间跨度
    )
    # 限制为小样本
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

    # ── 3. 创建 ElioModel (慢权重) ──
    print("\n── Creating ElioModel ──")
    from trainer.model import ElioModel

    elio = ElioModel(
        llama_dim=2048,
        visual_queries=16,
        visual_heads=8,
    )
    elio = elio.to(device)
    elio.train()  # 确保 dropout/batchnorm 行为正确

    T_per_frame = elio.tokens_per_frame
    print(f"  tokens_per_frame: {T_per_frame}  (1 audio + 16 full + 16 fovea + 1 action)")

    # ── 4. 加载冻结 Llama ──
    print("\n── Loading Llama ──")
    llama = load_llama(device, use_4bit=use_4bit)

    # ── 5. 取一个 batch，前向 ──
    print(f"\n── Forward pass ({max_batches} batches) ──")
    torch.cuda.reset_peak_memory_stats()
    print_memory("before")

    all_ok = True

    for batch_idx, batch in enumerate(dl):
        if batch_idx >= max_batches:
            break

        # 移到 GPU
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.to(device)
            else:
                batch_gpu[k] = v  # list of tensors (events) — 已在 collate 处理
        batch = batch_gpu

        B_actual = batch["siglip"].shape[0]
        print(f"\n  Batch {batch_idx + 1}: B={B_actual}, K={K}")

        # ── ElioModel 前向 → inputs_embeds ──
        inputs_embeds = elio(batch)                               # [B, K*T, 2048]
        expected_seq_len = K * T_per_frame
        assert inputs_embeds.shape == (B_actual, expected_seq_len, 2048), \
            f"inputs_embeds shape: {inputs_embeds.shape}, expected ({B_actual}, {expected_seq_len}, 2048)"
        print(f"    inputs_embeds: {list(inputs_embeds.shape)}  OK")

        # ── Llama 前向 (梯度可穿过冻结 Llama 流回 proj) ──
        inputs_embeds_bf16 = inputs_embeds.to(torch.bfloat16)
        outputs = llama(inputs_embeds=inputs_embeds_bf16, output_hidden_states=True)

        # 最后一层 hidden state 作为 h_t
        h_t = outputs.hidden_states[-1]                            # [B, K*T, 2048]
        assert h_t.shape == inputs_embeds.shape[:2] + (2048,), \
            f"h_t shape: {h_t.shape}"
        print(f"    h_t:           {list(h_t.shape)}  OK")

        print_memory(f"after fwd batch {batch_idx + 1}")

    # ── 6. 反向验证 (只在最后一个 batch) ──
    print(f"\n── Backward check ──")
    print("  Running h_t.sum().backward()...")

    # 重新前向，这次保留计算图
    batch_gpu = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_gpu[k] = v.to(device)
        else:
            batch_gpu[k] = v

    elio.zero_grad(set_to_none=True)
    inputs_embeds = elio(batch_gpu)
    inputs_embeds_bf16 = inputs_embeds.to(torch.bfloat16)
    outputs = llama(inputs_embeds=inputs_embeds_bf16, output_hidden_states=True)
    h_t = outputs.hidden_states[-1]

    # 反向传播
    loss = h_t.sum()
    loss.backward()

    print_memory("after backward")

    # ── 验证梯度 ──
    proj_has_grad = False
    llama_has_grad = False

    # ElioModel 参数应该有梯度
    for name, p in elio.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            proj_has_grad = True
            if "proj" in name or "query" in name:
                print(f"    elio.{name}: grad norm={p.grad.norm().item():.6f}  OK")
        elif p.grad is None:
            print(f"    elio.{name}: grad=None  FAIL")

    # Llama 参数应该无梯度
    llama_params_with_grad = 0
    for name, p in llama.named_parameters():
        if p.grad is not None:
            llama_params_with_grad += 1
            llama_has_grad = True

    if llama_params_with_grad == 0:
        print(f"    Llama: 0/{sum(1 for _ in llama.parameters())} params have grad  OK")
    else:
        print(f"    Llama: {llama_params_with_grad} params have grad  FAIL")
        all_ok = False

    if not proj_has_grad:
        print("  FAIL: No ElioModel params received gradient!")
        all_ok = False

    # ── 汇总 ──
    print(f"\n{'=' * 50}")
    print(f"  T_per_frame: {T_per_frame}")
    print(f"  K*T_per_frame: {K * T_per_frame}")
    print(f"  inputs_embeds: [{B_actual}, {K * T_per_frame}, 2048]")
    print(f"  h_t:           [{B_actual}, {K * T_per_frame}, 2048]")
    print(f"  proj.grad:     {'non-zero OK' if proj_has_grad else 'MISSING FAIL'}")
    print(f"  Llama.grad:    {'None OK' if not llama_has_grad else 'PRESENT FAIL'}")
    print_memory("final")

    if all_ok:
        print(f"\n  ALL CHECKS PASSED")
    else:
        print(f"\n  SOME CHECKS FAILED")

    # cleanup
    ds.close()
    del llama, elio
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="验证前向骨架：数据→merge→proj→Llama→h_t")
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
