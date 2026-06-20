#!/usr/bin/env python3
"""
Step 5B/6 训练循环: 每个 session 从头滚到尾, W_fast 连续, 每 K 帧 TBPTT 截断。

用法:
    python -m trainer.train --max-sessions 1 --log-every 10
    python -m trainer.train --K 4 --epochs 1 --lr 1e-4
"""

import argparse
import gc
import time
from pathlib import Path

import torch
import torch.nn as nn

from trainer.dataset import ProcessedDataset, collate_fn
from trainer.model import ElioModel
from trainer.ttt import TTTMemory
from trainer.heads import SimpleHead, FrameHead, AutoregHead
from trainer.loop import run_ttt_segment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLAMA_PATH = PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct-abliterated"


def load_llama(device: str, use_4bit: bool = True):
    """加载 Llama-3.2-1B，冻结参数。"""
    from transformers import LlamaForCausalLM

    print(f"Loading Llama from: {LLAMA_PATH}")
    print(f"  4bit={use_4bit}")

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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Llama params: {total_params:,}  (all frozen)")
    return model


def print_memory(msg: str = ""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        p = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  [MEM {msg}] allocated={a:.2f}GB  reserved={r:.2f}GB  peak={p:.2f}GB")


def build_models(device, llama_dim=2048):
    elio = ElioModel(llama_dim=llama_dim).to(device)
    elio.train()
    ttt = TTTMemory(llama_dim=llama_dim, d_mem=256).to(device)
    ttt.train()
    heads = nn.ModuleDict({
        "audio": SimpleHead(2048, 512, 1024),
        "gaze":  SimpleHead(2048, 2, 1024),
        "frame": FrameHead(),
        "kb":    AutoregHead(stream_type="keyboard"),
        "mouse": AutoregHead(stream_type="mouse"),
    }).to(device)
    heads.train()
    return elio, ttt, heads


def save_checkpoint(path, elio, ttt, heads, opt, step):
    torch.save({
        "elio": elio.state_dict(),
        "ttt": ttt.state_dict(),
        "heads": heads.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
    }, path)
    print(f"  saved {path}")


def train(
    device="cuda", K=4, epochs=1, lr=1e-4,
    use_4bit=True, use_checkpoint=True,
    grad_clip=1.0, log_every=50,
    max_sessions=None,
):
    # --- 数据 ---
    data_dir = PROJECT_ROOT / "data" / "processed"
    processed_dirs = sorted(data_dir.glob("*/processed"))
    processed_dirs = [str(d) for d in processed_dirs if (d / "spec.json").exists()]
    if not processed_dirs:
        print("ERROR: no processed data found", file=__import__("sys").stderr)
        return

    ds = ProcessedDataset(processed_dirs=processed_dirs, K=K, stride=1)
    n_sess = min(len(ds.session_lengths), max_sessions or len(ds.session_lengths))
    print(f"Sessions: {n_sess}/{len(ds.session_lengths)}, lengths: {ds.session_lengths[:n_sess]}")
    total_frames = sum(ds.session_lengths[:n_sess])
    total_segs = sum((N - 1 + K - 1) // K for N in ds.session_lengths[:n_sess])
    print(f"Total frames: {total_frames:,}, estimated segments: {total_segs:,}")

    # --- 模型 ---
    elio, ttt, heads = build_models(device)
    llama = load_llama(device, use_4bit=use_4bit)

    # 只优化慢权重 (elio + ttt + heads)，Llama 冻结
    params = list(elio.parameters()) + list(ttt.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    n_trainable = sum(p.numel() for p in params if p.requires_grad)
    print(f"Trainable (slow weights): {n_trainable:,}")

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    global_step = 0

    for epoch in range(epochs):
        for si in range(n_sess):
            N = ds.session_lengths[si]
            W = ttt.init_state(B=1, device=device)          # session 起点重置黑板
            torch.cuda.reset_peak_memory_stats()
            t_start = time.time()
            running = {k: 0.0 for k in ["audio", "gaze", "frame", "kb", "mouse"]}
            n_seg = 0

            # --- K 步截断从头滚到尾 ---
            for t0 in range(0, N - 1, K):
                seg = ds.get_segment(si, t0)
                batch = collate_fn([seg])                      # [1, K, ...]
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}

                outer, raw, W_new = run_ttt_segment(
                    elio, ttt, llama, heads, batch, device,
                    detach_state=W, use_checkpoint=use_checkpoint,
                )

                opt.zero_grad(set_to_none=True)
                outer.backward()
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()

                # TBPTT 截断: 数值继续滚, 图切断
                W = W_new.detach().requires_grad_(use_checkpoint)

                for k in running:
                    running[k] += raw[k]
                n_seg += 1
                global_step += 1

                if global_step % log_every == 0:
                    avg = {k: running[k] / n_seg for k in running}
                    print(f"  ep{epoch} sess{si} t0={t0}/{N} step{global_step} | "
                          f"a={avg['audio']:.3f} g={avg['gaze']:.3f} "
                          f"f={avg['frame']:.3f} kb={avg['kb']:.3f} m={avg['mouse']:.3f}")

            dt = time.time() - t_start
            print_memory(f"sess{si} done")
            print(f"  sess{si}: {n_seg} segs, {dt:.1f}s, {N / dt:.1f} fps")

            # 每 session 存 checkpoint
            ckpt_path = ckpt_dir / f"ep{epoch}_sess{si}.pt"
            save_checkpoint(ckpt_path, elio, ttt, heads, opt, global_step)

    ds.close()
    print(f"\nTraining done. {global_step} total steps.")


def main():
    parser = argparse.ArgumentParser(description="Elio V3 TBPTT 训练")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--K", type=int, default=4, help="TBPTT 截断帧数")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true", help="禁用 activation checkpoint")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    if not LLAMA_PATH.exists():
        print(f"ERROR: Llama not found at {LLAMA_PATH}", file=__import__("sys").stderr)
        __import__("sys").exit(1)

    train(
        device=args.device, K=args.K, epochs=args.epochs, lr=args.lr,
        use_4bit=not args.no_4bit, use_checkpoint=not args.no_checkpoint,
        max_sessions=args.max_sessions, log_every=args.log_every,
        grad_clip=args.grad_clip,
    )


if __name__ == "__main__":
    main()
