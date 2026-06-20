#!/usr/bin/env python3
"""
持续学习训练器 — 复用 train.py 的 RSSM。

核心: 数据按时间顺序"流式"到达 → 进有上限的 replay buffer →
      每来一批新数据, 从 buffer 随机采样做若干梯度步。
      这就是在线版的 shuffle, 避免灾难性遗忘。

用法:
    python continual_train.py
    python continual_train.py --buffer-size 8000 --steps-per-ingest 4
"""

import argparse, sys
from pathlib import Path
from collections import deque
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train import RSSM, PROJECT_ROOT, CKPT_DIR, save_checkpoint


# ═══════════════════════════════════════════════════════════
#  Replay Buffer — 有上限, 存 latent 序列, 随机采样
# ═══════════════════════════════════════════════════════════

class ReplayBuffer:
    """存放 (siglip, dinov2, action, audio?) 的定长序列片段。
    满了就丢最老的 (滚动覆盖)。
    """

    def __init__(self, capacity: int, device: str, has_audio: bool = False):
        self.buf = deque(maxlen=capacity)
        self.device = device
        self.has_audio = has_audio

    def add(self, s, d, a, mel=None):
        entry = (
            s.to(torch.float16).cpu(),
            d.to(torch.float16).cpu(),
            a.cpu(),
            mel.to(torch.float16).cpu() if mel is not None else None,
        )
        self.buf.append(entry)

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        s = torch.stack([b[0] for b in batch]).to(self.device, torch.float32)
        d = torch.stack([b[1] for b in batch]).to(self.device, torch.float32)
        a = torch.stack([b[2] for b in batch]).to(self.device, torch.float32)
        if self.has_audio and batch[0][3] is not None:
            mel = torch.stack([b[3] for b in batch]).to(self.device, torch.float32)
            return s, d, a, mel
        return s, d, a


# ═══════════════════════════════════════════════════════════
#  流式数据源 — 把数据集当成"实时到达"的流
# ═══════════════════════════════════════════════════════════

class FrameStream:
    """按时间顺序产出定长序列片段, 模拟在线场景下数据逐段到达。"""

    def __init__(self, data_dir: Path, seq_len: int):
        self.seq_len = seq_len
        print("Loading latents into RAM (float16)...")
        self.siglip = np.load(str(data_dir / "visual_siglip.npy")).astype(np.float16)
        self.dinov2 = np.load(str(data_dir / "visual_dinov2.npy")).astype(np.float16)
        self.actions = np.load(str(data_dir / "actions.npy")).astype(np.float32)
        self.N = min(len(self.siglip), len(self.dinov2), len(self.actions))
        self.act_dim = self.actions.shape[1]

        audio_path = data_dir / "audio.npy"
        self.has_audio = audio_path.exists()
        if self.has_audio:
            self.audio = np.load(str(audio_path)).astype(np.float16)
            self.N = min(self.N, len(self.audio))
            print(f"  Audio {self.audio.shape}")
        else:
            self.audio = None

        print(f"  SigLIP {self.siglip.shape}  DINOv2 {self.dinov2.shape}  "
              f"actions {self.actions.shape}  N={self.N}  audio={self.has_audio}")

    def segments(self):
        """逐段顺序产出 (模拟时间流)。"""
        for start in range(0, self.N - self.seq_len, self.seq_len):
            end = start + self.seq_len
            s = torch.from_numpy(self.siglip[start:end].astype(np.float32))
            d = torch.from_numpy(self.dinov2[start:end].astype(np.float32))
            a = torch.from_numpy(self.actions[start:end])
            if self.has_audio:
                mel = torch.from_numpy(self.audio[start:end].astype(np.float32))
                yield s, d, a, mel
            else:
                yield s, d, a


# ═══════════════════════════════════════════════════════════
#  一个梯度步
# ═══════════════════════════════════════════════════════════

def train_step(model, opt, scaler, s, d, a, beta, device, mel=None, audio_weight=1.0):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(s, d, a, mel)
        loss_s = F.mse_loss(out["siglip_hat"], s)
        loss_d = F.mse_loss(out["dinov2_hat"], d)
        loss_a = F.mse_loss(out["act_hat"], a)
        loss_kl = out["kl"].mean()
        loss = loss_s + loss_d + loss_a + beta * loss_kl
        loss_audio = None
        if mel is not None and "audio_hat" in out:
            loss_audio = F.mse_loss(out["audio_hat"], mel)
            loss = loss + audio_weight * loss_audio

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    scaler.step(opt)
    scaler.update()
    return (loss.item(), loss_a.item(), loss_kl.item(),
            loss_audio.item() if loss_audio is not None else 0.0)


# ═══════════════════════════════════════════════════════════
#  持续学习主循环
# ═══════════════════════════════════════════════════════════

def continual_train(data_dir, buffer_size=8000, batch_size=8, seq_len=32,
                    steps_per_ingest=4, warmup=200, lr=3e-4, beta=0.1,
                    audio_weight=1.0, passes=3, device="cuda"):
    stream = FrameStream(data_dir, seq_len)
    buffer = ReplayBuffer(buffer_size, device, has_audio=stream.has_audio)

    model = RSSM(
        act_dim=stream.act_dim,
        siglip_tokens=stream.siglip.shape[1],
        dinov2_tokens=stream.dinov2.shape[1],
        use_audio=stream.has_audio,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    CKPT_DIR.mkdir(exist_ok=True)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Audio: {stream.has_audio}  audio_weight={audio_weight}")
    print(f"Buffer={buffer_size} seq={seq_len} batch={batch_size} "
          f"steps/ingest={steps_per_ingest}\n")

    step = 0
    model.train()
    for p in range(passes):
        pbar = tqdm(stream.segments(), desc=f"Pass {p+1}/{passes}", ncols=120)
        for batch_data in pbar:
            if stream.has_audio and len(batch_data) == 4:
                s, d, a, mel = batch_data
                buffer.add(s, d, a, mel)
            else:
                s, d, a = batch_data
                buffer.add(s, d, a)

            if len(buffer) < warmup:
                continue

            for _ in range(steps_per_ingest):
                if stream.has_audio:
                    bs, bd, ba, bmel = buffer.sample(batch_size)
                else:
                    bs, bd, ba = buffer.sample(batch_size)
                    bmel = None
                loss, la, kl, laudio = train_step(
                    model, opt, scaler, bs, bd, ba, beta, device, bmel, audio_weight)
                step += 1

            postfix = {"buf": len(buffer), "loss": f"{loss:.3f}",
                       "act": f"{la:.3f}", "kl": f"{kl:.3f}", "step": step}
            if stream.has_audio:
                postfix["aud"] = f"{laudio:.3f}"
            pbar.set_postfix(postfix)

            if step % 2000 == 0 and step > 0:
                save_checkpoint(model, opt, opt, p, stream.act_dim,
                                CKPT_DIR / "rssm_continual_latest.pt")

    save_checkpoint(model, opt, opt, passes, stream.act_dim,
                    CKPT_DIR / "rssm_continual_final.pt")
    print(f"\nDone. total grad steps={step}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--buffer-size", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--steps-per-ingest", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--audio-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.dataset:
        data_dir = Path(args.dataset)
    else:
        runs = sorted((PROJECT_ROOT / "dataset").glob("*/processed/"),
                      key=lambda p: p.parent.name)
        if not runs:
            print("ERROR: 找不到 dataset/*/processed/", file=sys.stderr); sys.exit(1)
        data_dir = runs[-1]
        print(f"Auto dataset: {data_dir}")

    continual_train(data_dir, args.buffer_size, args.batch_size, args.seq_len,
                    args.steps_per_ingest, args.warmup, args.lr, args.beta,
                    args.audio_weight, args.passes, args.device)


if __name__ == "__main__":
    main()
