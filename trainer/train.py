#!/usr/bin/env python3
"""
RSSM 世界模型训练脚本 — 三模态：SigLIP + DINOv2 + Audio (mel spectrogram)。
保留完整 patch token 序列，不池化。

用法:
    python train.py
    python train.py --epochs 200 --batch-size 8 --seq-len 32
    python train.py --resume checkpoints/rssm_epoch0040.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "trainer" / "checkpoints"


# ═══════════════════════════════════════════════════════════
#  Token 投影：降维但保留逐 token 空间结构
# ═══════════════════════════════════════════════════════════

class TokenProjector(nn.Module):
    """将每个 token 从 768 投影到 d_model，加可学习位置编码。
    输入 [B, P, 768] → 输出 [B, P, d_model]，不做池化。
    """

    def __init__(self, num_tokens: int, in_dim: int = 768, d_model: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, P, 768] → [B, P, d_model]"""
        x = self.proj(x)                          # [B, P, d_model]
        x = x + self.pos_emb                      # 加位置编码
        return self.norm(x)


# ═══════════════════════════════════════════════════════════
#  Audio 投影：mel spectrogram → token 序列
# ═══════════════════════════════════════════════════════════

class AudioProjector(nn.Module):
    """将 mel spectrogram [B, n_mels, n_frames] 转为 token 序列 [B, P_a, d_model]。
    使用 2D 卷积做 patching，类似视觉的 patch embedding。

    默认: n_mels=64, n_frames=10 (100ms @ 16kHz, hop=10ms)
          kernel=(8,5), stride=(8,5) → P_a = 8×2 = 16 tokens
    """

    def __init__(self, n_mels: int = 64, n_frames: int = 10,
                 d_model: int = 256, kernel: tuple = (8, 5)):
        super().__init__()
        self.conv = nn.Conv2d(1, d_model, kernel_size=kernel, stride=kernel)
        # 计算输出 token 数
        h_out = (n_mels - kernel[0]) // kernel[0] + 1
        w_out = (n_frames - kernel[1]) // kernel[1] + 1
        self.num_tokens = h_out * w_out
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, n_mels, n_frames] → [B, P_a, d_model]"""
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)                  # [B, 1, n_mels, n_frames]
        x = self.conv(mel)                          # [B, d_model, h, w]
        x = x.flatten(2).transpose(1, 2)            # [B, P_a, d_model]
        x = x + self.pos_emb
        return self.norm(x)


# ═══════════════════════════════════════════════════════════
#  Cross-Attention 编码器：RSSM 隐状态查询 token 序列
# ═══════════════════════════════════════════════════════════

class CrossAttentionEncoder(nn.Module):
    """用 h_t 作为 query 对 token 序列做 cross-attention，
    输出一个融合了空间信息的观测编码向量。
    """

    def __init__(self, d_model: int = 256, h_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            kdim=d_model, vdim=d_model, batch_first=True,
        )
        self.q_proj = nn.Linear(h_dim, d_model)           # h → query
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
        )

    def forward(self, h: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """h: [B, h_dim],  tokens: [B, P, d_model] → [B, h_dim]"""
        q = self.q_proj(h).unsqueeze(1)           # [B, 1, d_model]
        out, _ = self.attn(q, tokens, tokens)      # [B, 1, d_model]
        return self.out_proj(out.squeeze(1))        # [B, h_dim]


# ═══════════════════════════════════════════════════════════
#  Token 解码器：从隐状态重建完整 token 序列
# ═══════════════════════════════════════════════════════════

class TokenDecoder(nn.Module):
    """从 [h, z] 重建视觉 token 序列。
    用一组可学习的 position query 对 latent 做 cross-attention。
    """

    def __init__(self, num_tokens: int, h_dim: int = 1024, z_dim: int = 128,
                 d_model: int = 256, in_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        # 可学习的位置 query
        self.pos_query = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
        # latent → query
        self.latent_proj = nn.Linear(h_dim + z_dim, d_model)
        # cross-attention: position queries attend to latent
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            kdim=d_model, vdim=d_model, batch_first=True,
        )
        # 输出投影: d_model → in_dim
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, in_dim),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """(h:[B,h_dim], z:[B,z_dim]) → tokens: [B, P, in_dim]"""
        B = h.shape[0]
        latent = torch.cat([h, z], dim=-1)               # [B, h_dim+z_dim]
        k = v = self.latent_proj(latent).unsqueeze(1)     # [B, 1, d_model]
        q = self.pos_query.expand(B, -1, -1)              # [B, P, d_model]
        out, _ = self.cross_attn(q, k, v)                 # [B, P, d_model]
        return self.out_proj(out)                         # [B, P, in_dim]


# ═══════════════════════════════════════════════════════════
#  Audio 解码器：从隐状态重建 mel spectrogram
# ═══════════════════════════════════════════════════════════

class AudioDecoder(nn.Module):
    """从 [h, z] 重建 mel spectrogram。
    先解码为 token 序列，再用转置卷积重建 2D mel 图。
    """

    def __init__(self, n_mels: int = 64, n_frames: int = 10,
                 h_dim: int = 1024, z_dim: int = 128,
                 d_model: int = 256, num_audio_tokens: int = 16,
                 num_heads: int = 8):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        # 位置 query（与 AudioProjector 输出的 token 数一致）
        self.pos_query = nn.Parameter(torch.randn(1, num_audio_tokens, d_model) * 0.02)
        self.latent_proj = nn.Linear(h_dim + z_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            kdim=d_model, vdim=d_model, batch_first=True,
        )
        # 转置卷积重建 mel spectrogram
        # 从 [B, P_a, d_model] reshape 为 [B, d_model, h, w] 再上采样
        h_conv = (n_mels - 8) // 8 + 1   # = 8
        w_conv = (n_frames - 5) // 5 + 1  # = 2
        self.conv_hw = (h_conv, w_conv)   # (8, 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model, kernel_size=(8, 5), stride=(8, 5)),
            nn.Conv2d(d_model, d_model // 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(d_model // 2, 1, 3, padding=1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """(h:[B,h_dim], z:[B,z_dim]) → mel: [B, n_mels, n_frames]"""
        B = h.shape[0]
        latent = torch.cat([h, z], dim=-1)
        k = v = self.latent_proj(latent).unsqueeze(1)
        q = self.pos_query.expand(B, -1, -1)
        tokens, _ = self.cross_attn(q, k, v)             # [B, P_a, d_model]
        # reshape → [B, d_model, h_conv, w_conv] → deconv
        x = tokens.transpose(1, 2).reshape(B, -1, *self.conv_hw)
        mel = self.deconv(x)                              # [B, 1, n_mels, n_frames]
        return mel.squeeze(1)                             # [B, n_mels, n_frames]


# ═══════════════════════════════════════════════════════════
#  RSSM 模型（三模态：视觉 + 视觉 + 音频）
# ═══════════════════════════════════════════════════════════

class RSSM(nn.Module):
    """Recurrent State-Space Model，观测 = patch token 序列 + mel spectrogram。

    - TokenProjector:   768→d_model (SigLIP / DINOv2)
    - AudioProjector:   mel→d_model tokens
    - CrossAttnEncoder: h_t 查询 tokens → 观测编码
    - TokenDecoder:     h+z → 重建完整 token 序列
    - AudioDecoder:     h+z → 重建 mel spectrogram
    - GRU 动力学:       h_{t-1}, z_{t-1}, a_{t-1} → h_t
    """

    def __init__(
        self,
        act_dim: int = 178,
        h_dim: int = 1024,
        z_dim: int = 128,
        d_model: int = 256,
        num_heads: int = 8,
        siglip_tokens: int = 196,
        dinov2_tokens: int = 257,
        use_audio: bool = True,
        n_mels: int = 64,
        n_frames: int = 10,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.use_audio = use_audio

        # ── 视觉管线 ──
        self.siglip_proj = TokenProjector(siglip_tokens, 768, d_model)
        self.dinov2_proj = TokenProjector(dinov2_tokens, 768, d_model)

        self.siglip_enc = CrossAttentionEncoder(d_model, h_dim, num_heads)
        self.dinov2_enc = CrossAttentionEncoder(d_model, h_dim, num_heads)

        self.siglip_dec = TokenDecoder(siglip_tokens, h_dim, z_dim, d_model, 768, num_heads)
        self.dinov2_dec = TokenDecoder(dinov2_tokens, h_dim, z_dim, d_model, 768, num_heads)

        # ── 音频管线 ──
        if use_audio:
            self.audio_proj = AudioProjector(n_mels, n_frames, d_model)
            self.audio_enc = CrossAttentionEncoder(d_model, h_dim, num_heads)
            self.audio_dec = AudioDecoder(n_mels, n_frames, h_dim, z_dim,
                                          d_model, self.audio_proj.num_tokens, num_heads)
            fusion_in = h_dim * 3  # siglip + dinov2 + audio
        else:
            fusion_in = h_dim * 2

        # 融合多模态观测编码
        self.obs_fusion = nn.Sequential(
            nn.Linear(fusion_in, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim),
        )

        # ── RSSM 核心 ──
        self.rnn = nn.GRUCell(z_dim + act_dim, h_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, 2 * z_dim),
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, 2 * z_dim),
        )

        self.action_decoder = nn.Sequential(
            nn.Linear(h_dim + z_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim // 2), nn.ReLU(),
            nn.Linear(h_dim // 2, act_dim),
        )

    # ── 分布工具 ──

    @staticmethod
    def _dist_params(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """将网络输出的 [*, 2*z_dim] 拆为 μ, σ。"""
        μ, logσ = params.chunk(2, dim=-1)
        σ = F.softplus(logσ) + 1e-4
        return μ, σ

    @staticmethod
    def _sample_z(μ: torch.Tensor, σ: torch.Tensor) -> torch.Tensor:
        """重参数化采样。"""
        eps = torch.randn_like(μ)
        return μ + σ * eps

    @staticmethod
    def _kl_divergence(μ_q, σ_q, μ_p, σ_p) -> torch.Tensor:
        """KL( N(μ_q,σ_q²) ‖ N(μ_p,σ_p²) )，逐元素求和。"""
        return (torch.log(σ_p / σ_q)
                + (σ_q.pow(2) + (μ_q - μ_p).pow(2)) / (2 * σ_p.pow(2))
                - 0.5).sum(dim=-1)

    def forward(
        self,
        siglip_tok: torch.Tensor,       # [B, T, P_s, 768]
        dinov2_tok: torch.Tensor,       # [B, T, P_d, 768]
        act: torch.Tensor,              # [B, T, act_dim]
        audio_mel: torch.Tensor | None = None,  # [B, T, n_mels, n_frames] or None
        h0: torch.Tensor | None = None,
    ) -> dict:
        B, T = act.shape[0], act.shape[1]

        # ── 投影全部时间步的视觉 token（并行） ──
        s_flat = siglip_tok.view(B * T, siglip_tok.shape[2], 768)
        d_flat = dinov2_tok.view(B * T, dinov2_tok.shape[2], 768)
        s_proj_all = self.siglip_proj(s_flat).view(B, T, siglip_tok.shape[2], self.d_model)
        d_proj_all = self.dinov2_proj(d_flat).view(B, T, dinov2_tok.shape[2], self.d_model)

        # ── 投影全部时间步的音频（并行） ──
        if self.use_audio and audio_mel is not None:
            a_flat = audio_mel.view(B * T, audio_mel.shape[2], audio_mel.shape[3])
            a_proj_all = self.audio_proj(a_flat).view(B, T, self.audio_proj.num_tokens, self.d_model)
            has_audio = True
        else:
            has_audio = False

        if h0 is None:
            h = torch.zeros(B, self.h_dim, device=act.device)
        else:
            h = h0
        z = torch.zeros(B, self.z_dim, device=act.device)

        h_list, z_list = [], []
        s_hat_list, d_hat_list, a_mel_hat_list, act_hat_list, kl_list = [], [], [], [], []

        for t in range(T):
            s_tok = s_proj_all[:, t]   # [B, Ps, d_model]
            d_tok = d_proj_all[:, t]   # [B, Pd, d_model]
            a_prev = act[:, t - 1] if t > 0 else torch.zeros(B, self.act_dim, device=act.device)

            # ── 动力学 ──
            rnn_in = torch.cat([z, a_prev], dim=-1)
            h = self.rnn(rnn_in, h)

            # ── 观测编码（保留空间信息） ──
            s_enc = self.siglip_enc(h, s_tok)       # [B, h_dim]
            d_enc = self.dinov2_enc(h, d_tok)       # [B, h_dim]
            enc_list = [s_enc, d_enc]

            if has_audio:
                a_tok = a_proj_all[:, t]             # [B, P_a, d_model]
                a_enc = self.audio_enc(h, a_tok)     # [B, h_dim]
                enc_list.append(a_enc)

            o_enc = self.obs_fusion(torch.cat(enc_list, dim=-1))  # [B, h_dim]

            # ── 先验 p(z_t | h_t)：只看历史，推理时用 ──
            prior_params = self.prior_net(h)
            μ_p, σ_p = self._dist_params(prior_params)

            # ── 后验 q(z_t | h_t, o_t)：看历史+观测，训练时用 ──
            posterior_params = self.posterior_net(torch.cat([h, o_enc], dim=-1))
            μ_q, σ_q = self._dist_params(posterior_params)

            # 从后验采样 z（训练时用观测增强）
            z = self._sample_z(μ_q, σ_q)

            # KL(posterior ‖ prior) — RSSM 的灵魂
            kl = self._kl_divergence(μ_q, σ_q, μ_p, σ_p)

            # ── 重建 ──
            s_hat = self.siglip_dec(h, z)           # [B, Ps, 768]
            d_hat = self.dinov2_dec(h, z)           # [B, Pd, 768]
            a_hat = self.action_decoder(torch.cat([h, z], dim=-1))

            h_list.append(h); z_list.append(z)
            s_hat_list.append(s_hat); d_hat_list.append(d_hat)
            act_hat_list.append(a_hat); kl_list.append(kl)

            if has_audio:
                a_mel_hat = self.audio_dec(h, z)    # [B, n_mels, n_frames]
                a_mel_hat_list.append(a_mel_hat)

        result = {
            "siglip_hat": torch.stack(s_hat_list, dim=1),    # [B, T, Ps, 768]
            "dinov2_hat": torch.stack(d_hat_list, dim=1),    # [B, T, Pd, 768]
            "act_hat":    torch.stack(act_hat_list, dim=1),  # [B, T, act_dim]
            "kl":         torch.stack(kl_list, dim=1),       # [B, T]
            "h":          torch.stack(h_list, dim=1),        # [B, T, h_dim]
        }
        if has_audio:
            result["audio_hat"] = torch.stack(a_mel_hat_list, dim=1)  # [B, T, n_mels, n_frames]

        return result

    @torch.no_grad()
    def imagine(self, h, z, act, output_audio: bool = False):
        """单步想象：用 prior_net（仅历史）采 z，不依赖观测。"""
        rnn_in = torch.cat([z, act], dim=-1)
        h_next = self.rnn(rnn_in, h)
        prior_params = self.prior_net(h_next)
        μ_p, σ_p = self._dist_params(prior_params)
        z_next = self._sample_z(μ_p, σ_p)          # prior 采样（训练过的！）
        s_hat = self.siglip_dec(h_next, z_next)
        d_hat = self.dinov2_dec(h_next, z_next)
        a_hat = self.action_decoder(torch.cat([h_next, z_next], dim=-1))
        result = (h_next, z_next, s_hat, d_hat, a_hat)
        if output_audio and self.use_audio:
            result = result + (self.audio_dec(h_next, z_next),)
        return result


# ═══════════════════════════════════════════════════════════
#  保存 / 加载
# ═══════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, scheduler, epoch, act_dim, path):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "act_dim": act_dim,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["act_dim"]


# ═══════════════════════════════════════════════════════════
#  数据集
# ═══════════════════════════════════════════════════════════

class ProcessedDataset(Dataset):
    """加载全部 patch token 和可选的 mel spectrogram。"""

    def __init__(self, data_dir: Path, seq_len: int = 32):
        self.seq_len = seq_len
        data_dir = Path(data_dir)
        print("Loading (memory-mapped, full tokens)...")
        self.siglip = np.load(str(data_dir / "visual_siglip.npy"), mmap_mode="r")
        self.dinov2 = np.load(str(data_dir / "visual_dinov2.npy"), mmap_mode="r")
        self.actions = np.load(str(data_dir / "actions.npy")).astype(np.float32)
        self.act_dim = self.actions.shape[1]
        self.N = min(self.siglip.shape[0], self.dinov2.shape[0], self.actions.shape[0])

        # 可选的音频数据
        audio_path = data_dir / "audio.npy"
        self.has_audio = audio_path.exists()
        if self.has_audio:
            self.audio = np.load(str(audio_path), mmap_mode="r")
            self.N = min(self.N, self.audio.shape[0])
            print(f"  Audio:   {self.audio.shape}   float16")
        else:
            self.audio = None

        print(f"  SigLIP:  {self.siglip.shape}  float16")
        print(f"  DINOv2:  {self.dinov2.shape}  float16")
        print(f"  actions: {self.actions.shape}")
        print(f"  N={self.N} seq={seq_len} act_dim={self.act_dim} audio={self.has_audio}")

    def __len__(self):  return max(1, self.N - self.seq_len)

    def __getitem__(self, idx):
        start = min(idx, self.N - self.seq_len - 1)
        end = start + self.seq_len
        s = torch.from_numpy(self.siglip[start:end].astype(np.float32))
        d = torch.from_numpy(self.dinov2[start:end].astype(np.float32))
        a = torch.from_numpy(self.actions[start:end])
        if self.has_audio:
            mel = torch.from_numpy(self.audio[start:end].astype(np.float32))
            return s, d, a, mel
        return s, d, a


# ═══════════════════════════════════════════════════════════
#  训练
# ═══════════════════════════════════════════════════════════

def train(data_dir, epochs=200, batch_size=8, seq_len=32, lr=3e-4, beta=0.1,
          device="cuda", resume: str | None = None, audio_weight: float = 1.0):
    ds = ProcessedDataset(data_dir, seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                    num_workers=0, pin_memory=True)

    model = RSSM(
        act_dim=ds.act_dim,
        siglip_tokens=ds.siglip.shape[1],
        dinov2_tokens=ds.dinov2.shape[1],
        use_audio=ds.has_audio,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start_epoch = 1

    # ── 断点续训 ──
    if resume:
        resume_path = Path(resume)
        if not resume_path.exists():
            print(f"ERROR: checkpoint not found: {resume_path}", file=sys.stderr)
            sys.exit(1)
        start_epoch, act_dim = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, act_dim={act_dim}")
        scheduler.T_max = epochs

    CKPT_DIR.mkdir(exist_ok=True)
    save_every = 20

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Training: epoch {start_epoch}→{epochs}, batch={batch_size}, seq={seq_len}")
    print(f"Audio: {ds.has_audio}  audio_weight={audio_weight}")
    print(f"Save every {save_every} epochs → {CKPT_DIR}\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        ep_loss = ep_obs_s = ep_obs_d = ep_act = ep_kl = 0.0
        ep_audio = 0.0

        pbar = tqdm(dl, desc=f"Epoch {epoch:3d}/{epochs}", ncols=120, leave=False)
        for batch in pbar:
            has_audio_batch = ds.has_audio and len(batch) == 4
            if has_audio_batch:
                s_tok, d_tok, act, audio_mel = [x.to(device) for x in batch]
            else:
                s_tok, d_tok, act = [x.to(device) for x in batch]
                audio_mel = None

            out = model(s_tok, d_tok, act, audio_mel)

            loss_s = F.mse_loss(out["siglip_hat"], s_tok)
            loss_d = F.mse_loss(out["dinov2_hat"], d_tok)
            loss_act = F.mse_loss(out["act_hat"], act)
            loss_kl = out["kl"].mean()
            loss = loss_s + loss_d + loss_act + beta * loss_kl

            if has_audio_batch and "audio_hat" in out:
                loss_audio = F.mse_loss(out["audio_hat"], audio_mel)
                loss = loss + audio_weight * loss_audio
                ep_audio += loss_audio.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            ep_loss += loss.item(); ep_obs_s += loss_s.item()
            ep_obs_d += loss_d.item(); ep_act += loss_act.item()
            ep_kl  += loss_kl.item()
            postfix = {"loss": f"{loss.item():.4f}", "act": f"{loss_act.item():.4f}",
                       "kl": f"{loss_kl.item():.4f}"}
            if has_audio_batch and "audio_hat" in out:
                postfix["aud"] = f"{loss_audio.item():.4f}"
            pbar.set_postfix(postfix)

        scheduler.step()
        n = len(dl)
        line = (f"Epoch {epoch:3d} | loss={ep_loss/n:.4f}  "
                f"siglip={ep_obs_s/n:.4f}  dinov2={ep_obs_d/n:.4f}  "
                f"act={ep_act/n:.4f}  kl={ep_kl/n:.4f}")
        if ds.has_audio:
            line += f"  audio={ep_audio/n:.4f}"
        line += f"  lr={scheduler.get_last_lr()[0]:.2e}"
        print(line)

        # 周期保存
        if epoch % save_every == 0 or epoch == epochs:
            path = CKPT_DIR / f"rssm_epoch{epoch:04d}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, ds.act_dim, path)
            save_checkpoint(model, optimizer, scheduler, epoch, ds.act_dim,
                            CKPT_DIR / "rssm_latest.pt")
            print(f"  → {path}")


def main():
    parser = argparse.ArgumentParser(description="RSSM token-level training (multi-modal)")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--audio-weight", type=float, default=1.0,
                        help="audio reconstruction loss 权重")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练 (e.g. --resume checkpoints/rssm_epoch0040.pt)")
    args = parser.parse_args()

    if args.dataset:
        data_dir = Path(args.dataset)
    else:
        runs = sorted((PROJECT_ROOT / "dataset").glob("*/processed/"),
                      key=lambda p: p.parent.name)
        if not runs:
            print("ERROR: 找不到 dataset/*/processed/", file=sys.stderr); sys.exit(1)
        data_dir = runs[-1]
        print(f"Auto dataset: {data_dir}")

    train(data_dir, args.epochs, args.batch_size, args.seq_len,
          args.lr, args.beta, args.device, args.resume, args.audio_weight)


if __name__ == "__main__":
    main()
