#!/usr/bin/env python3
"""
慢权重模块：Attention Pool 压缩 + 投影层对齐 Llama-2048。

设计:
  - 全屏和焦点各自独立的 AttentionMerge（不共享 query）
  - 每路内 SigLIP + DINOv2 concat 后统一 merge → 16 token
  - 三个独立投影器映射到 Llama hidden_size=2048
  - 所有参数 requires_grad=True（慢权重，云端 meta-train 时优化）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
#  Attention Pool — 从 patch token 序列抽浓缩 token
# ═══════════════════════════════════════════════════════════

class AttentionPool(nn.Module):
    """可学习 query 通过 cross-attention 从 token 序列中抽取固定数量 token。

    输入 [B, P, D] → 输出 [B, Q, D]
    """

    def __init__(
        self,
        num_queries: int = 16,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # 可学习 query token
        self.query = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, P, D] → [B, Q, D]"""
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)          # [B, Q, D]
        out, _ = self.cross_attn(q, x, x)          # Q attends to tokens
        return self.norm(out)


# ═══════════════════════════════════════════════════════════
#  Projector — 映射到 Llama hidden_size
# ═══════════════════════════════════════════════════════════

class Projector(nn.Module):
    """两层 MLP + GELU → LayerNorm，将任意维度映射到 llama_dim=2048。"""

    def __init__(self, in_dim: int, out_dim: int = 2048, hidden_mult: int = 2):
        super().__init__()
        hidden = in_dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] → [..., out_dim]"""
        return self.net(x)


# ═══════════════════════════════════════════════════════════
#  ModalityEncoder — 单模态的 merge + project 组合
# ═══════════════════════════════════════════════════════════

class VisualEncoder(nn.Module):
    """全屏/焦点视觉编码：concat(SigLIP, DINOv2) → AttentionPool → Projector。

    输入:
      siglip: [B, K, 196, 768]
      dinov2: [B, K, 257, 768]
    输出:
      [B, K, num_queries, llama_dim]
    """

    def __init__(
        self,
        num_queries: int = 16,
        embed_dim: int = 768,
        llama_dim: int = 2048,
        attn_heads: int = 8,
        name: str = "visual",
    ):
        super().__init__()
        self.name = name
        total_tokens = 196 + 257  # SigLIP + DINOv2 concat
        self.total_tokens = total_tokens
        self.num_queries = num_queries

        self.pool = AttentionPool(
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_heads=attn_heads,
        )
        self.proj = Projector(in_dim=embed_dim, out_dim=llama_dim)

    def forward(self, siglip: torch.Tensor, dinov2: torch.Tensor) -> torch.Tensor:
        """siglip: [B, K, 196, 768], dinov2: [B, K, 257, 768] → [B, K, Q, llama_dim]"""
        B, K = siglip.shape[0], siglip.shape[1]

        # 沿 patch 维度拼接
        x = torch.cat([siglip, dinov2], dim=2)                           # [B, K, 453, 768]

        # 压平 B*K 做 batch attention
        x = x.view(B * K, self.total_tokens, 768)                       # [B*K, 453, 768]
        x = self.pool(x)                                                  # [B*K, Q, 768]
        x = self.proj(x)                                                  # [B*K, Q, llama_dim]
        x = x.view(B, K, self.num_queries, -1)                           # [B, K, Q, llama_dim]
        return x


# ═══════════════════════════════════════════════════════════
#  ElioModel — 组装所有慢权重模块
# ═══════════════════════════════════════════════════════════

class ElioModel(nn.Module):
    """多模态 → 压缩 token → 投影 → Llama 的完整慢权重包装。

    冻结部分 (外部管理):
      - Llama (4bit, requires_grad=False)
      - SigLIP / DINOv2 / CLAP 编码器 (离线预计算，不在训练循环)

    可训练部分 (本模块):
      - visual_full: AttentionPool + Projector (全屏)
      - visual_fovea: AttentionPool + Projector (焦点)
      - proj_audio: Projector (512→2048)
      - proj_action: Projector (178→2048)
    """

    def __init__(
        self,
        llama_dim: int = 2048,
        visual_queries: int = 16,
        visual_heads: int = 8,
        audio_dim: int = 512,
        action_dim: int = 178,
    ):
        super().__init__()
        self.llama_dim = llama_dim
        self.visual_queries = visual_queries

        # ── 视觉 merge + proj ──
        self.visual_full = VisualEncoder(
            num_queries=visual_queries, embed_dim=768,
            llama_dim=llama_dim, attn_heads=visual_heads,
            name="full",
        )
        self.visual_fovea = VisualEncoder(
            num_queries=visual_queries, embed_dim=768,
            llama_dim=llama_dim, attn_heads=visual_heads,
            name="fovea",
        )

        # ── 音频投影 (单向量，不需要 merge) ──
        self.proj_audio = Projector(in_dim=audio_dim, out_dim=llama_dim)

        # ── 动作投影 (单向量) ──
        self.proj_action = Projector(in_dim=action_dim, out_dim=llama_dim)

        # ── 统计参数量 ──
        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ElioModel] total params: {total:,}  trainable: {trainable:,}")

    def forward(self, batch: dict) -> torch.Tensor:
        """将一个 batch 编码为 inputs_embeds，准备喂给 Llama。

        Args:
            batch: collate_fn 输出，各张量形状 [B, K, ...]

        Returns:
            inputs_embeds: [B, K * tokens_per_frame, llama_dim]
                tokens_per_frame = 1 (audio) + Q (full) + Q (fovea) + 1 (action)
        """
        B, K = batch["siglip"].shape[0], batch["siglip"].shape[1]

        # ── 视觉编码 ──
        full_vis = self.visual_full(batch["siglip"], batch["dinov2"])          # [B, K, Q, D]
        fovea_vis = self.visual_fovea(batch["siglip_fov"], batch["dinov2_fov"])# [B, K, Q, D]

        # ── 音频投影 ──
        audio = self.proj_audio(batch["audio"])                                 # [B, K, D]
        audio = audio.unsqueeze(2)                                              # [B, K, 1, D]

        # ── 动作投影 ──
        action = self.proj_action(batch["actions"])                             # [B, K, D]
        action = action.unsqueeze(2)                                            # [B, K, 1, D]

        # ── 每帧拼接: [audio 1][full Q][fovea Q][action 1] ──
        frame_tokens = torch.cat([audio, full_vis, fovea_vis, action], dim=2)  # [B, K, T_per_frame, D]
        T_per_frame = frame_tokens.shape[2]  # 1 + Q + Q + 1

        # ── 帧顺序展开 ──
        inputs_embeds = frame_tokens.view(B, K * T_per_frame, self.llama_dim)   # [B, K*T, D]

        return inputs_embeds

    @property
    def tokens_per_frame(self) -> int:
        return 1 + self.visual_queries + self.visual_queries + 1
