#!/usr/bin/env python3
"""
Step 4 预测头：IntentPool + 5 个 Head + 总 Loss。

包含:
  - IntentPool: 5-query cross-attn，每帧 34→5 意图向量
  - SimpleHead: 简单 MLP 头 (audio / gaze)
  - FrameHead: 完整 patch 残差预测 (全屏 SigLIP + DINOv2)
  - EventEmbedder: 10-dim 事件 → d_model
  - AutoregHead: Transformer 自回归头 (键盘 / 鼠标各一)
  - 各 loss 函数 + detach 归一化总 loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════
#  数据扫描得出的常量
# ═══════════════════════════════════════════════════════════

N_TYPE = 6        # type_id ∈ {0..5}
N_KEY = 85        # key_id: 0..82 → shift(+1), plus -1 sentinel at index 0
N_BTN = 5         # button_id: {-1,0,1,2,3} → shift(+1), sentinel at index 0
MAX_EVENTS = 10   # max events per frame (实测 max=10)

# 连续字段 z-score 统计量 (7 维: x, y, dx, dy, path_len, scroll_dy, dt_ms)
CONT_MEAN = [0.443, 0.531, 0.0, 0.0, 0.018, -0.043, 63.769]
CONT_STD  = [0.222, 0.239, 0.041, 0.049, 0.049, 0.349, 31.651]

# 字段有效性 mask: 按 type_id 索引
# 字段顺序: key_id, button_id, x, y, dx, dy, path_len, scroll_dy, dt_ms
# 1=该字段对该事件类型有意义, 0=应屏蔽 loss
FIELD_MASK = {
    0: [1, 0, 1, 1, 0, 0, 0, 0, 1],   # key_down
    1: [1, 0, 1, 1, 0, 0, 0, 0, 1],   # key_up
    2: [0, 1, 1, 1, 0, 0, 0, 0, 1],   # mouse_down
    3: [0, 1, 1, 1, 0, 0, 0, 0, 1],   # mouse_up
    4: [0, 0, 1, 1, 1, 1, 1, 0, 1],   # move
    5: [0, 0, 1, 1, 0, 0, 0, 1, 1],   # scroll
}

# 键盘流 type 集合, 鼠标流 type 集合
KB_TYPES  = {0, 1}
MOU_TYPES = {2, 3, 4, 5}


# ═══════════════════════════════════════════════════════════
#  IntentPool — 5 query cross-attn 聚合
# ═══════════════════════════════════════════════════════════

class IntentPool(nn.Module):
    """5 个可学习 query，从每帧 34 个 h 聚合出 5 个意图向量。

    输入: h_frames [B, K, T=34, D=2048]
    输出: intents  [B, K, 5, 2048]
    """

    def __init__(self, dim: int = 2048, num_heads: int = 8, num_intents: int = 5):
        super().__init__()
        self.num_intents = num_intents
        self.queries = nn.Parameter(torch.randn(num_intents, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h_frames: torch.Tensor) -> torch.Tensor:
        """h_frames: [B, K, T, D] → [B, K, 5, D]"""
        B, K, T, D = h_frames.shape
        h = h_frames.reshape(B * K, T, D)                     # [B*K, 34, D]
        q = self.queries.unsqueeze(0).expand(B * K, -1, -1)   # [B*K, 5, D]
        out, _ = self.cross_attn(q, h, h)                      # [B*K, 5, D]
        out = self.norm(out)
        return out.reshape(B, K, self.num_intents, D)          # [B, K, 5, D]


# ═══════════════════════════════════════════════════════════
#  SimpleHead — audio / gaze 共用 MLP
# ═══════════════════════════════════════════════════════════

class SimpleHead(nn.Module):
    """两层 MLP: [..., in_dim] → [..., out_dim]。

    用于 audio_head (out=512) 和 gaze_head (out=2)。
    gaze_head 输出过 sigmoid 约束到 [0,1]。
    """

    def __init__(self, in_dim: int = 2048, out_dim: int = 512, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] → [..., out_dim]"""
        return self.net(x)


# ═══════════════════════════════════════════════════════════
#  PatchMLP — 共享的 per-patch 渲染器
# ═══════════════════════════════════════════════════════════

class PatchMLP(nn.Module):
    """单个 patch seed → 完整 embedding 的共享 MLP。

    所有 patch 共享同一套权重："把描述变成画面"对每个 patch 同构。
    """

    def __init__(self, in_dim: int = 128, hidden: int = 512, out_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] → [..., out_dim]"""
        return self.net(x)


# ═══════════════════════════════════════════════════════════
#  FrameHead — 全屏 patch 残差 (SigLIP 196×768 + DINOv2 257×768)
# ═══════════════════════════════════════════════════════════

class FrameHead(nn.Module):
    """预测完整 patch 残差 Δz = z_{t+1} - z_t。

    低秩分解: intent [2048] → per-patch seed [P, 256] → 共享 MLP(256→1024→768) → [P, 768]。
    每个子头约 104M 参数 (vs 原 308M/404M 全连接)。

    两个独立子头，分别预测 SigLIP 196×768 和 DINOv2 257×768。
    """

    def __init__(self, in_dim: int = 2048, seed_dim: int = 256,
                 mlp_hidden: int = 1024, out_dim: int = 768):
        super().__init__()
        self.seed_dim = seed_dim
        self.out_dim = out_dim

        # SigLIP 子头: intent → 196 个 seed → 共享 MLP → [196, 768]
        self.siglip_seed_proj = nn.Linear(in_dim, 196 * seed_dim)
        self.siglip_mlp = PatchMLP(seed_dim, mlp_hidden, out_dim)

        # DINOv2 子头: intent → 257 个 seed → 共享 MLP → [257, 768]
        self.dino_seed_proj = nn.Linear(in_dim, 257 * seed_dim)
        self.dino_mlp = PatchMLP(seed_dim, mlp_hidden, out_dim)

    def forward(self, intent_f: torch.Tensor):
        """intent_f: [B, K, 2048] → siglip_dz [B,K,196,768], dino_dz [B,K,257,768]"""
        B, K = intent_f.shape[0], intent_f.shape[1]

        # SigLIP: intent → seeds → reshape → MLP
        siglip_seeds = self.siglip_seed_proj(intent_f)                 # [B, K, 196*128]
        siglip_seeds = siglip_seeds.view(B, K, 196, self.seed_dim)    # [B, K, 196, 128]
        siglip_dz = self.siglip_mlp(siglip_seeds)                     # [B, K, 196, 768]

        # DINOv2: 同上
        dino_seeds = self.dino_seed_proj(intent_f)                     # [B, K, 257*128]
        dino_seeds = dino_seeds.view(B, K, 257, self.seed_dim)        # [B, K, 257, 128]
        dino_dz = self.dino_mlp(dino_seeds)                           # [B, K, 257, 768]

        return siglip_dz, dino_dz


def frame_loss(pred_dz: torch.Tensor, true_dz: torch.Tensor) -> torch.Tensor:
    """变化区加权 MSE: 真值 Δz 绝对值大的 patch 权重高。

    Args:
        pred_dz: [B, K, P, 768] 预测残差
        true_dz: [B, K, P, 768] 真值残差
    Returns:
        scalar loss
    """
    err = (pred_dz - true_dz) ** 2                                # [B, K, P, 768]
    weight = 1.0 + true_dz.abs().mean(-1, keepdim=True)           # [B, K, P, 1]
    return (err * weight).mean()


# ═══════════════════════════════════════════════════════════
#  EventEmbedder — 10-dim 事件 → d_model
# ═══════════════════════════════════════════════════════════

class EventEmbedder(nn.Module):
    """10 维事件编码 → d_model。

    离散字段 (type, key, btn) 查 embedding 表，连续字段 (7 个) 过 Linear。
    拼接后投影到 d_model。
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        D = d_model
        self.d_model = D

        self.type_emb = nn.Embedding(N_TYPE, D // 4)       # 6 → 128
        self.key_emb  = nn.Embedding(N_KEY, D // 2)        # 85 → 256
        self.btn_emb  = nn.Embedding(N_BTN, D // 4)        # 5 → 128

        self.cont_proj = nn.Linear(7, D // 2)               # 7 → 256

        # 拼接后投影: D//4 + D//2 + D//4 + D//2 = 1.5*D → D
        concat_dim = D // 4 + D // 2 + D // 4 + D // 2     # = 1.5 * D
        self.out_proj = nn.Linear(concat_dim, D)

        # 连续字段归一化 (buffer 不参与训练)
        self.register_buffer("cont_mean", torch.tensor(CONT_MEAN, dtype=torch.float32))
        self.register_buffer("cont_std", torch.tensor(CONT_STD, dtype=torch.float32))

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """events: [..., 10] → [..., d_model]

        events[..., 0] = type_id   (int, 0..5)
        events[..., 1] = key_id    (int, -1..82) → +1 shift
        events[..., 2] = button_id (int, -1..3)  → +1 shift
        events[..., 3:10] = continuous (7 dims)
        """
        *batch_dims, _ = events.shape

        type_id = events[..., 0].long()                        # [...]
        key_id  = (events[..., 1].long() + 1).clamp(0, N_KEY - 1)   # shift -1→0
        btn_id  = (events[..., 2].long() + 1).clamp(0, N_BTN - 1)   # shift -1→0
        cont    = events[..., 3:10]                            # [..., 7]

        # z-score 归一化
        cont = (cont - self.cont_mean) / (self.cont_std.clamp(min=1e-6))

        t_emb = self.type_emb(type_id)                         # [..., D//4]
        k_emb = self.key_emb(key_id)                           # [..., D//2]
        b_emb = self.btn_emb(btn_id)                           # [..., D//4]
        c_emb = self.cont_proj(cont)                           # [..., D//2]

        x = torch.cat([t_emb, k_emb, b_emb, c_emb], dim=-1)   # [..., 1.5*D]
        return self.out_proj(x)                                 # [..., D]


# ═══════════════════════════════════════════════════════════
#  AutoregHead — Transformer 自回归事件生成
# ═══════════════════════════════════════════════════════════

class AutoregHead(nn.Module):
    """decoder-only transformer 自回归生成事件序列。

    键盘或鼠标各一个实例，type 集合不同:
      - 键盘: stream_types = {0, 1}
      - 鼠标: stream_types = {2, 3, 4, 5}

    Intent 作为 decoder memory (cross-attn)，BOS 是序列起点。
    Teacher forcing: [BOS, ev0, ..., ev_{n-1}] → [ev0, ..., ev_{n-1}, EOS]。
    """

    def __init__(
        self,
        intent_dim: int = 2048,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        max_len: int = MAX_EVENTS,
        stream_type: str = "keyboard",
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.stream_type = stream_type
        self.stream_types = KB_TYPES if stream_type == "keyboard" else MOU_TYPES
        self.n_type_stream = len(self.stream_types)

        # Intent 投影: 做 decoder memory
        self.intent_proj = nn.Linear(intent_dim, d_model)

        # 事件嵌入
        self.event_embed = EventEmbedder(d_model)

        # 位置编码 + BOS
        self.pos_embed = nn.Parameter(torch.randn(max_len + 1, d_model) * 0.02)
        self.bos = nn.Parameter(torch.randn(d_model) * 0.02)

        # Transformer decoder
        layer = nn.TransformerDecoderLayer(
            d_model, nhead, batch_first=True,
            dim_feedforward=d_model * 4, dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers)

        # 输出分支
        self.type_head = nn.Linear(d_model, self.n_type_stream)   # 仅本流 type
        self.key_head  = nn.Linear(d_model, N_KEY)                # key_id 全量
        self.btn_head  = nn.Linear(d_model, N_BTN)                # button_id 全量
        self.cont_head = nn.Linear(d_model, 7)                    # 连续 7 维
        self.eos_head  = nn.Linear(d_model, 1)                    # EOS 二分类

        # type_id 全局 → 流内索引的映射 (用于 type 分类 loss)
        type_to_stream_idx = {}
        for i, t in enumerate(sorted(self.stream_types)):
            type_to_stream_idx[t] = i
        self.register_buffer(
            "type_to_stream_idx",
            torch.tensor([type_to_stream_idx.get(t, -1) for t in range(N_TYPE)],
                         dtype=torch.long),
            persistent=False,
        )

        self._causal_mask: torch.Tensor | None = None

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask is None or self._causal_mask.shape[0] < seq_len:
            self._causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device),
                diagonal=1,
            )
        return self._causal_mask[:seq_len, :seq_len]

    def forward(
        self,
        intent: torch.Tensor,              # [B*K, 2048]
        events: torch.Tensor,              # [B*K, max_ev, 10]  padded
        events_mask: torch.Tensor,         # [B*K, max_ev]  True=real event
    ) -> dict:
        """Teacher forcing 前向。

        Returns:
            dict with:
              type_logits: [B*K, max_ev+1, n_type_stream]
              key_logits:  [B*K, max_ev+1, N_KEY]
              btn_logits:  [B*K, max_ev+1, N_BTN]
              cont:        [B*K, max_ev+1, 7]
              eos_logits:  [B*K, max_ev+1, 1]
        """
        N_frames = intent.shape[0]          # B*K
        max_ev = events.shape[1]
        device = intent.device

        # ── Intent → memory ──
        memory = self.intent_proj(intent).unsqueeze(1)         # [N, 1, D]

        # ── 每帧真实事件数 ──
        n_real = events_mask.sum(dim=1).long()                  # [N]

        # ── 构建 tgt 序列 [N, max_ev+1, D] ──
        # position 0: BOS
        bos = self.bos.unsqueeze(0).unsqueeze(0).expand(N_frames, 1, -1)  # [N, 1, D]

        # positions 1..max_ev: embedded events
        ev_emb = self.event_embed(events)                       # [N, max_ev, D]

        tgt_seq = torch.cat([bos, ev_emb], dim=1)               # [N, max_ev+1, D]
        tgt_seq = tgt_seq + self.pos_embed[:max_ev + 1, :]      # add positional embedding

        # ── tgt_key_padding_mask: mask positions > n (not n+1!) ──
        # position i in tgt corresponds to prediction i (which predicts event_i or EOS)
        # For n events: positions 0..n are valid, n+1..max_ev are padding
        #   pred[0..n-1] predicts events 0..n-1
        #   pred[n]     predicts EOS
        # tgt at position n+1 (BOS padding) → pred[n+1] is masked
        pad_mask = torch.arange(max_ev + 1, device=device).unsqueeze(0)  # [1, max_ev+1]
        pad_mask = pad_mask > n_real.unsqueeze(1)                # [N, max_ev+1]
        # For n=0: mask positions 1..max_ev (only position 0 = BOS → EOS is valid)
        # For n=max_ev: mask position max_ev (all positions valid, last predicts EOS)

        # ── Decoder ──
        causal_mask = self._get_causal_mask(max_ev + 1, device)

        out = self.decoder(
            tgt=tgt_seq,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )                                                        # [N, max_ev+1, D]

        # ── 各输出分支 ──
        return {
            "type_logits": self.type_head(out),                  # [N, max_ev+1, n_type_stream]
            "key_logits":  self.key_head(out),                   # [N, max_ev+1, N_KEY]
            "btn_logits":  self.btn_head(out),                   # [N, max_ev+1, N_BTN]
            "cont":        self.cont_head(out),                  # [N, max_ev+1, 7]
            "eos_logits":  self.eos_head(out),                   # [N, max_ev+1, 1]
        }

    def loss(
        self,
        preds: dict,                        # 来自 self.forward()
        events: torch.Tensor,              # [N, max_ev, 10]
        events_mask: torch.Tensor,         # [N, max_ev]  True=real event
    ) -> torch.Tensor:
        """计算自回归混合 loss (teacher forcing 后)。

        字段有效性 mask 按每个事件自身的 type_id 查询 FIELD_MASK。
        """
        N, max_ev = events.shape[0], events.shape[1]
        device = events.device
        n_real = events_mask.sum(dim=1).long()                   # [N]

        # ── 构建目标 ──
        # Events at positions 0..n-1
        target_events = events                                    # [N, max_ev, 10]
        target_type_id = events[..., 0].long()                   # [N, max_ev]

        # EOS targets: position i should predict EOS iff i == n (first position after all events)
        eos_target = torch.zeros(N, max_ev + 1, device=device)
        # For n=0: position 0 is EOS
        # For n>0: position n is EOS
        eos_target[torch.arange(N, device=device), n_real] = 1.0
        eos_target = eos_target.unsqueeze(-1)                     # [N, max_ev+1, 1]

        # ── 预测位置 mask: positions 0..n are valid ──
        valid_pos = torch.arange(max_ev + 1, device=device).unsqueeze(0) <= n_real.unsqueeze(1)
        valid_pos = valid_pos.float()                             # [N, max_ev+1]
        # Exclude position max_ev if n == max_ev (n_real can be max_ev, position max_ev is EOS)
        # valid_pos already handles this: if n_real=10, positions 0..10 are valid
        # but events only go up to position 9. Position 10 is EOS-only.
        # For loss on events at position i: need i < n_real

        # ── Event prediction loss at positions 0..max_ev-1 ──
        # preds[k] at position i predicts event_i (for i < n_real)
        # preds[k] at position n_real predicts EOS (no event target)

        pred_type = preds["type_logits"][:, :max_ev]              # [N, max_ev, n_type_stream]
        pred_key  = preds["key_logits"][:, :max_ev]               # [N, max_ev, N_KEY]
        pred_btn  = preds["btn_logits"][:, :max_ev]               # [N, max_ev, N_BTN]
        pred_cont = preds["cont"][:, :max_ev]                      # [N, max_ev, 7]

        # 只取 real event 位置
        ev_mask = events_mask.float()                              # [N, max_ev]

        # ── Type CE loss ──
        # 全局 type_id → 流内索引
        tgt_type_stream = self.type_to_stream_idx[target_type_id]  # [N, max_ev]
        type_valid = (tgt_type_stream >= 0) & events_mask          # [N, max_ev]
        if type_valid.any():
            type_loss = F.cross_entropy(
                pred_type[type_valid],
                tgt_type_stream[type_valid],
            )
        else:
            type_loss = torch.tensor(0.0, device=device)

        # ── Key CE loss (仅 field_mask 有效的 position) ──
        key_loss = torch.tensor(0.0, device=device)
        key_valid_count = 0
        for tidx in self.stream_types:
            fm = FIELD_MASK[tidx]
            if fm[0] == 0:                                        # key_id not valid
                continue
            pos_mask = (target_type_id == tidx) & events_mask     # [N, max_ev]
            if pos_mask.any():
                tgt_key = (events[..., 1].long() + 1).clamp(0, N_KEY - 1)
                key_loss += F.cross_entropy(
                    pred_key[pos_mask],
                    tgt_key[pos_mask],
                )
                key_valid_count += 1
        if key_valid_count > 0:
            key_loss = key_loss / key_valid_count

        # ── Button CE loss ──
        btn_loss = torch.tensor(0.0, device=device)
        btn_valid_count = 0
        for tidx in self.stream_types:
            fm = FIELD_MASK[tidx]
            if fm[1] == 0:                                        # button_id not valid
                continue
            pos_mask = (target_type_id == tidx) & events_mask
            if pos_mask.any():
                tgt_btn = (events[..., 2].long() + 1).clamp(0, N_BTN - 1)
                btn_loss += F.cross_entropy(
                    pred_btn[pos_mask],
                    tgt_btn[pos_mask],
                )
                btn_valid_count += 1
        if btn_valid_count > 0:
            btn_loss = btn_loss / btn_valid_count

        # ── Continuous MSE loss (per-type, per-field mask) ──
        # 用方差加权归一化，避免 dt_ms(≈64) 主导 loss
        cont_std = self.event_embed.cont_std                        # [7]
        cont_loss = torch.tensor(0.0, device=device)
        cont_count = 0
        for tidx in self.stream_types:
            fm = FIELD_MASK[tidx]
            cont_valid = fm[2:9]                                   # 7-dim bool mask
            pos_mask = (target_type_id == tidx) & events_mask      # [N, max_ev]
            if pos_mask.any():
                raw_err = (pred_cont[pos_mask] - events[pos_mask][:, 3:10]) ** 2  # [n_pos, 7]
                # 逐维度方差加权: err / std^2
                scaled_err = raw_err / (cont_std ** 2).clamp(min=1e-6)
                # 只保留有效字段
                weight = torch.tensor(cont_valid, device=device, dtype=torch.float32)
                err = (scaled_err * weight).sum() / weight.sum().clamp(min=1)
                cont_loss += err
                cont_count += 1
        if cont_count > 0:
            cont_loss = cont_loss / cont_count

        # ── EOS BCE loss ──
        eos_pred = preds["eos_logits"].squeeze(-1)                # [N, max_ev+1]
        eos_target_s = eos_target.squeeze(-1)                     # [N, max_ev+1]
        eos_loss = F.binary_cross_entropy_with_logits(
            eos_pred[valid_pos.bool()],
            eos_target_s[valid_pos.bool()],
        )

        # ── 总计 ──
        total = type_loss + key_loss + btn_loss + cont_loss + eos_loss
        return total


# ═══════════════════════════════════════════════════════════
#  Audio / Gaze loss
# ═══════════════════════════════════════════════════════════

def audio_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity(CLAP 语义向量)。"""
    # pred, target: [B, K', 512]
    cos = F.cosine_similarity(pred, target, dim=-1)               # [B, K']
    return (1.0 - cos).mean()


def gaze_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE (归一化屏幕坐标)。pred 已过 sigmoid。"""
    return F.mse_loss(pred, target)


# ═══════════════════════════════════════════════════════════
#  总 Loss — detach 归一化
# ═══════════════════════════════════════════════════════════

def total_loss(losses: dict) -> torch.Tensor:
    """detach 归一化求和，5 头梯度自动同量级。

    Args:
        losses: {"audio": L_a, "frame": L_f, "gaze": L_g, "kb": L_kb, "mouse": L_m}
    Returns:
        scalar loss
    """
    eps = 1e-8
    L = (
        losses["audio"] / (losses["audio"].detach() + eps)
        + losses["frame"] / (losses["frame"].detach() + eps)
        + losses["gaze"] / (losses["gaze"].detach() + eps)
        + losses["kb"] / (losses["kb"].detach() + eps)
        + losses["mouse"] / (losses["mouse"].detach() + eps)
    )
    return L
