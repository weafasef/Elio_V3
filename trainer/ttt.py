#!/usr/bin/env python3
"""
TTT (Test-Time Training) 海马体记忆模块。

W_fast 黑板矩阵 [B, d_mem, d_mem] — 跨帧记忆的载体。
θ_meta (慢权重): W_init, log_lr, q_proj, k_proj, v_proj, read_proj, norm。
内层更新: 自包含重建 loss ||W·k - v||², 闭式梯度 2(Wk-v)kᵀ。

二阶路径: grad_W 由 k_proj/v_proj (θ_meta) 算出的可微张量构成，
外层 task loss 经 W_new → grad_W → k_proj/v_proj 反传时 autograd
自然再求一次导 = 二阶。不需要 create_graph 嵌套。
"""

import torch
import torch.nn as nn


class TTTMemory(nn.Module):
    """W_fast 黑板记忆。

    READ (Llama 前): 用帧的感知摘要查黑板 → memory_token
    WRITE (Llama 后): 用 Llama 输出摘要更新黑板 (闭式一步 SGD)

    d_mem=256 时慢权重约 2.6M 参数。
    """

    def __init__(self, llama_dim: int = 2048, d_mem: int = 256):
        super().__init__()
        self.llama_dim = llama_dim
        self.d_mem = d_mem

        # ── θ_meta 慢权重 ──
        self.W_init = nn.Parameter(torch.zeros(d_mem, d_mem))
        self.log_lr = nn.Parameter(torch.tensor(-2.0))          # 内层学习率(log 参数化保正)

        self.q_proj = nn.Linear(llama_dim, d_mem)              # READ 查询 (Llama 前)
        self.k_proj = nn.Linear(llama_dim, d_mem)              # WRITE 键 (Llama 后)
        self.v_proj = nn.Linear(llama_dim, d_mem)              # WRITE 值 (Llama 后)
        self.read_proj = nn.Linear(d_mem, llama_dim)            # 记忆向量 → Llama 维度
        self.norm = nn.LayerNorm(llama_dim)

        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[TTTMemory] total params: {total:,}  trainable: {trainable:,}")

    def init_state(self, B: int, device, dtype=torch.float32) -> torch.Tensor:
        """返回 W_fast 初始状态 [B, d_mem, d_mem]，从 W_init 广播。

        expand 链通 requires_grad 到 W_init — 不用 torch.zeros 新建。
        """
        return self.W_init.to(dtype).unsqueeze(0).expand(B, -1, -1).contiguous()

    def read(self, W: torch.Tensor, frame_summary: torch.Tensor) -> torch.Tensor:
        """Llama 前调用：用感知摘要查黑板。

        Args:
            W:             [B, d_mem, d_mem]  当前黑板
            frame_summary: [B, llama_dim]      34 token 均值 (Llama 前的感知帧)

        Returns:
            memory_token:  [B, 1, llama_dim]   供拼接进 Llama 序列
        """
        q = self.q_proj(frame_summary)                           # [B, d_mem]
        m = torch.bmm(W, q.unsqueeze(-1)).squeeze(-1)            # [B, d_mem]  黑板读出
        token = self.norm(self.read_proj(m))                      # [B, llama_dim]
        return token.unsqueeze(1)                                  # [B, 1, llama_dim]

    def update(self, W: torch.Tensor, frame_summary: torch.Tensor):
        """Llama 后调用：用输出摘要写黑板。

        闭式梯度 grad_W = 2(Wk - v)kᵀ，等价于 autograd 对 ||Wk-v||² 的梯度。
        grad_W 由 k_proj/v_proj (θ_meta) 可微算出 — 外层 task loss 经此反传 = 二阶。

        Args:
            W:             [B, d_mem, d_mem]  当前黑板
            frame_summary: [B, llama_dim]      35 token 均值 (Llama 后)

        Returns:
            W_new: [B, d_mem, d_mem]  更新后黑板
            ssl:   scalar              自监督重建 loss (仅监控, 不用于 backward)
        """
        k = self.k_proj(frame_summary)                           # [B, d_mem]
        v = self.v_proj(frame_summary)                           # [B, d_mem]
        pred_v = torch.bmm(W, k.unsqueeze(-1)).squeeze(-1)       # [B, d_mem]
        ssl = ((pred_v - v) ** 2).sum(-1).mean()                  # 标量 (仅监控)

        # 闭式梯度 ∂||Wk-v||²/∂W = 2(Wk-v)kᵀ
        # 这行绝对不能 detach (除对照实验外)，否则二阶断链
        err = pred_v - v                                          # [B, d_mem]
        grad_W = 2.0 * torch.bmm(err.unsqueeze(-1), k.unsqueeze(1))  # [B, d_mem, d_mem]

        lr = self.log_lr.exp()
        W_new = W - lr * grad_W                                   # 一步梯度下降

        return W_new, ssl
