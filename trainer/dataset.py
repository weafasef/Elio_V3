#!/usr/bin/env python3
"""
K 帧片段 Dataset — mmap 懒加载，不爆内存。

用法:
    from trainer.dataset import ProcessedDataset, collate_fn
    ds = ProcessedDataset(processed_dirs=["data/processed/xxx/processed"], K=4)
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ── 每帧各模态 shape 常量 ──
SIGLIP_TOKENS = 196       # SigLIP patch tokens
DINO_TOKENS = 257         # DINOv2 CLS + patches
VISUAL_DIM = 768          # SigLIP/DINOv2 hidden
AUDIO_DIM = 512           # CLAP embedding
ACTION_DIM = 178          # actions snapshot
EVENT_DIM = 10            # events_flat 列数
GAZE_DIM = 2              # gaze_pseudo


class ProcessedDataset(Dataset):
    """对已预处理目录提供 K 帧连续窗口的随机/顺序访问。

    - 所有 .npy 文件用 mmap_mode='r' 打开，按页懒加载
    - 窗口绝不跨 session（跨 session 无时间连续性）
    - events 存为 list of tensors，留给 collate 统一 pad
    """

    def __init__(
        self,
        processed_dirs: list[str | Path],
        K: int = 4,
        stride: int = 1,
    ):
        """
        Args:
            processed_dirs: 已预处理目录列表（每个含 12 个 .npy + spec.json）
            K: 每段连续帧数
            stride: 窗口起点步长（默认 1 即滑动窗）
        """
        self.K = K
        self.stride = stride

        # ── 为每个 session 打开 mmap ──
        self.sessions: list[dict] = []
        self.window_starts: list[tuple[int, int]] = []  # [(session_idx, frame_start), ...]

        for si, pdir in enumerate(processed_dirs):
            pdir = Path(pdir)
            spec = json.loads((pdir / "spec.json").read_text())
            N = spec["N"]
            if N < K:
                continue  # 太短，跳过

            sess = {
                "N": N,
                "spec": spec,
                "siglip": np.load(str(pdir / "visual_siglip.npy"), mmap_mode="r"),
                "dinov2": np.load(str(pdir / "visual_dinov2.npy"), mmap_mode="r"),
                "siglip_fov": np.load(str(pdir / "visual_siglip_fovea.npy"), mmap_mode="r"),
                "dinov2_fov": np.load(str(pdir / "visual_dinov2_fovea.npy"), mmap_mode="r"),
                "audio": np.load(str(pdir / "audio_clap.npy"), mmap_mode="r"),
                "actions": np.load(str(pdir / "actions.npy"), mmap_mode="r"),
                "gaze": np.load(str(pdir / "gaze_pseudo.npy"), mmap_mode="r"),
            }

            # events CSR — 只在有事件时打开
            flat_path = pdir / "events_flat.npy"
            off_path = pdir / "events_offsets.npy"
            if flat_path.exists() and off_path.exists():
                sess["events_flat"] = np.load(str(flat_path), mmap_mode="r")
                sess["events_offsets"] = np.load(str(off_path))
            else:
                sess["events_flat"] = None
                sess["events_offsets"] = None

            # 索引所有合法窗口起点
            max_start = N - K
            starts = list(range(0, max_start + 1, stride))
            for t0 in starts:
                self.window_starts.append((si, t0))

            self.sessions.append(sess)

        if not self.sessions:
            raise ValueError("No valid sessions found (all too short for K)")

        self._total_windows = len(self.window_starts)

    def __len__(self) -> int:
        return self._total_windows

    @property
    def session_lengths(self) -> list[int]:
        """各 session 帧数。"""
        return [s["N"] for s in self.sessions]

    def get_segment(self, si: int, t0: int) -> dict:
        """训练循环用：按坐标顺序取段。K 帧不足时用剩余帧（末尾）。

        Args:
            si: session 索引
            t0: 起始帧索引，越界自动 clamp 到 N-2（至少留 1 帧做 target）
        """
        sess = self.sessions[si]
        t0 = min(t0, sess["N"] - 2)          # 至少留 1 帧做 target
        return self._load(si, t0)

    def _load(self, si: int, t0: int) -> dict:
        """按 (session_idx, frame_start) 加载 K 帧段。__getitem__ 和 get_segment 共用。"""
        sess = self.sessions[si]
        K = self.K
        t1 = t0 + K  # 不含

        # ── 视觉 ──
        siglip = torch.from_numpy(sess["siglip"][t0:t1].copy()).float()     # [K, 196, 768]
        dinov2 = torch.from_numpy(sess["dinov2"][t0:t1].copy()).float()     # [K, 257, 768]
        siglip_fov = torch.from_numpy(sess["siglip_fov"][t0:t1].copy()).float()
        dinov2_fov = torch.from_numpy(sess["dinov2_fov"][t0:t1].copy()).float()

        # ── 音频 ──
        audio = torch.from_numpy(sess["audio"][t0:t1].copy()).float()       # [K, 512]

        # ── 动作快照 ──
        actions = torch.from_numpy(sess["actions"][t0:t1].copy()).float()   # [K, 178]

        # ── 注视 ──
        gaze = torch.from_numpy(sess["gaze"][t0:t1].copy()).float()         # [K, 2]

        # ── 事件 CSR → K 个变长 tensor ──
        events: list[torch.Tensor] = []
        if sess["events_flat"] is not None:
            offsets = sess["events_offsets"]
            ev_flat = sess["events_flat"]
            for t in range(t0, t1):
                e0, e1 = int(offsets[t]), int(offsets[t + 1])
                if e1 > e0:
                    ev = torch.from_numpy(ev_flat[e0:e1].copy()).float()    # [n_t, 10]
                else:
                    ev = torch.zeros((0, EVENT_DIM), dtype=torch.float32)
                events.append(ev)
        else:
            events = [torch.zeros((0, EVENT_DIM), dtype=torch.float32) for _ in range(K)]

        # ── 帧级 target: Δz (残差) ──
        pdir = Path(sess["siglip"].filename).parent
        tgt_s_path = pdir / "frame_targets_siglip.npy"
        tgt_d_path = pdir / "frame_targets_dinov2.npy"

        # 懒加载 target mmap (缓存到 session dict)
        if "tgt_siglip" not in sess:
            sess["tgt_siglip"] = np.load(str(tgt_s_path), mmap_mode="r")
            sess["tgt_dinov2"] = np.load(str(tgt_d_path), mmap_mode="r")

        tgt_s = torch.from_numpy(sess["tgt_siglip"][t0:t1].copy()).float()  # [K, 196, 768]
        tgt_d = torch.from_numpy(sess["tgt_dinov2"][t0:t1].copy()).float()  # [K, 257, 768]

        return {
            "siglip": siglip,              # [K, 196, 768]
            "dinov2": dinov2,              # [K, 257, 768]
            "siglip_fov": siglip_fov,      # [K, 196, 768]
            "dinov2_fov": dinov2_fov,      # [K, 257, 768]
            "audio": audio,                # [K, 512]
            "actions": actions,            # [K, 178]
            "gaze": gaze,                  # [K, 2]
            "events": events,              # list[K] of [n_t, 10]
            "tgt_siglip": tgt_s,           # [K, 196, 768]
            "tgt_dinov2": tgt_d,           # [K, 257, 768]
        }

    def __getitem__(self, idx: int) -> dict:
        si, t0 = self.window_starts[idx]
        return self._load(si, t0)

    def close(self):
        """关闭所有 mmap（DataLoader worker 退出时调用）。"""
        for sess in self.sessions:
            for key in list(sess.keys()):
                val = sess[key]
                if hasattr(val, "_mmap"):
                    val._mmap.close()
                elif isinstance(val, np.ndarray) and hasattr(val, "base"):
                    # 可能是 mmap 的 view
                    base = val
                    while hasattr(base, "base") and base.base is not None:
                        base = base.base
                    if hasattr(base, "_mmap"):
                        base._mmap.close()


def collate_fn(batch: list[dict]) -> dict:
    """将一个 batch 的样本拼成规整张量。

    常规模态 stack 成 [B, K, ...]。
    事件 pad 成 [B, K, max_ev, 10]，附带 events_mask。

    Returns:
        dict with all modalities batched, plus:
        - events_mask: [B, K, max_ev]  True=有效事件, False=padding
    """
    B = len(batch)

    # ── 常规张量直接 stack ──
    def _stack(key: str) -> torch.Tensor:
        return torch.stack([s[key] for s in batch], dim=0)  # [B, K, ...]

    out = {
        "siglip": _stack("siglip"),
        "dinov2": _stack("dinov2"),
        "siglip_fov": _stack("siglip_fov"),
        "dinov2_fov": _stack("dinov2_fov"),
        "audio": _stack("audio"),
        "actions": _stack("actions"),
        "gaze": _stack("gaze"),
        "tgt_siglip": _stack("tgt_siglip"),
        "tgt_dinov2": _stack("tgt_dinov2"),
    }

    # ── 事件 pad ──
    K = batch[0]["siglip"].shape[0]
    # 收集所有事件的长度
    all_lens = []
    for s in batch:
        for ev in s["events"]:
            all_lens.append(ev.shape[0])
    max_ev = max(all_lens) if all_lens else 0

    events_padded = torch.zeros(B, K, max_ev, EVENT_DIM, dtype=torch.float32)
    events_mask = torch.zeros(B, K, max_ev, dtype=torch.bool)

    for b in range(B):
        for k in range(K):
            ev = batch[b]["events"][k]
            n = ev.shape[0]
            if n > 0:
                events_padded[b, k, :n] = ev
                events_mask[b, k, :n] = True

    out["events"] = events_padded
    out["events_mask"] = events_mask

    return out
