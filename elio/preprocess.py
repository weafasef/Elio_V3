#!/usr/bin/env python3
"""
预处理录制的 session → 训练就绪的 .npy 张量。

相比 encode.py 的增量:
  - CLAP 音频嵌入 (512-dim) 取代 mel 频谱
  - motion saliency gaze 伪标签 → gaze_head 监督
  - 残差帧目标 Δz_f = z_{t+1} - z_t → frame_head 监督

用法:
    python preprocess.py                              # 自动选最新 session
    python preprocess.py session_20260618_212739      # 指定 session
    python preprocess.py --batch-size 8 --device cuda
    python preprocess.py --legacy-audio               # 同时保存 mel 频谱 (兼容旧 trainer)
    python preprocess.py --max-frames 100             # 限制帧数 (测试用)

输出:
    dataset/<timestamp>/processed/
    ├── visual_siglip.npy          [N, Ps, 768]  float16
    ├── visual_dinov2.npy          [N, Pd, 768]  float16
    ├── visual_siglip_fovea.npy    [N, Pf, 768]  float16  ← 焦点路
    ├── visual_dinov2_fovea.npy    [N, Pd, 768]  float16  ← 焦点路
    ├── frame_targets_siglip.npy   [N, Ps, 768]  float16  ← Δz_f 残差
    ├── frame_targets_dinov2.npy   [N, Pd, 768]  float16  ← Δz_f 残差
    ├── audio_clap.npy             [N, 512]      float16  ← CLAP 嵌入
    ├── gaze_pseudo.npy            [N, 2]        float32  ← motion saliency
    ├── actions.npy                [N, 178]      float32
    ├── events_flat.npy            [M, 10]       float32  ← 变长事件序列
    ├── events_offsets.npy         [N+1]         int64    ← CSR 行指针
    ├── timestamps.npy             [N]           float64
    └── spec.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── 路径（均相对 Elio_Agent_v3） ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SIGLIP_PATH = MODELS_DIR / "siglip_base"
DINOV2_PATH = MODELS_DIR / "dinov2_base"
CLAP_SNAPSHOT = MODELS_DIR / "clap-official" / "models--laion--clap-htsat-unfused" / \
    "snapshots" / "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
SESSION_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_DIR = PROJECT_ROOT / "data" / "processed"

FRAME_INTERVAL_NS = 100_000_000   # 10Hz → 100ms
CLAP_SR = 48000                    # CLAP 要求采样率
AUDIO_SR = 16000                   # 录制采样率
AUDIO_CHUNK_SAMPLES = 1600         # 100ms @ 16kHz

# ── 标准键盘映射（与 encode.py 一致，84 键） ─────────
KEY_ORDER = [
    'a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z',
    '0','1','2','3','4','5','6','7','8','9',
    'up','down','left','right',
    'shift_l','shift_r', 'ctrl_l','ctrl_r', 'alt_l','alt_r', 'win_l','win_r',
    'f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12',
    '`','-','=','[',']','\\',';',"'",',','.','/',
    'space','enter','backspace','tab','esc',
    'insert','delete','home','end','page_up','page_down',
    'caps_lock','print_screen',
]
K = len(KEY_ORDER)
A_DIM = 10 + K * 2                # 178
KEY_TO_IDX = {k: i for i, k in enumerate(KEY_ORDER)}

E_DIM = 10                         # 单事件编码维度
BUTTON_TO_ID = {"left": 0, "right": 1, "middle": 2}


# ═══════════════════════════════════════════════════════════
#  工具函数（从 encode.py 复用）
# ═══════════════════════════════════════════════════════════

def _normalize_key(raw: str) -> str:
    """pynput 键名 → KEY_ORDER 标准名。"""
    k = raw.lower()
    if k in ('cmd', 'cmd_l'):
        return 'win_l'
    if k == 'cmd_r':
        return 'win_r'
    if k == 'shift':
        return 'shift_l'
    if k == 'ctrl':
        return 'ctrl_l'
    if k == 'alt':
        return 'alt_l'
    return k


def find_latest_session(data_dir: str = ".") -> Path | None:
    base = Path(data_dir)
    sessions = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("session_")],
        key=lambda d: d.name, reverse=True,
    )
    return sessions[0] if sessions else None


def load_video_info(session_dir: Path) -> tuple[Path, int]:
    """返回 (video_path, total_frames)。"""
    for ext in ('.mp4', '.avi'):
        candidate = session_dir / f"video{ext}"
        if candidate.exists():
            cap = cv2.VideoCapture(str(candidate))
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {candidate}")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total == 0:
                raise RuntimeError(f"视频文件无帧: {candidate}")
            return candidate, total
    raise FileNotFoundError(f"找不到视频文件: {session_dir}/video.mp4")


def load_events(session_dir: Path) -> list[dict]:
    events: list[dict] = []
    events_path = session_dir / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    events.sort(key=lambda e: e["ts_ns"])
    return events


def get_screen_size(video_path: Path) -> tuple[int, int]:
    """从视频第一帧读取实际分辨率。"""
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


# ═══════════════════════════════════════════════════════════
#  动作向量 (与 encode.py 完全一致)
# ═══════════════════════════════════════════════════════════

def compute_actions(
    events: list[dict],
    t0_ns: int,
    total_frames: int,
    screen_w: int,
    screen_h: int,
) -> np.ndarray:
    """从事件序列计算每帧的动作向量 [N, A_DIM] float32。"""
    actions = np.zeros((total_frames, A_DIM), dtype=np.float32)
    if total_frames == 0:
        return actions

    cur_x = 0.0
    cur_y = 0.0
    left_held = False
    right_held = False
    keys_held = {k: False for k in KEY_ORDER}

    event_idx = 0
    E = len(events)

    for frame_idx in range(total_frames):
        frame_start = t0_ns + frame_idx * FRAME_INTERVAL_NS
        frame_end = t0_ns + (frame_idx + 1) * FRAME_INTERVAL_NS

        win_events: list[dict] = []
        while event_idx < E and events[event_idx]["ts_ns"] < frame_end:
            ev = events[event_idx]
            if ev["ts_ns"] >= frame_start:
                win_events.append(ev)
            event_idx += 1

        first_x, first_y = cur_x, cur_y
        path_len = 0.0
        prev_x, prev_y = cur_x, cur_y
        scroll_dy = 0.0
        left_press_ev = False
        right_press_ev = False
        key_press_ev = {k: False for k in KEY_ORDER}

        for ev in win_events:
            t = ev["type"]
            if t == "mouse_move":
                cur_x = float(ev["x"])
                cur_y = float(ev["y"])
                path_len += np.sqrt((cur_x - prev_x)**2 + (cur_y - prev_y)**2)
                prev_x, prev_y = cur_x, cur_y
            elif t == "mouse_click":
                btn = ev.get("button", "")
                pressed = ev.get("pressed", False)
                if btn == "left":
                    left_held = pressed
                    if pressed:
                        left_press_ev = True
                elif btn == "right":
                    right_held = pressed
                    if pressed:
                        right_press_ev = True
            elif t == "mouse_scroll":
                scroll_dy += float(ev.get("dy", 0))
            elif t == "key_press":
                k = _normalize_key(ev.get("key", ""))
                if k in KEY_TO_IDX:
                    key_press_ev[k] = True
                    keys_held[k] = True
            elif t == "key_release":
                k = _normalize_key(ev.get("key", ""))
                if k in KEY_TO_IDX:
                    keys_held[k] = False

        act = actions[frame_idx]
        act[0] = (cur_x - first_x) / screen_w
        act[1] = (cur_y - first_y) / screen_h
        act[2] = cur_x / screen_w
        act[3] = cur_y / screen_h
        act[4] = path_len / screen_w
        act[5] = scroll_dy
        act[6] = 1.0 if left_held else 0.0
        act[7] = 1.0 if right_held else 0.0
        act[8] = 1.0 if left_press_ev else 0.0
        act[9] = 1.0 if right_press_ev else 0.0
        for i, k in enumerate(KEY_ORDER):
            base = 10 + i * 2
            act[base + 0] = 1.0 if key_press_ev[k] else 0.0
            act[base + 1] = 1.0 if keys_held[k] else 0.0

    return actions


# ═══════════════════════════════════════════════════════════
#  变长事件序列 (CSR)
# ═══════════════════════════════════════════════════════════

def _flush_move(
    event_rows: list[list],
    start_x: float, start_y: float,
    end_x: float, end_y: float,
    path_len: float, last_dt_ms: float,
    screen_w: int, screen_h: int,
):
    """写一个 type=4 (move) 行，合并段的终点作为坐标。"""
    event_rows.append([
        4,                                    # type_id = move
        -1,                                   # key_id: n/a
        -1,                                   # button_id: n/a
        end_x / screen_w,                     # x_norm: 段终点
        end_y / screen_h,                     # y_norm: 段终点
        (end_x - start_x) / screen_w,         # dx_norm: 净位移
        (end_y - start_y) / screen_h,         # dy_norm: 净位移
        path_len / screen_w,                  # path_len_norm
        0.0,                                  # scroll_dy: n/a
        last_dt_ms,                           # dt_ms: 段最后事件的帧内偏移
    ])


def compute_events_csr(
    events: list[dict],
    t0_ns: int,
    total_frames: int,
    screen_w: int,
    screen_h: int,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """从事件序列计算变长事件 CSR。

    Returns:
        events_flat   [M, E_DIM] float32  所有帧事件拍平
        events_offsets [N+1]    int64      帧 t 事件 = flat[off[t]:off[t+1]]

    核心原则:
      - down/up 各自独立按 ts_ns 落桶，绝不配对
      - mouse_move 合并成段，遇非 move 事件时 flush
      - 游标 cur_x/cur_y 跨帧保持
      - 帧分桶逻辑与 compute_actions() 完全一致
    """
    N = total_frames if max_frames is None else min(total_frames, max_frames)

    event_rows: list[list] = []
    offsets = np.zeros(N + 1, dtype=np.int64)

    cur_x = 0.0
    cur_y = 0.0

    # move 合并缓冲
    move_buf_start = None    # (start_x, start_y)
    move_buf_end = (0.0, 0.0)
    move_buf_path = 0.0
    move_buf_last_dt = 0.0
    move_buf_prev = (0.0, 0.0)

    event_idx = 0
    E = len(events)

    for frame_idx in range(N):
        frame_start = t0_ns + frame_idx * FRAME_INTERVAL_NS
        frame_end = t0_ns + (frame_idx + 1) * FRAME_INTERVAL_NS

        # 起始偏移
        offsets[frame_idx] = len(event_rows)

        while event_idx < E and events[event_idx]["ts_ns"] < frame_end:
            ev = events[event_idx]
            if ev["ts_ns"] < frame_start:
                event_idx += 1
                continue

            dt_ms = (ev["ts_ns"] - frame_start) / 1e6
            t = ev["type"]

            if t == "mouse_move":
                x = float(ev["x"])
                y = float(ev["y"])

                if move_buf_start is None:
                    move_buf_start = (cur_x, cur_y)
                    move_buf_prev = (cur_x, cur_y)
                    move_buf_path = 0.0

                move_buf_path += np.sqrt(
                    (x - move_buf_prev[0])**2 + (y - move_buf_prev[1])**2
                )
                move_buf_prev = (x, y)
                move_buf_end = (x, y)
                move_buf_last_dt = dt_ms
                cur_x, cur_y = x, y

            else:
                # 非 move 事件 → 先 flush move 段
                if move_buf_start is not None:
                    _flush_move(
                        event_rows,
                        move_buf_start[0], move_buf_start[1],
                        move_buf_end[0], move_buf_end[1],
                        move_buf_path, move_buf_last_dt,
                        screen_w, screen_h,
                    )
                    move_buf_start = None

                if t == "mouse_click":
                    btn = ev.get("button", "")
                    pressed = ev.get("pressed", False)
                    btn_id = BUTTON_TO_ID.get(btn, -1)
                    type_id = 2 if pressed else 3   # 2=mouse_down, 3=mouse_up
                    event_rows.append([
                        type_id,
                        -1,                        # key_id: n/a
                        btn_id,
                        cur_x / screen_w,          # 游标位置
                        cur_y / screen_h,
                        0.0, 0.0, 0.0,            # dx/dy/path: n/a
                        0.0,                       # scroll_dy: n/a
                        dt_ms,
                    ])

                elif t == "mouse_scroll":
                    scroll_dy = float(ev.get("dy", 0))
                    event_rows.append([
                        5,                          # type_id = scroll
                        -1,                        # key_id: n/a
                        -1,                        # button_id: n/a
                        cur_x / screen_w,
                        cur_y / screen_h,
                        0.0, 0.0, 0.0,            # dx/dy/path: n/a
                        scroll_dy,                 # scroll_dy
                        dt_ms,
                    ])

                elif t == "key_press":
                    k = _normalize_key(ev.get("key", ""))
                    key_id = KEY_TO_IDX.get(k, -1)
                    if key_id >= 0:
                        event_rows.append([
                            0,                        # type_id = key_down
                            key_id,
                            -1,                      # button_id: n/a
                            cur_x / screen_w,        # 按键时游标位置
                            cur_y / screen_h,
                            0.0, 0.0, 0.0,          # dx/dy/path: n/a
                            0.0,                     # scroll_dy: n/a
                            dt_ms,
                        ])

                elif t == "key_release":
                    k = _normalize_key(ev.get("key", ""))
                    key_id = KEY_TO_IDX.get(k, -1)
                    if key_id >= 0:
                        event_rows.append([
                            1,                        # type_id = key_up
                            key_id,
                            -1,                      # button_id: n/a
                            cur_x / screen_w,
                            cur_y / screen_h,
                            0.0, 0.0, 0.0,          # dx/dy/path: n/a
                            0.0,                     # scroll_dy: n/a
                            dt_ms,
                        ])

            event_idx += 1

        # 帧事件遍历完 → flush 残留 move 段
        if move_buf_start is not None:
            _flush_move(
                event_rows,
                move_buf_start[0], move_buf_start[1],
                move_buf_end[0], move_buf_end[1],
                move_buf_path, move_buf_last_dt,
                screen_w, screen_h,
            )
            move_buf_start = None

    # 末尾偏移
    offsets[N] = len(event_rows)

    # → np arrays
    M = len(event_rows)
    flat = np.zeros((M, E_DIM), dtype=np.float32)
    for i, row in enumerate(event_rows):
        flat[i] = row

    return flat, offsets


# ═══════════════════════════════════════════════════════════
#  Gaze 伪标签 (motion saliency)
# ═══════════════════════════════════════════════════════════

def compute_gaze_pseudo_labels(
    video_path: Path,
    total_frames: int,
    screen_w: int,
    screen_h: int,
    max_frames: int | None = None,
) -> np.ndarray:
    """Pass 1: 读视频帧差 → motion saliency argmax → [N, 2] 归一化坐标。

    为省内存，帧差计算在缩小的 1/4 分辨率上进行，坐标再映射回原分辨率。
    """
    N = total_frames if max_frames is None else min(total_frames, max_frames)
    gaze = np.zeros((N, 2), dtype=np.float32)
    gaze[0] = (0.5, 0.5)          # 首帧默认中心

    if N < 2:
        return gaze

    # 缩小分辨率以加速 diff 计算
    scale = 0.25
    small_w = max(1, int(screen_w * scale))
    small_h = max(1, int(screen_h * scale))

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, prev_raw = cap.read()
    if not ret:
        cap.release()
        return gaze
    prev = cv2.resize(prev_raw, (small_w, small_h))

    pbar = tqdm(total=N - 1, desc="Gaze labels", unit="frame", ncols=100)
    for t in range(1, N):
        ret, cur_raw = cap.read()
        if not ret:
            break
        cur = cv2.resize(cur_raw, (small_w, small_h))

        # 绝对差 → 灰度 → 高斯模糊
        diff = np.abs(cur.astype(np.float32) - prev.astype(np.float32))
        gray = diff.mean(axis=2)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        # argmax → 映射回原分辨率
        gy, gx = np.unravel_index(np.argmax(blur), blur.shape)
        gaze[t, 0] = gx / small_w     # x 归一化
        gaze[t, 1] = gy / small_h     # y 归一化

        prev = cur
        pbar.update(1)

    pbar.close()
    cap.release()

    # 如果首帧未设（极端情况），用第二帧
    if N >= 2 and np.all(gaze[0] == 0.5):
        pass  # 已设默认

    return gaze


# ═══════════════════════════════════════════════════════════
#  残差目标
# ═══════════════════════════════════════════════════════════

def compute_residual_targets(
    src_path: Path,
    dst_path: Path,
    chunk_size: int = 512,
) -> tuple:
    """逐块计算残差目标，直接写 memmap，不分配全量内存。

    targets[t] = features[t+1] - features[t]; 末帧 = 0。
    返回 (shape, dtype)。
    """
    src = np.load(str(src_path), mmap_mode="r")
    shape = src.shape
    dtype = src.dtype
    N = shape[0]

    dst = np.lib.format.open_memmap(
        str(dst_path), mode="w+", dtype=dtype, shape=shape,
    )
    # 末帧保持零，不写入

    for i in range(0, N - 1, chunk_size):
        end = min(i + chunk_size, N - 1)
        dst[i:end] = src[i + 1:end + 1] - src[i:end]

    dst.flush()
    src._mmap.close()  # type: ignore[union-attr]
    return shape, dtype


# ═══════════════════════════════════════════════════════════
#  CLAP 音频编码
# ═══════════════════════════════════════════════════════════

def load_clap_model(device: str):
    """加载 CLAP 模型 (仅音频塔)。"""
    # ── workaround: transformers is_torch_greater_or_equal("2.6") 对 torch 2.5.1 错误返回 True ──
    import transformers.utils.import_utils as _tf_iu
    _tf_orig_check = _tf_iu.is_torch_greater_or_equal
    def _patched_check(version, accept_dev=False):
        from packaging import version as _ver
        _tv = _ver.parse(torch.__version__.split("+")[0].replace("dev", ""))
        _rv = _ver.parse(version.replace("dev", ""))
        return _tv >= _rv
    _tf_iu.is_torch_greater_or_equal = _patched_check
    # ──────────────────────────────────────────────────────────────────────────────────────────

    from transformers import ClapModel, ClapProcessor
    import torchaudio.transforms as T

    snap = str(CLAP_SNAPSHOT)
    print(f"Loading CLAP from: {snap}")
    model = ClapModel.from_pretrained(snap).to(device).eval()
    processor = ClapProcessor.from_pretrained(snap)
    resampler = T.Resample(orig_freq=AUDIO_SR, new_freq=CLAP_SR).to(device)
    return model, processor, resampler


def encode_audio_clap(
    audio_chunks: np.ndarray,       # [N_audio, 1600] int16
    model,
    processor,
    resampler,
    device: str,
    batch_size: int,
) -> np.ndarray:                    # [N_audio, 512] float16
    """100ms 音频 chunk → CLAP 512-dim 嵌入。

    流程: int16→float32→归一化→重采样 16k→48k→ClapProcessor→ClapModel
    """
    from transformers import ClapProcessor as CP

    N = audio_chunks.shape[0]
    embeddings = np.zeros((N, 512), dtype=np.float16)
    CLAP_DIM = 512

    pbar = tqdm(total=N, desc="Audio CLAP", unit="chunk", ncols=100)

    for i in range(0, N, batch_size):
        batch_end = min(i + batch_size, N)
        B = batch_end - i

        # int16 → float32, 归一化 [-1, 1]
        waves = audio_chunks[i:batch_end].astype(np.float32) / 32768.0

        # 重采样 16kHz → 48kHz
        wave_t = torch.from_numpy(waves).to(device)         # [B, 1600]
        wave_48k = resampler(wave_t)                         # [B, 4800]

        # 过 CLAP processor + model
        # ClapProcessor 返回 input_features (mel spec) 用于音频塔
        # 但 processor.__call__ 期望原始波形, 内部做 mel
        inputs = processor(
            audio=[w.cpu().numpy() for w in wave_48k],
            sampling_rate=CLAP_SR,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_audio_features(**inputs)
            # 兼容不同 transformers 版本返回类型
        if hasattr(outputs, "pooler_output"):
            emb = outputs.pooler_output.cpu().numpy().astype(np.float16)
        elif isinstance(outputs, torch.Tensor):
            emb = outputs.cpu().numpy().astype(np.float16)
        else:
            # BaseModelOutputWithPooling 或类似 namedtuple
            emb = outputs[0].cpu().numpy().astype(np.float16)
        embeddings[i:batch_end] = emb

        pbar.update(B)

    pbar.close()
    return embeddings


# ═══════════════════════════════════════════════════════════
#  视觉编码
# ═══════════════════════════════════════════════════════════

def load_vision_models(device: str):
    """加载 SigLIP + DINOv2。"""
    from transformers import SiglipVisionModel, Dinov2Model, AutoImageProcessor

    print(f"Loading SigLIP from: {SIGLIP_PATH}")
    siglip = SiglipVisionModel.from_pretrained(str(SIGLIP_PATH)).to(device).eval()
    print(f"Loading DINOv2 from: {DINOV2_PATH}")
    dinov2 = Dinov2Model.from_pretrained(str(DINOV2_PATH)).to(device).eval()

    siglip_processor = AutoImageProcessor.from_pretrained(str(SIGLIP_PATH))
    dinov2_processor = AutoImageProcessor.from_pretrained(str(DINOV2_PATH))
    return siglip, dinov2, siglip_processor, dinov2_processor


def encode_visual(
    video_path: Path,
    N: int,
    siglip_model,
    dinov2_model,
    siglip_processor,
    dinov2_processor,
    device: str,
    batch_size: int,
    output_dir: Path,
    max_frames: int | None = None,
) -> tuple[int, int, int, int]:
    """Pass 2: 逐帧过 SigLIP + DINOv2 → 写入 memmap .npy。

    Returns:
        (P_siglip, P_dino, D_siglip, D_dino) 维度信息
    """
    N_enc = N if max_frames is None else min(N, max_frames)

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── 跑首帧确认 patch 数 ──
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("无法读取视频第一帧")
    test_img = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))

    with torch.no_grad():
        si = siglip_processor(images=test_img, return_tensors="pt").to(device)
        so = siglip_model(**si)
        P_siglip = so.last_hidden_state.shape[1]
        D_siglip = siglip_model.config.hidden_size

        di = dinov2_processor(images=test_img, return_tensors="pt").to(device)
        do = dinov2_model(**di)
        P_dino = do.last_hidden_state.shape[1]
        D_dino = dinov2_model.config.hidden_size

    print(f"  SigLIP: [{N_enc}, {P_siglip}, {D_siglip}] float16")
    print(f"  DINOv2: [{N_enc}, {P_dino}, {D_dino}] float16")

    # ── memmap 预分配 ──
    siglip_mmap = np.lib.format.open_memmap(
        str(output_dir / "visual_siglip.npy"), mode="w+",
        dtype=np.float16, shape=(N_enc, P_siglip, D_siglip),
    )
    dinov2_mmap = np.lib.format.open_memmap(
        str(output_dir / "visual_dinov2.npy"), mode="w+",
        dtype=np.float16, shape=(N_enc, P_dino, D_dino),
    )

    # ── 回退到开头，批量编码 ──
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pbar = tqdm(total=N_enc, desc="Visual enc", unit="frame", ncols=100)
    frame_buf: list[np.ndarray] = []

    for batch_start in range(0, N_enc, batch_size):
        batch_end = min(batch_start + batch_size, N_enc)
        batch_count = batch_end - batch_start

        frame_buf.clear()
        for _ in range(batch_count):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buf.append(frame_rgb)

        images = [Image.fromarray(f) for f in frame_buf]
        actual = len(images)
        if actual == 0:
            break

        si = siglip_processor(images=images, return_tensors="pt").to(device)
        di = dinov2_processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            s_out = siglip_model(**si)
            d_out = dinov2_model(**di)
            s_vecs = s_out.last_hidden_state.cpu().numpy().astype(np.float16)
            d_vecs = d_out.last_hidden_state.cpu().numpy().astype(np.float16)

        siglip_mmap[batch_start:batch_start + actual] = s_vecs
        dinov2_mmap[batch_start:batch_start + actual] = d_vecs

        pbar.update(actual)

    pbar.close()
    cap.release()

    siglip_mmap.flush()
    dinov2_mmap.flush()

    return P_siglip, P_dino, D_siglip, D_dino


# ═══════════════════════════════════════════════════════════
#  焦点路视觉编码 (fovea)
# ═══════════════════════════════════════════════════════════

def _crop_fovea(
    frame_rgb: np.ndarray,
    gx: float,                 # 归一化 x [0,1]
    gy: float,                 # 归一化 y [0,1]
    W: int,
    H: int,
    crop_size: int,
) -> np.ndarray:
    """以 gaze (gx, gy) 为中心裁 crop_size×crop_size，clamp 边界。

    注意: frame_rgb 是 [H, W, 3] numpy 数组，切片用 [y, x] 顺序。
    """
    half = crop_size // 2
    cx = int(gx * W)
    cy = int(gy * H)

    # clamp：中心推回屏内，保证裁框完整无黑边
    cx = min(max(cx, half), W - half)
    cy = min(max(cy, half), H - half)

    crop = frame_rgb[cy - half:cy + half, cx - half:cx + half]
    return crop


def encode_fovea(
    video_path: Path,
    N: int,
    gaze: np.ndarray,           # [N, 2] float32 归一化坐标
    screen_w: int,
    screen_h: int,
    siglip_model,
    dinov2_model,
    siglip_processor,
    dinov2_processor,
    device: str,
    batch_size: int,
    output_dir: Path,
    crop_size: int = 448,
    max_frames: int | None = None,
) -> tuple[int, int, int, int]:
    """一次视频遍历同时编码 SigLIP + DINOv2 焦点路 → 写入 memmap。

    Returns: (P_siglip, P_dino, D_siglip, D_dino) 维度信息。
    """
    N_enc = N if max_frames is None else min(N, max_frames)

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── 首帧确认 patch 数 ──
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("无法读取视频第一帧")
    first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    gx0, gy0 = float(gaze[0, 0]), float(gaze[0, 1])
    first_crop = _crop_fovea(first_rgb, gx0, gy0, screen_w, screen_h, crop_size)
    test_img = Image.fromarray(first_crop)

    with torch.no_grad():
        si = siglip_processor(images=test_img, return_tensors="pt").to(device)
        so = siglip_model(**si)
        P_siglip = so.last_hidden_state.shape[1]
        D_siglip = siglip_model.config.hidden_size

        di = dinov2_processor(images=test_img, return_tensors="pt").to(device)
        do = dinov2_model(**di)
        P_dino = do.last_hidden_state.shape[1]
        D_dino = dinov2_model.config.hidden_size

    print(f"  Fovea SigLIP: [{N_enc}, {P_siglip}, {D_siglip}] float16  (crop={crop_size})")
    print(f"  Fovea DINOv2: [{N_enc}, {P_dino}, {D_dino}] float16  (crop={crop_size})")

    # ── memmap 预分配 ──
    siglip_mmap = np.lib.format.open_memmap(
        str(output_dir / "visual_siglip_fovea.npy"), mode="w+",
        dtype=np.float16, shape=(N_enc, P_siglip, D_siglip),
    )
    dinov2_mmap = np.lib.format.open_memmap(
        str(output_dir / "visual_dinov2_fovea.npy"), mode="w+",
        dtype=np.float16, shape=(N_enc, P_dino, D_dino),
    )

    # ── 回退开头，批量编码 ──
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pbar = tqdm(total=N_enc, desc="Fovea visual", unit="frame", ncols=100)

    for batch_start in range(0, N_enc, batch_size):
        batch_end = min(batch_start + batch_size, N_enc)
        batch_count = batch_end - batch_start

        crops: list[Image.Image] = []
        for t in range(batch_start, batch_end):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gx = float(gaze[t, 0])
            gy = float(gaze[t, 1])
            crop = _crop_fovea(frame_rgb, gx, gy, screen_w, screen_h, crop_size)
            crops.append(Image.fromarray(crop))

        actual = len(crops)
        if actual == 0:
            break

        si = siglip_processor(images=crops, return_tensors="pt").to(device)
        di = dinov2_processor(images=crops, return_tensors="pt").to(device)

        with torch.no_grad():
            s_out = siglip_model(**si)
            d_out = dinov2_model(**di)
            s_vecs = s_out.last_hidden_state.cpu().numpy().astype(np.float16)
            d_vecs = d_out.last_hidden_state.cpu().numpy().astype(np.float16)

        siglip_mmap[batch_start:batch_start + actual] = s_vecs
        dinov2_mmap[batch_start:batch_start + actual] = d_vecs
        pbar.update(actual)

    pbar.close()
    cap.release()

    siglip_mmap.flush()
    dinov2_mmap.flush()

    return P_siglip, P_dino, D_siglip, D_dino


# ═══════════════════════════════════════════════════════════
#  Legacy mel spectrogram (兼容旧 trainer)
# ═══════════════════════════════════════════════════════════

def _mel_filterbank(n_fft: int, sr: int, n_mels: int) -> np.ndarray:
    mel_min, mel_max = 0, 2595 * np.log10(1 + (sr / 2) / 700)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    freq_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts = ((n_fft + 1) * freq_pts / sr).astype(np.int32)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins))
    for m in range(n_mels):
        lo, ctr, hi = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
        fb[m, lo:ctr] = (np.arange(lo, ctr) - lo) / max(1, ctr - lo)
        fb[m, ctr:hi] = (hi - np.arange(ctr, hi)) / max(1, hi - ctr)
    return fb


def compute_mel_legacy(
    audio: np.ndarray,              # [N, 1600] float32
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 64,
    n_frames: int = 10,
) -> np.ndarray:
    """numpy 手写 mel spectrogram（与 encode.py 回退路径一致）。"""
    try:
        import torchaudio.transforms as T
        mel_transform = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, win_length=n_fft,
            hop_length=hop_length, n_mels=n_mels, power=2.0,
        )
        has_torchaudio = True
    except Exception:
        has_torchaudio = False

    N = audio.shape[0]
    audio_float = audio.astype(np.float32) / 32768.0

    if has_torchaudio:
        mels = np.zeros((N, n_mels, n_frames), dtype=np.float16)
        wave = torch.from_numpy(audio_float)
        mel = mel_transform(wave)
        mel = torch.log(mel + 1e-6)
        if mel.shape[2] < n_frames:
            pad = torch.zeros(mel.shape[0], n_mels, n_frames - mel.shape[2])
            mel = torch.cat([mel, pad], dim=2)
        mel = mel[:, :, :n_frames]
        mels = mel.numpy().astype(np.float16)
        return mels
    else:
        window = np.hanning(n_fft)
        mel_fb = _mel_filterbank(n_fft, sr, n_mels)
        mels = np.zeros((N, n_mels, n_frames), dtype=np.float16)
        for i in range(N):
            wave = audio_float[i]
            for j in range(n_frames):
                start = j * hop_length
                frame = wave[start:start + n_fft] * window
                if len(frame) < n_fft:
                    frame = np.pad(frame, (0, n_fft - len(frame)))
                spec = np.abs(np.fft.rfft(frame, n=n_fft))
                mel_energy = np.dot(mel_fb, spec ** 2)
                mels[i, :, j] = np.log(mel_energy + 1e-6)
        return mels


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def preprocess_session(
    session_dir: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    max_frames: int | None = None,
    legacy_audio: bool = False,
    resume: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 辅助: resume 跳过已有文件 ──
    def _skip(fname: str) -> bool:
        exists = (output_dir / fname).exists()
        if exists and resume:
            size = (output_dir / fname).stat().st_size
            print(f"  [SKIP] {fname} already exists ({size / 1024**2:.1f} MB)")
        return exists and resume

    # ── 加载原始数据元信息 ──
    video_path, total_frames = load_video_info(session_dir)
    events = load_events(session_dir)
    t0_ns = events[0]["ts_ns"] if events else 0
    screen_w, screen_h = get_screen_size(video_path)

    N = total_frames if max_frames is None else min(total_frames, max_frames)

    print(f"Session:  {session_dir.name}")
    print(f"Video:    {video_path.name}  ({screen_w}x{screen_h})")
    print(f"Frames:   {N} / {total_frames}")
    print(f"Events:   {len(events)}")
    print(f"t0:       {t0_ns}")

    # ═══════════════════════════════════════════════════════
    #  Step 1: Gaze 伪标签 (Pass 1 — 只读像素, 不跑模型)
    # ═══════════════════════════════════════════════════════
    print("\n── Step 1/6: Gaze pseudo-labels ──")
    if _skip("gaze_pseudo.npy"):
        gaze = np.load(str(output_dir / "gaze_pseudo.npy"))
    else:
        gaze = compute_gaze_pseudo_labels(video_path, total_frames, screen_w, screen_h, max_frames)
        np.save(str(output_dir / "gaze_pseudo.npy"), gaze)
    print(f"  gaze_pseudo:  {gaze.shape}  float32  "
          f"x∈[{gaze[:,0].min():.3f}, {gaze[:,0].max():.3f}]  "
          f"y∈[{gaze[:,1].min():.3f}, {gaze[:,1].max():.3f}]")

    # ═══════════════════════════════════════════════════════
    #  Step 2: CLAP 音频编码 → 卸载
    # ═══════════════════════════════════════════════════════
    audio_path = session_dir / "audio_raw.npy"
    has_audio = audio_path.exists()

    if has_audio:
        print("\n── Step 2/6: CLAP audio encoding ──")
        if _skip("audio_clap.npy"):
            audio_clap = np.load(str(output_dir / "audio_clap.npy"))
        else:
            audio_raw = np.load(str(audio_path))                         # [N_a, 1600] int16
            N_audio = min(N, len(audio_raw))
            audio_raw = audio_raw[:N_audio]

            clap_model, clap_processor, resampler = load_clap_model(device)
            audio_clap = encode_audio_clap(audio_raw, clap_model, clap_processor,
                                           resampler, device, batch_size)
            np.save(str(output_dir / "audio_clap.npy"), audio_clap)
        print(f"  audio_clap:  {audio_clap.shape}  float16  "
              f"mean={audio_clap.mean():.4f}  std={audio_clap.std():.4f}")

        # legacy mel 兼容
        if legacy_audio:
            print("  Computing legacy mel spectrogram...")
            audio_mel = compute_mel_legacy(audio_raw)
            np.save(str(output_dir / "audio.npy"), audio_mel)
            print(f"  audio (mel): {audio_mel.shape} float16")

        # 卸载 CLAP 释放显存
        del clap_model, clap_processor, resampler
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print("  CLAP unloaded")
    else:
        print("\n── Step 2/6: No audio_raw.npy — skipping ──")

    # ═══════════════════════════════════════════════════════
    #  Step 3: 视觉编码 (Pass 2 — SigLIP + DINOv2, full + fovea)
    # ═══════════════════════════════════════════════════════
    print("\n── Step 3/6: Visual encoding ──")
    vis_siglip_path = output_dir / "visual_siglip.npy"
    vis_dinov2_path = output_dir / "visual_dinov2.npy"
    fov_siglip_path = output_dir / "visual_siglip_fovea.npy"
    fov_dinov2_path = output_dir / "visual_dinov2_fovea.npy"

    need_full = not (_skip("visual_siglip.npy") and _skip("visual_dinov2.npy"))
    need_fovea = not (_skip("visual_siglip_fovea.npy") and _skip("visual_dinov2_fovea.npy"))

    P_sf = P_df = D_sf = D_df = 0

    if need_full or need_fovea:
        siglip, dinov2, siglip_proc, dinov2_proc = load_vision_models(device)

        if need_full:
            P_s, P_d, D_s, D_d = encode_visual(
                video_path, N, siglip, dinov2, siglip_proc, dinov2_proc,
                device, batch_size, output_dir, max_frames,
            )
        else:
            # 读已有全屏特征形状 (用于 spec)
            siglip_feat = np.load(str(vis_siglip_path), mmap_mode="r")
            dinov2_feat = np.load(str(vis_dinov2_path), mmap_mode="r")
            P_s, D_s = siglip_feat.shape[1], siglip_feat.shape[2]
            P_d, D_d = dinov2_feat.shape[1], dinov2_feat.shape[2]
            siglip_feat._mmap.close()
            dinov2_feat._mmap.close()

        if need_fovea:
            P_sf, P_df, D_sf, D_df = encode_fovea(
                video_path, N, gaze, screen_w, screen_h,
                siglip, dinov2, siglip_proc, dinov2_proc,
                device, batch_size, output_dir, max_frames=max_frames,
            )
        else:
            # 读已有焦点特征形状 (用于 spec)
            if fov_siglip_path.exists():
                fov_s = np.load(str(fov_siglip_path), mmap_mode="r")
                P_sf, D_sf = fov_s.shape[1], fov_s.shape[2]
                fov_s._mmap.close()
            if fov_dinov2_path.exists():
                fov_d = np.load(str(fov_dinov2_path), mmap_mode="r")
                P_df, D_df = fov_d.shape[1], fov_d.shape[2]
                fov_d._mmap.close()

        # 卸载视觉模型
        del siglip, dinov2, siglip_proc, dinov2_proc
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print("  Vision models unloaded")
    else:
        # resume: 所有 4 个文件都存在，只读形状
        siglip_feat = np.load(str(vis_siglip_path), mmap_mode="r")
        dinov2_feat = np.load(str(vis_dinov2_path), mmap_mode="r")
        P_s, D_s = siglip_feat.shape[1], siglip_feat.shape[2]
        P_d, D_d = dinov2_feat.shape[1], dinov2_feat.shape[2]
        siglip_feat._mmap.close()
        dinov2_feat._mmap.close()

        fov_s = np.load(str(fov_siglip_path), mmap_mode="r")
        fov_d = np.load(str(fov_dinov2_path), mmap_mode="r")
        P_sf, D_sf = fov_s.shape[1], fov_s.shape[2]
        P_df, D_df = fov_d.shape[1], fov_d.shape[2]
        fov_s._mmap.close()
        fov_d._mmap.close()

    # ═══════════════════════════════════════════════════════
    #  Step 4: 残差目标
    # ═══════════════════════════════════════════════════════
    print("\n── Step 4/6: Residual targets ──")
    if _skip("frame_targets_siglip.npy"):
        shape_s = np.load(str(output_dir / "frame_targets_siglip.npy"), mmap_mode="r").shape
    else:
        shape_s, _ = compute_residual_targets(
            output_dir / "visual_siglip.npy",
            output_dir / "frame_targets_siglip.npy",
        )
    if _skip("frame_targets_dinov2.npy"):
        shape_d = np.load(str(output_dir / "frame_targets_dinov2.npy"), mmap_mode="r").shape
    else:
        shape_d, _ = compute_residual_targets(
            output_dir / "visual_dinov2.npy",
            output_dir / "frame_targets_dinov2.npy",
        )

    # 快速验证：读首尾帧检查
    def _verify_residual(fpath, label):
        arr = np.load(str(fpath), mmap_mode="r")
        nz = (arr[:1] != 0).sum().item() / arr[0].size
        last_zero = np.all(arr[-1] == 0)
        print(f"  {label}: {arr.shape}  float16  "
              f"head_nonzero_ratio={nz:.4f}  "
              f"last_frame_zeros={last_zero}")
        arr._mmap.close()

    _verify_residual(output_dir / "frame_targets_siglip.npy", "frame_targets_siglip")
    _verify_residual(output_dir / "frame_targets_dinov2.npy", "frame_targets_dinov2")

    # ═══════════════════════════════════════════════════════
    #  Step 5: 动作向量 + 时间戳
    # ═══════════════════════════════════════════════════════
    print("\n── Step 5/6: Actions & timestamps ──")
    actions = compute_actions(events, t0_ns, N, screen_w, screen_h)
    np.save(str(output_dir / "actions.npy"), actions)
    print(f"  actions:    {actions.shape}  float32  "
          f"nonzero={(actions != 0).sum()}")

    timestamps = np.array(
        [t0_ns + i * FRAME_INTERVAL_NS for i in range(N)],
        dtype=np.float64,
    )
    np.save(str(output_dir / "timestamps.npy"), timestamps)
    print(f"  timestamps: {timestamps.shape}  float64")

    # ═══════════════════════════════════════════════════════
    #  Step 6: 变长事件序列 (CSR)
    # ═══════════════════════════════════════════════════════
    print("\n── Step 6/6: Events CSR ──")
    flat_path = output_dir / "events_flat.npy"
    off_path = output_dir / "events_offsets.npy"

    if _skip("events_flat.npy") and _skip("events_offsets.npy"):
        flat_mmap = np.load(str(flat_path), mmap_mode="r")
        M = flat_mmap.shape[0]
        flat = flat_mmap
        offsets = np.load(str(off_path))
        flat_is_mmap = True
    else:
        flat, offsets = compute_events_csr(
            events, t0_ns, N, screen_w, screen_h, max_frames,
        )
        M = flat.shape[0]

        # 校验自洽
        assert offsets[-1] == M, \
            f"CSR self-consistency: offsets[-1]={offsets[-1]} != M={M}"
        assert np.all(np.diff(offsets) >= 0), \
            "CSR offsets not monotonic"

        np.save(str(flat_path), flat)
        np.save(str(off_path), offsets)
        flat_is_mmap = False

    # 打印统计
    counts = np.diff(offsets)
    type_ids = flat[:M][:, 0].astype(np.int32) if M > 0 else np.array([], dtype=np.int32)
    type_names = {0: "key_down", 1: "key_up", 2: "mouse_down",
                  3: "mouse_up", 4: "move", 5: "scroll"}
    print(f"  events_flat:    [{M}, {E_DIM}]  float32")
    print(f"  events_offsets: [{N + 1}]  int64")
    print(f"  per-frame: mean={counts.mean():.1f}  max={counts.max()}  "
          f"p50={int(np.median(counts))}  p99={int(np.percentile(counts, 99))}")
    for tid in sorted(type_names):
        n = int((type_ids == tid).sum())
        if n > 0:
            print(f"    type={tid} ({type_names[tid]}): {n:,}")
    if flat_is_mmap:
        flat_mmap._mmap.close()

    # ═══════════════════════════════════════════════════════
    #  spec.json
    # ═══════════════════════════════════════════════════════
    spec = {
        "session_id": session_dir.name,
        "N": N,
        "N_total": total_frames,
        "version": "v3-phase2",
        "screen": {"width": screen_w, "height": screen_h},
        "frame_interval_ns": FRAME_INTERVAL_NS,
        "visual_siglip": {
            "file": "visual_siglip.npy",
            "shape": [N, P_s, D_s],
            "dtype": "float16",
            "model": "siglip-base-patch16-224",
            "notes": "last_hidden_state, includes all patch tokens",
        },
        "visual_dinov2": {
            "file": "visual_dinov2.npy",
            "shape": [N, P_d, D_d],
            "dtype": "float16",
            "model": "dinov2-base",
            "notes": "last_hidden_state, includes CLS + all patch tokens",
        },
        "visual_siglip_fovea": {
            "file": "visual_siglip_fovea.npy",
            "shape": [N, P_sf, D_sf],
            "dtype": "float16",
            "model": "siglip-base-patch16-224",
            "source": "gaze-centered crop",
            "crop_size": 448,
            "gaze_source": "motion_saliency_pseudo",
            "note": "no frame_target; foveated view around predicted gaze point",
        },
        "visual_dinov2_fovea": {
            "file": "visual_dinov2_fovea.npy",
            "shape": [N, P_df, D_df],
            "dtype": "float16",
            "model": "dinov2-base",
            "source": "gaze-centered crop",
            "crop_size": 448,
            "gaze_source": "motion_saliency_pseudo",
            "note": "no frame_target; foveated view around predicted gaze point",
        },
        "frame_targets_siglip": {
            "file": "frame_targets_siglip.npy",
            "shape": [N, P_s, D_s],
            "dtype": "float16",
            "target_type": "residual_delta",
            "formula": "z_{t+1} - z_t  (last frame = zeros)",
        },
        "frame_targets_dinov2": {
            "file": "frame_targets_dinov2.npy",
            "shape": [N, P_d, D_d],
            "dtype": "float16",
            "target_type": "residual_delta",
            "formula": "z_{t+1} - z_t  (last frame = zeros)",
        },
        "audio_clap": {
            "file": "audio_clap.npy",
            "shape": [N, 512],
            "dtype": "float16",
            "model": "laion/clap-htsat-unfused",
            "input_sample_rate": AUDIO_SR,
            "clap_sample_rate": CLAP_SR,
            "chunk_ms": 100,
            "notes": "CLAP audio embedding (audio tower pooled output)",
        } if has_audio else None,
        "gaze_pseudo": {
            "file": "gaze_pseudo.npy",
            "shape": [N, 2],
            "dtype": "float32",
            "algorithm": "motion_saliency_argmax",
            "formula": "argmax( GaussianBlur(|frame_t - frame_{t-1}|) )",
            "normalization": "[0, 1] relative to frame dimensions",
        },
        "actions": {
            "file": "actions.npy",
            "shape": [N, A_DIM],
            "dtype": "float32",
            "dim_order": [
                "mouse_dx", "mouse_dy", "mouse_x", "mouse_y",
                "mouse_path_len", "scroll_dy",
                "left_held", "right_held", "left_click", "right_click",
            ] + [f"{k}_{suffix}" for k in KEY_ORDER for suffix in ["down", "held"]],
        },
        "events": {
            "flat": "events_flat.npy",
            "offsets": "events_offsets.npy",
            "shape_flat": [M, E_DIM],
            "shape_offsets": [N + 1],
            "event_dim": E_DIM,
            "format": "CSR (Compressed Sparse Row)",
            "dtype_flat": "float32",
            "dtype_offsets": "int64",
            "dim_spec": [
                "type_id", "key_id", "button_id",
                "x_norm", "y_norm", "dx_norm", "dy_norm",
                "path_len_norm", "scroll_dy", "dt_ms",
            ],
            "type_map": {
                "0": "key_down", "1": "key_up",
                "2": "mouse_down", "3": "mouse_up",
                "4": "move", "5": "scroll",
            },
            "button_map": {"0": "left", "1": "right", "2": "middle"},
            "key_count": K,
            "move_merge": "segments flushed on non-move events; cursor persists across frames",
            "down_up_rule": "down and up events each fall into their own frame by ts_ns, never paired",
            "stats": {
                "M_total": M,
                "per_frame_mean": float(counts.mean()),
                "per_frame_max": int(counts.max()),
                "per_frame_p50": float(np.median(counts)),
                "per_frame_p99": float(np.percentile(counts, 99)),
            },
        },
        "timestamps": {
            "file": "timestamps.npy",
            "shape": [N],
            "dtype": "float64",
            "clock": "perf_counter_ns",
            "t0_ns": t0_ns,
        },
    }
    spec = {k: v for k, v in spec.items() if v is not None}
    (output_dir / "spec.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # ── 统计 ──
    total_bytes = sum(
        f.stat().st_size for f in output_dir.iterdir() if f.is_file()
    )
    print(f"\n── Done: {output_dir}")
    print(f"Total: {total_bytes / (1024**3):.2f} GB")


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="预处理 session → 训练 .npy (CLAP + SigLIP + DINOv2 + gaze + 残差)")
    parser.add_argument("session", nargs="?", default=None,
                        help="Session 目录名 (默认自动选择最新)")
    parser.add_argument("--data-dir", default=str(SESSION_DIR),
                        help="session 所在目录")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="批大小 (默认 16)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="推理设备")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="限制编码帧数 (测试用)")
    parser.add_argument("--legacy-audio", action="store_true",
                        help="同时保存 mel 频谱 audio.npy (兼容旧 trainer)")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已有输出文件的步骤")
    args = parser.parse_args()

    # 检查路径
    if not SIGLIP_PATH.exists():
        print(f"ERROR: SigLIP 路径不存在: {SIGLIP_PATH}", file=sys.stderr)
        sys.exit(1)
    if not DINOV2_PATH.exists():
        print(f"ERROR: DINOv2 路径不存在: {DINOV2_PATH}", file=sys.stderr)
        sys.exit(1)
    if not CLAP_SNAPSHOT.exists():
        print(f"ERROR: CLAP 路径不存在: {CLAP_SNAPSHOT}", file=sys.stderr)
        sys.exit(1)

    if args.session:
        session_dir = Path(args.data_dir) / args.session
    else:
        session_dir = find_latest_session(args.data_dir)
        if session_dir is None:
            print(f"ERROR: 在 {args.data_dir} 下找不到 session_* 目录。", file=sys.stderr)
            sys.exit(1)
        print(f"Auto session: {session_dir.name}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DATASET_DIR / ts / "processed"

    t_start = time.perf_counter()
    preprocess_session(
        session_dir, output_dir, args.device, args.batch_size,
        args.max_frames, args.legacy_audio, args.resume,
    )
    elapsed = time.perf_counter() - t_start
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
