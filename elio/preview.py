#!/usr/bin/env python3
"""
采集数据预览工具 —— 回放 10Hz 截屏帧，叠加键鼠事件，同步播放音频。

用法:
    python preview.py                                    # 自动选最新 session
    python preview.py session_20260614_214000             # 指定 session 目录
    python preview.py --speed 2.0                         # 2 倍速
    python preview.py --no-audio                          # 禁用音频

键盘控制:
    空格             暂停 / 播放
    ← → / A D        前后 1 帧
    ↑ ↓              前后 30 帧
    1 / 2 / 3 / 4    速度 0.5x / 1x / 2x / 4x
    m                静音 / 取消静音
    0                跳回开头
    q / Esc          退出

依赖: pip install opencv-python numpy sounddevice
"""

import argparse
import collections
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import sounddevice as sd
    HAS_AUDIO = True
except (ImportError, OSError):
    HAS_AUDIO = False

# ── 颜色常量 (BGR) ────────────────────────────────────
COLOR_MOUSE = (0, 255, 255)
COLOR_CLICK_DOWN = (0, 0, 255)
COLOR_CLICK_UP = (0, 255, 0)
COLOR_SCROLL = (255, 0, 255)
COLOR_KEY = (180, 180, 255)
COLOR_PROGRESS_BG = (80, 80, 80)
COLOR_PROGRESS_FG = (0, 220, 0)
COLOR_BAR_BG = (30, 30, 30)
COLOR_TEXT = (220, 220, 220)
COLOR_AUDIO_ON = (0, 220, 0)
COLOR_AUDIO_OFF = (80, 80, 80)

FRAME_SAMPLES = 1600          # 100ms @ 16kHz
AUDIO_SR = 16000


# ── 音频重采样 ────────────────────────────────────────

def resample_chunk(chunk: np.ndarray, speed: float) -> np.ndarray:
    """线性插值重采样：1600 samples → 1600/speed samples。"""
    if speed == 1.0 or len(chunk) == 0:
        return chunk.copy()
    src_len = len(chunk)
    target_len = max(1, int(src_len / speed))
    src_idx = np.linspace(0, src_len - 1, target_len)
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, src_len - 1)
    frac = (src_idx - lo).astype(np.float32)
    return (chunk[lo].astype(np.float32) * (1 - frac)
            + chunk[hi].astype(np.float32) * frac).astype(np.int16)


# ── 音频播放器 ────────────────────────────────────────

class AudioPlayer:
    """后台 sounddevice OutputStream 播放音频，ring buffer 解耦主循环。

    关键设计:
      - blocksize=400 (25ms) — 每帧 4 次回调，低延迟
      - 启动前预填充 5 帧音频，避免冷启动欠载
      - 主循环每帧 push_frame() 推入 1600/speed 样本
    """

    BLOCKSIZE = 400          # 每次回调 400 samples (25ms @ 16kHz)

    def __init__(self, audio: np.ndarray | None):
        self._audio = audio if audio is not None else np.array([], dtype=np.int16)
        self._total_samples = len(self._audio)
        self._cursor = 0
        self._buffer: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._speed = 1.0
        self._muted = False
        self._active = False
        self._consumed_samples = 0  # 声卡实际已输出的样本数（不含静音填充）

    @property
    def available(self) -> bool:
        return HAS_AUDIO and self._total_samples > 0

    @property
    def muted(self) -> bool:
        return self._muted

    @muted.setter
    def muted(self, v: bool) -> None:
        self._muted = v
        if v:
            with self._lock:
                self._buffer.clear()

    @property
    def position_samples(self) -> int:
        """声卡实际已播放的样本数（不含静音填充）。用作音画同步的主时钟。"""
        return self._consumed_samples

    # ── 播放控制 ──

    def start(self) -> None:
        if not self.available:
            return
        self._consumed_samples = 0
        # 预填充静音 + 第 0 帧实际音频，避免冷启动欠载
        for _ in range(5):
            with self._lock:
                self._buffer.extend([0] * self.BLOCKSIZE)
        self.push_frame(0)
        self._cursor = 0

        self._stream = sd.OutputStream(
            samplerate=AUDIO_SR, channels=1, dtype='int16',
            blocksize=self.BLOCKSIZE, latency='low',
            callback=self._callback,
        )
        self._stream.start()
        self._active = True

    def set_speed(self, speed: float) -> None:
        self._speed = speed

    def push_frame(self, frame_idx: int) -> None:
        """推入 frame_idx 对应的 100ms 音频（已重采样到当前速度）。"""
        if not self._active or self._muted:
            return
        start = frame_idx * FRAME_SAMPLES
        end = start + FRAME_SAMPLES
        if start < self._total_samples:
            chunk = self._audio[start:min(end, self._total_samples)]
            if len(chunk) < FRAME_SAMPLES:
                chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
            chunk = resample_chunk(chunk, self._speed)
        else:
            chunk = np.zeros(max(1, int(FRAME_SAMPLES / self._speed)),
                             dtype=np.int16)
        with self._lock:
            self._buffer.extend(chunk.tolist())
        self._cursor = end

    def seek(self, frame_idx: int) -> None:
        """跳转到指定帧，清空缓冲并预推后续帧。"""
        self._cursor = frame_idx * FRAME_SAMPLES
        self._consumed_samples = frame_idx * FRAME_SAMPLES
        with self._lock:
            self._buffer.clear()
        # 预推当前帧 + 后续 2 帧，避免 seek 后缓冲区欠载
        for i in range(3):
            self.push_frame(frame_idx + i)

    def stop(self) -> None:
        self._active = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _callback(self, outdata, frames, time_info, status):
        """OutputStream 回调：从 ring buffer 取数据，跟踪实际播放位置。"""
        with self._lock:
            avail = len(self._buffer)
            take = min(avail, frames)
            if take > 0:
                chunk = np.array([self._buffer.popleft() for _ in range(take)],
                                 dtype=np.int16)
                outdata[:take, 0] = chunk
                self._consumed_samples += take  # 只计数实际音频，不含静音
            if take < frames:
                outdata[take:, 0] = 0
                # 静音填充不计入 _consumed_samples —— 暂停时时钟冻结


# ═══════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════

class FrameCache:
    """后台线程预加载视频帧到内存，消除 seek 解码抖动。

    使用独立的 cv2.VideoCapture 实例（与主线程的 fallback cap 分离），
    避免线程间竞争解码器状态。

    维护一个以当前播放位置为中心的滑动窗口缓存。
    """

    def __init__(self, video_path: str, total_frames: int, cache_size: int = 90):
        self._video_path = video_path
        self._total = total_frames
        self._cache: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self._next = 0           # 下一个要预加载的帧号
        self._cache_size = cache_size
        self._running = False
        self._thread: threading.Thread | None = None

        # 预加载线程专用 VideoCapture
        self._preload_cap: cv2.VideoCapture | None = None
        # 主线程 fallback 专用 VideoCapture（延迟创建）
        self._main_cap: cv2.VideoCapture | None = None

    @property
    def total(self) -> int:
        return self._total

    def start(self) -> None:
        self._preload_cap = cv2.VideoCapture(self._video_path)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._preload_cap is not None:
            self._preload_cap.release()
            self._preload_cap = None
        if self._main_cap is not None:
            self._main_cap.release()
            self._main_cap = None

    def set_anchor(self, idx: int) -> None:
        """将预加载窗口定位到指定帧附近。"""
        with self._lock:
            self._next = max(0, min(idx, self._total))

    def get(self, idx: int) -> np.ndarray | None:
        """获取帧。缓存命中直接返回，未命中用主线程 VideoCapture seek 读取。"""
        # 先查缓存
        with self._lock:
            img = self._cache.get(idx)
            if img is not None:
                self._next = max(self._next, idx + 1)
                return img

        # 缓存未命中 —— 主线程自己的 cap seek 读取
        if idx < 0 or idx >= self._total:
            return None
        if self._main_cap is None:
            self._main_cap = cv2.VideoCapture(self._video_path)
        cap = self._main_cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            with self._lock:
                self._cache[idx] = frame
                self._next = max(self._next, idx + 1)
            return frame
        return None

    def _loop(self) -> None:
        """预加载线程：用独立 VideoCapture 顺序读取前方帧。"""
        cap = self._preload_cap
        while self._running:
            with self._lock:
                idx = self._next
            if idx >= self._total:
                time.sleep(0.02)
                continue

            # 顺序读取（比 seek 更快）
            cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if cur_pos != idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            ret, frame = cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._cache[idx] = frame
                    self._next = idx + 1
                    # 淘汰远离当前窗口的旧帧
                    half = self._cache_size // 2
                    anchor = max(0, self._next - half)
                    stale = [k for k in self._cache
                             if k < anchor - half
                             or k > anchor + self._cache_size]
                    for k in stale:
                        del self._cache[k]
            else:
                # 读取失败，跳过
                with self._lock:
                    self._next = idx + 1


def find_latest_session(data_dir: str = ".") -> Path | None:
    """自动找最新的 session_* 目录。"""
    base = Path(data_dir)
    sessions = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("session_")],
        key=lambda d: d.name, reverse=True,
    )
    return sessions[0] if sessions else None


def load_session(session_dir: Path) -> dict:
    """加载 session 数据。

    Returns:
        {
            'video_path': Path,         # video.mp4 路径
            'total_frames': int,        # 视频总帧数
            'events': [{...}, ...],     # 按 ts_ns 排序的事件列表
            'audio': np.ndarray | None, # 音频数据 1D int16
            'session_id': str,
        }
    """
    if not session_dir.exists():
        raise FileNotFoundError(f"Session 目录不存在: {session_dir}")

    # 查找视频文件
    video_path = None
    for ext in ('.mp4', '.avi'):
        candidate = session_dir / f"video{ext}"
        if candidate.exists():
            video_path = candidate
            break

    if video_path is None:
        raise FileNotFoundError(
            f"找不到视频文件: {session_dir}/video.mp4 (或 .avi)")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 事件列表
    events: list[dict] = []
    events_path = session_dir / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    events.sort(key=lambda e: e["ts_ns"])

    # 音频文件
    audio = None
    audio_path = session_dir / "audio_raw.npy"
    if audio_path.exists():
        audio = np.load(str(audio_path)).astype(np.int16)  # [N, 1600] or flat
        if audio.ndim == 2:
            audio = audio.reshape(-1)  # flatten to 1D

    return {
        "video_path": video_path,
        "total_frames": total_frames,
        "events": events,
        "audio": audio,
        "session_id": session_dir.name,
    }


def format_ns(ns: int) -> str:
    """纳秒 → HH:MM:SS.mmm。"""
    sec = ns / 1e9
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def draw_crosshair(img: np.ndarray, x: int, y: int, color: tuple, size: int = 12) -> None:
    h, w = img.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        cv2.line(img, (x - size, y), (x + size, y), color, 2)
        cv2.line(img, (x, y - size), (x, y + size), color, 2)
        cv2.circle(img, (x, y), 15, color, 1)


def draw_click(img: np.ndarray, x: int, y: int, color: tuple) -> None:
    h, w = img.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(img, (x, y), 7, color, -1)
        cv2.circle(img, (x, y), 15, color, 2)


def render_status_bar(
    img: np.ndarray,
    frame_idx: int,
    total_frames: int,
    elapsed_s: float,
    speed: float,
    paused: bool,
    recent_events: list[dict],
    session_id: str,
    audio_available: bool = False,
    audio_muted: bool = False,
) -> None:
    h, w = img.shape[:2]
    bar_h = 130
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), COLOR_BAR_BG, -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    y = h - bar_h + 20

    # 状态行
    state = "PAUSED" if paused else "PLAYING"
    audio_str = ""
    if audio_available:
        audio_str = " [MUTED]" if audio_muted else " [AUDIO]"
    cv2.putText(img,
                f"[{session_id}]  Frame: {frame_idx + 1}/{total_frames}  "
                f"Elapsed: {format_ns(int(elapsed_s * 1e9))}  "
                f"Speed: {speed}x  {state}{audio_str}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    y += 22

    # 最近事件
    for ev in recent_events[-5:]:
        t = ev["type"]
        if t == "mouse_move":
            text = f"  move       ({ev['x']}, {ev['y']})"
        elif t == "mouse_click":
            text = f"  click      {ev['button']} {'down' if ev['pressed'] else 'up'}  ({ev['x']}, {ev['y']})"
        elif t == "mouse_scroll":
            text = f"  scroll     dx={ev['dx']} dy={ev['dy']}  ({ev['x']}, {ev['y']})"
        elif t == "key_press":
            text = f"  key_down   '{ev['key']}'"
        elif t == "key_release":
            text = f"  key_up     '{ev['key']}'"
        else:
            text = f"  {t}"
        cv2.putText(img, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1)
        y += 18

    # 进度条
    bar_y = h - 6
    bar_x = 5
    bar_w = w - 10
    prog = frame_idx / max(total_frames - 1, 1)
    cv2.rectangle(img, (bar_x, bar_y - 4), (bar_x + bar_w, bar_y),
                  COLOR_PROGRESS_BG, -1)
    cv2.rectangle(img, (bar_x, bar_y - 4), (bar_x + int(bar_w * prog), bar_y),
                  COLOR_PROGRESS_FG, -1)


def run(session_data: dict, start_speed: float = 1.0,
        enable_audio: bool = True) -> None:
    video_path = session_data["video_path"]
    events = session_data["events"]
    session_id = session_data["session_id"]
    total = session_data["total_frames"]

    if total == 0:
        print("ERROR: 视频中没有帧。")
        return

    # 帧间隔（10Hz）
    FRAME_INTERVAL_NS = 100_000_000  # 100ms → ns

    # 以第一个事件的时间戳为参考起点
    t0_ns = events[0]["ts_ns"] if events else 0

    # ── 帧预加载缓存 ──
    cache = FrameCache(str(video_path), total)
    cache.start()

    # ── 音频播放器 ──
    player = AudioPlayer(session_data.get("audio")) if enable_audio else AudioPlayer(None)
    has_audio = player.available
    if has_audio:
        player.start()
        audio_len_s = len(session_data["audio"]) / AUDIO_SR
        print(f"Audio: {audio_len_s:.1f}s  (16kHz mono)")
    elif enable_audio:
        if HAS_AUDIO:
            print("Audio: session 中无 audio_raw.npy，仅播放画面")
        else:
            print("Audio: sounddevice 未安装 (pip install sounddevice)")

    print(f"Session: {session_id}")
    print(f"Frames: {total}  |  Events: {len(events)}")
    print(f"Controls: Space=pause  Arrows=seek  1-4=speed  "
          f"m=mute  q=quit\n")

    frame_idx = 0
    prev_frame = -1          # 上一帧索引，用于检测 seek（无音频模式）
    paused = False
    speed = start_speed
    event_cursor = 0
    manual_seek = False      # 标记用户手动跳转，跳过本轮的音频时钟同步

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Preview", 1280, 800)

    last_tick = time.perf_counter()

    try:
        while True:
            # ── 帧号确定 ──
            # 有音频时：以音频播放位置为主时钟，视频追踪音频
            # 无音频时：手动计数 + waitKey 计时
            if has_audio and not paused and not manual_seek:
                audio_frame = player.position_samples // FRAME_SAMPLES
                frame_idx = min(audio_frame, total - 1)
            else:
                frame_idx = max(0, min(frame_idx, total - 1))

            manual_seek = False

            # 检测 seek（帧不连续或倒退）—— 仅无音频模式需要
            if not has_audio:
                if frame_idx != prev_frame + 1 and prev_frame >= 0 and not paused:
                    manual_seek = True
                if prev_frame >= 0 and frame_idx < prev_frame:
                    manual_seek = True

            # 当前帧的估计时间戳
            frame_ts_ns = t0_ns + frame_idx * FRAME_INTERVAL_NS
            prev_ts_ns = t0_ns + max(0, frame_idx - 1) * FRAME_INTERVAL_NS

            # ── 加载帧（缓存优先，内部 VideoCapture 兜底）──
            img = cache.get(frame_idx)
            if img is None:
                # 视频文件可能损坏或帧索引越界
                prev_frame = frame_idx
                if not has_audio and not paused:
                    frame_idx += 1
                continue
            img_h, img_w = img.shape[:2]

            # 更新预加载窗口
            cache.set_anchor(frame_idx)

            # 收集本帧区间的事件
            window_events: list[dict] = []
            while event_cursor < len(events) and events[event_cursor]["ts_ns"] <= frame_ts_ns:
                ev = events[event_cursor]
                if ev["ts_ns"] >= prev_ts_ns:
                    window_events.append(ev)
                event_cursor += 1

            # 如果 seek 导致 event_cursor 超前，回退
            if event_cursor > 0 and events[event_cursor - 1]["ts_ns"] > frame_ts_ns:
                lo, hi = 0, len(events)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if events[mid]["ts_ns"] <= frame_ts_ns:
                        lo = mid + 1
                    else:
                        hi = mid
                event_cursor = lo
                window_events.clear()
                for i in range(event_cursor):
                    if events[i]["ts_ns"] >= prev_ts_ns:
                        window_events.append(events[i])

            # ── 音频：推送当前帧（seek 已在键盘处理中完成）──
            if has_audio and not player.muted:
                player.push_frame(frame_idx)

            # ── 叠加事件到画面 ──
            mx, my = -1, -1
            for ev in window_events:
                t = ev["type"]
                if t == "mouse_move":
                    mx, my = ev["x"], ev["y"]
                    if 0 <= mx < img_w and 0 <= my < img_h:
                        cv2.circle(img, (mx, my), 2, COLOR_MOUSE, -1)
                elif t == "mouse_click":
                    x, y = ev["x"], ev["y"]
                    color = COLOR_CLICK_DOWN if ev["pressed"] else COLOR_CLICK_UP
                    draw_click(img, x, y, color)
                    mx, my = x, y
                elif t == "mouse_scroll":
                    draw_click(img, ev["x"], ev["y"], COLOR_SCROLL)
                    mx, my = ev["x"], ev["y"]

            # 当前鼠标十字准星
            if mx >= 0 and my >= 0:
                draw_crosshair(img, mx, my, COLOR_MOUSE)

            # 状态栏
            render_status_bar(img, frame_idx, total,
                              (frame_ts_ns - t0_ns) / 1e9,
                              speed, paused, window_events, session_id,
                              audio_available=has_audio,
                              audio_muted=player.muted)

            cv2.imshow("Preview", img)

            # ── 帧率控制（仅无音频模式，有音频时由声卡时钟驱动）──
            if not has_audio:
                base_interval = 1.0 / (10.0 * speed)
                elapsed = time.perf_counter() - last_tick
                wait_ms = max(1, int((base_interval - elapsed) * 1000)) if not paused else 30
            else:
                wait_ms = 1  # 有音频时只做最小等待，保持 UI 响应
            last_tick = time.perf_counter()

            key = cv2.waitKey(wait_ms)
            key_ascii = key & 0xFF

            # ── 键盘 ──
            if key_ascii == ord("q") or key_ascii == 27:
                break
            elif key_ascii == ord(" "):
                paused = not paused
                last_tick = time.perf_counter()
            elif key_ascii == ord("m"):
                if has_audio:
                    player.muted = not player.muted
                    if not player.muted:
                        player.seek(frame_idx)
            elif key == 0x250000 or key_ascii == ord("a"):
                frame_idx -= 1
                manual_seek = True
                if has_audio:
                    player.seek(frame_idx)
                event_cursor = 0
                last_tick = time.perf_counter()
            elif key == 0x270000 or key_ascii == ord("d"):
                frame_idx += 1
                manual_seek = True
                if has_audio:
                    player.seek(frame_idx)
                event_cursor = 0
                last_tick = time.perf_counter()
            elif key == 0x260000:
                frame_idx += 30
                manual_seek = True
                if has_audio:
                    player.seek(frame_idx)
                event_cursor = 0
                last_tick = time.perf_counter()
            elif key == 0x280000:
                frame_idx -= 30
                manual_seek = True
                if has_audio:
                    player.seek(frame_idx)
                event_cursor = 0
                last_tick = time.perf_counter()
            elif key_ascii == ord("1"):
                speed = 0.5
                if has_audio:
                    player.set_speed(0.5)
            elif key_ascii == ord("2"):
                speed = 1.0
                if has_audio:
                    player.set_speed(1.0)
            elif key_ascii == ord("3"):
                speed = 2.0
                if has_audio:
                    player.set_speed(2.0)
            elif key_ascii == ord("4"):
                speed = 4.0
                if has_audio:
                    player.set_speed(4.0)
            elif key_ascii == ord("0"):
                frame_idx = 0
                manual_seek = True
                if has_audio:
                    player.seek(0)
                event_cursor = 0
                last_tick = time.perf_counter()

            prev_frame = frame_idx

            # ── 帧推进（仅无音频模式，有音频时由音频位置驱动）──
            if not has_audio and not paused:
                frame_idx += 1
                if frame_idx >= total:
                    print("Reached end.")
                    break
            # 有音频时检查是否播放完毕
            if has_audio and not paused and frame_idx >= total - 1:
                print("Reached end.")
                break
    finally:
        player.stop()
        cache.stop()

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="预览 10Hz 采集数据（含音频同步播放）")
    parser.add_argument("session", nargs="?", default=None, help="Session 目录名")
    parser.add_argument("--speed", "-s", type=float, default=1.0, help="播放速度")
    parser.add_argument("--no-audio", action="store_true", help="禁用音频播放")
    parser.add_argument("--data-dir", default=".", help="session 所在目录 (默认: 当前目录)")
    args = parser.parse_args()

    if args.session:
        session_dir = Path(args.data_dir) / args.session
    else:
        session_dir = find_latest_session(args.data_dir)
        if session_dir is None:
            print("ERROR: 找不到 session 目录。请指定路径。", file=sys.stderr)
            sys.exit(1)
        print(f"Auto: {session_dir}")

    data = load_session(session_dir)
    run(data, start_speed=args.speed, enable_audio=not args.no_audio)


if __name__ == "__main__":
    main()
