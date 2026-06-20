"""
10Hz 屏幕采集 + 键鼠事件记录。
mss 快速抓屏 + 提取真实硬件光标位图合成，pynput 记录键鼠事件到 JSONL。
"""

import json
import os
import queue
import sys
import threading
import time
import ctypes
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api
import mss
from PIL import Image
from pynput import mouse, keyboard


# ── DPI 设置 ──────────────────────────────────────────
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except AttributeError:
    ctypes.windll.user32.SetProcessDPIAware()


# ── 光标提取 ──────────────────────────────────────────

# ── BITMAP 尺寸查询 (ctypes 直调 GDI, 避免 pywin32 GetObject 版本兼容问题) ──

class _BITMAP(ctypes.Structure):
    _fields_ = [
        ("bmType", ctypes.c_long),
        ("bmWidth", ctypes.c_long),
        ("bmHeight", ctypes.c_long),
        ("bmWidthBytes", ctypes.c_long),
        ("bmPlanes", ctypes.c_ushort),
        ("bmBitsPixel", ctypes.c_ushort),
        ("bmBits", ctypes.c_void_p),
    ]


def _get_bitmap_size(hbm) -> tuple[int, int]:
    """用 ctypes 调 GDI GetObjectW 获取位图宽高。

    必须用 c_void_p 传柄 —— 64 位 HGDIOBJ 是 8 字节指针，
    ctypes 默认 int 只有 4 字节会截断。
    """
    bm = _BITMAP()
    gdi32 = ctypes.windll.gdi32
    gdi32.GetObjectW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    ret = gdi32.GetObjectW(ctypes.c_void_p(int(hbm)),
                           ctypes.sizeof(bm), ctypes.byref(bm))
    if ret == 0:
        return 32, 32  # 极端兜底
    return int(bm.bmWidth), int(bm.bmHeight)

def _extract_cursor_rgba() -> tuple[Image.Image, int, int] | None:
    """提取当前硬件光标为带透明通道的 PIL RGBA Image。

    用两次 DrawIconEx（黑底/白底）差分算出 alpha 通道，
    适用于所有光标类型（单色、彩色、动画首帧）。

    Returns:
        (cursor_img, hotspot_x, hotspot_y) 或 None（光标隐藏时）
    """
    cursor_info = win32gui.GetCursorInfo()
    flags, hcursor_shared, _ = cursor_info
    if not (flags & win32con.CURSOR_SHOWING):
        return None

    # CopyIcon 创建私有副本，防止两次 DrawIconEx 之间光标变化导致句柄失效
    hcursor = win32gui.CopyIcon(hcursor_shared)
    icon_info = None
    try:
        icon_info = win32gui.GetIconInfo(hcursor)
        _, hx, hy, hbmMask, hbmColor = icon_info

        # 查询实际光标尺寸（ctypes 直调 GetObjectW，避免 pywin32 GetObject 返回 PyHANDLE）
        if hbmColor:
            w, h_ = _get_bitmap_size(hbmColor)
        else:
            w, h_ = _get_bitmap_size(hbmMask)
            h_ //= 2  # 单色光标 AND/XOR 上下叠放

        w = max(w, 1)
        h = max(h_, 1)

        screen_dc = win32gui.GetDC(0)
        screen_dc_obj = win32ui.CreateDCFromHandle(screen_dc)

        def _render_on(bg_rgb: int) -> np.ndarray:
            """在纯色背景上绘制光标，返回 BGRA numpy 数组。"""
            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(screen_dc_obj, w, h)
            dc = screen_dc_obj.CreateCompatibleDC()
            dc.SelectObject(bmp)
            dc.FillSolidRect((0, 0, w, h), bg_rgb)
            win32gui.DrawIconEx(
                dc.GetSafeHdc(), 0, 0, hcursor, w, h,
                0, None, win32con.DI_NORMAL,
            )
            bits = bmp.GetBitmapBits(True)
            arr = np.frombuffer(bits, dtype=np.uint8).reshape(h, w, 4).copy()
            dc.DeleteDC()
            win32gui.DeleteObject(bmp.GetHandle())
            return arr

        black_arr = _render_on(0x00000000)  # 黑底 → BGRA
        white_arr = _render_on(0x00FFFFFF)  # 白底 → BGRA

        win32gui.ReleaseDC(0, screen_dc)

    finally:
        # 清理 GDI 资源
        if icon_info is not None:
            for idx in (3, 4):
                hbm = icon_info[idx] if idx < len(icon_info) else None
                if hbm:
                    win32gui.DeleteObject(hbm)
        win32gui.DestroyIcon(hcursor)

    # alpha = 255 - 黑白像素差异（diff 越大 = 背景影响越大 = 越透明）
    diff = np.abs(black_arr.astype(np.int16) - white_arr.astype(np.int16))
    raw_alpha = np.clip(diff.max(axis=2), 0, 255).astype(np.uint8)
    alpha = 255 - raw_alpha

    # 颜色重建：黑底版 / (alpha/255) 消除黑色背景的混入
    #   黑底渲染: result = cursor_color * alpha/255 + black * (1-alpha/255)
    #   所以:     cursor_color = result / (alpha/255)
    alpha_float = alpha.astype(np.float32).clip(1, 255) / 255.0
    rgb_bgr = (black_arr[:, :, :3].astype(np.float32) / alpha_float[:, :, None])
    rgb = np.clip(rgb_bgr, 0, 255).astype(np.uint8)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb[:, :, ::-1]  # BGR → RGB
    rgba[:, :, 3] = alpha

    cursor_img = Image.fromarray(rgba, 'RGBA')
    return cursor_img, hx, hy


# ── 截屏 ──────────────────────────────────────────────

def capture_frame(monitor: dict) -> np.ndarray:
    """mss 抓屏 → 合成硬件光标 → 返回 BGR numpy 数组 (供 VideoWriter 编码)。"""
    with mss.MSS() as sct:
        sct_img = sct.grab(monitor)

    # mss 返回 BGRA → PIL RGB
    pil_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

    # 提取光标并合成
    cursor_data = _extract_cursor_rgba()
    if cursor_data is not None:
        cursor_img, hx, hy = cursor_data
        # 光标屏幕坐标 → 图像内坐标
        cx, cy = win32api.GetCursorPos()
        paste_x = cx - monitor["left"] - hx
        paste_y = cy - monitor["top"] - hy
        # 合成（带 alpha 通道）
        pil_img.paste(cursor_img, (paste_x, paste_y), cursor_img)

    # PIL RGB → numpy BGR（OpenCV VideoWriter 需要 BGR 格式）
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── 视频编码线程 ──────────────────────────────────────

class VideoWriterThread:
    """后台线程：从 frame_queue 取帧，写入 cv2.VideoWriter。

    设计要点:
      - 有界队列 (maxsize=60 ≈ 6 秒缓冲)，防止内存膨胀
      - 队列满时 put_nowait 丢弃最旧帧，保证编码延迟不阻塞主循环
      - 线程在 stop() 后排空队列、释放 writer，防止视频损坏
    """

    def __init__(self, path: str, fps: float, width: int, height: int):
        self._queue: queue.Queue = queue.Queue(maxsize=60)
        self._path = path
        self._fps = fps
        self._width = width
        self._height = height
        self._writer: cv2.VideoWriter | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._frames_written = 0
        self._frames_dropped = 0
        self._actual_path = ""

    # ── 公共 API ──

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped

    @property
    def actual_path(self) -> str:
        return self._actual_path

    def start(self) -> None:
        """尝试编码器优先级列表，创建 VideoWriter 并启动后台线程。"""
        fourcc_list = [
            (cv2.VideoWriter_fourcc(*'avc1'), '.mp4', 'H.264/MP4'),
            (cv2.VideoWriter_fourcc(*'mp4v'), '.mp4', 'MPEG-4/MP4'),
            (cv2.VideoWriter_fourcc(*'XVID'), '.avi', 'Xvid/AVI'),
        ]
        writer = None
        for fourcc, ext, label in fourcc_list:
            candidate = str(self._path) + ext
            try:
                w = cv2.VideoWriter(candidate, fourcc, self._fps,
                                    (self._width, self._height))
                if w.isOpened():
                    writer = w
                    self._actual_path = candidate
                    print(f"[Video] {label} → {self._actual_path}")
                    break
                else:
                    w.release()
            except Exception:
                pass

        if writer is None:
            raise RuntimeError(
                "无法创建 VideoWriter。请安装 OpenCV 的 H.264 支持。\n"
                "  pip install opencv-python  (已含 mp4v)"
            )

        self._writer = writer
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def push(self, frame: np.ndarray) -> None:
        """非阻塞推入帧。队列满时丢弃并计数。"""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._frames_dropped += 1
            if self._frames_dropped % 100 == 1:
                print(f"[Video] 警告: 已丢弃 {self._frames_dropped} 帧 (编码落后于采集)")

    def stop(self) -> None:
        """优雅停止：通知线程停止 → 排空队列 → 释放 writer。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        # 兜底：强制释放（线程未正常退出时）
        if self._writer is not None:
            try:
                if self._writer.isOpened():
                    self._writer.release()
            except Exception:
                pass
            self._writer = None

    # ── 内部 ──

    def _loop(self) -> None:
        """后台编码循环。"""
        writer = self._writer
        q = self._queue
        while self._running:
            try:
                frame = q.get(timeout=0.3)
                writer.write(frame)
                self._frames_written += 1
            except queue.Empty:
                continue

        # 排空剩余帧
        drained = 0
        while True:
            try:
                frame = q.get_nowait()
                writer.write(frame)
                self._frames_written += 1
                drained += 1
            except queue.Empty:
                break

        if drained:
            print(f"[Video] 排空 {drained} 帧")

        writer.release()
        print(f"[Video] 写入 {self._frames_written} 帧, "
              f"丢弃 {self._frames_dropped} 帧")


# ── 键鼠事件回调 ──────────────────────────────────────

def key_to_str(key) -> str:
    """pynput 键对象 → 字符串。"""
    try:
        if hasattr(key, 'char') and key.char is not None:
            return key.char
    except Exception:
        pass
    if hasattr(key, 'name') and key.name is not None:
        return key.name
    return str(key)


class EventCollector:
    """收集 pynput 键鼠回调，推入线程安全队列。"""

    def __init__(self, event_queue: queue.Queue, stop_event: threading.Event):
        self._queue = event_queue
        self._stop_event = stop_event
        self._kb_listener: keyboard.Listener | None = None
        self._ms_listener: mouse.Listener | None = None

    def start(self) -> None:
        self._kb_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._ms_listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self._kb_listener.start()
        self._ms_listener.start()

    def stop(self) -> None:
        if self._kb_listener:
            self._kb_listener.stop()
        if self._ms_listener:
            self._ms_listener.stop()

    def _push(self, event: dict) -> None:
        if not self._stop_event.is_set():
            event["ts_ns"] = time.perf_counter_ns()
            self._queue.put(event)

    def _on_move(self, x: int, y: int) -> None:
        self._push({"type": "mouse_move", "x": x, "y": y})

    def _on_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        btn_name = button.name.lower() if hasattr(button, "name") else str(button)
        self._push({"type": "mouse_click", "x": x, "y": y,
                     "button": btn_name, "pressed": pressed})

    def _on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self._push({"type": "mouse_scroll", "x": x, "y": y, "dx": dx, "dy": dy})

    def _on_press(self, key) -> None:
        k = key_to_str(key)
        if k:
            self._push({"type": "key_press", "key": k})

    def _on_release(self, key) -> None:
        k = key_to_str(key)
        if k:
            self._push({"type": "key_release", "key": k})


# ── 音频采集（WASAPI loopback, 16kHz 单声道） ─────────

_HAS_SD = False
try:
    import sounddevice as _sd
    _HAS_SD = True
except ImportError:
    pass


class SystemAudioCapture:
    """录制系统音频。

    后端优先级:
      1. WASAPI loopback  — 直接抓默认播放设备输出
      2. VB-Cable / Stereo Mix — 虚拟音频录制设备（无需 loopback）

    后台 sounddevice 回调持续录制到内部列表，与主循环完全解耦。
    录制结束后调用 get_audio() 取回全部样本，按帧数重采样对齐。

    这避免了 read() 零填充导致的音频割裂（回调时钟与主循环 sleep
    不精确同步时，缓冲区不够就填零 → 输出 audible 的咔嚓声）。
    """

    # 输入设备名匹配规则（用作系统音频录制，不需要 loopback 标志）
    _LOOPBACK_INPUT_PATTERNS = [
        'cable output',      # VB-Cable
        'vb-audio',          # VB-Cable 变体
        'stereo mix',        # Windows 立体声混音
        'wave out mix',      # 波形输出混音
        'what u hear',       # Sound Blaster
        'loopback',          # 通用
    ]

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self._chunks: list[np.ndarray] = []  # 每个元素是 [chunk_len] int16 mono
        self._lock = threading.Lock()
        self._stream = None
        self._dev_name = "?"

    # ── 公共 API ──

    def start(self) -> bool:
        if not _HAS_SD:
            print("[Audio] sounddevice not installed (pip install sounddevice)")
            return False

        wasapi_idx = None
        for i, host in enumerate(_sd.query_hostapis()):
            if 'wasapi' in host['name'].lower():
                wasapi_idx = i
                break
        if wasapi_idx is None:
            print("[Audio] WASAPI host API not found")
            return False

        devices = _sd.query_devices()

        # ── 后端 1: WASAPI loopback（默认输出设备优先）──
        default_dev = _sd.query_hostapis()[wasapi_idx].get('default_output_device')
        loopback_candidates: list[int] = []

        if (default_dev is not None and default_dev >= 0
                and devices[default_dev]['hostapi'] == wasapi_idx):
            loopback_candidates.append(default_dev)

        for i, dev in enumerate(devices):
            if i in loopback_candidates:
                continue
            if dev['hostapi'] != wasapi_idx or dev['max_output_channels'] == 0:
                continue
            name = dev['name'].lower()
            if any(x in name for x in ['todesk', 'asio', 'virtual']):
                continue
            loopback_candidates.append(i)

        for dev_idx in loopback_candidates:
            dev = devices[dev_idx]
            try:
                self._stream = _sd.InputStream(
                    samplerate=self.sr, device=dev_idx, channels=2,
                    dtype='int16', latency='low',
                    extra_settings=_sd.WasapiSettings(exclusive=False),
                    callback=self._callback,
                )
                self._stream.start()
                self._dev_name = dev['name']
                tag = "(default)" if dev_idx == default_dev else ""
                print(f"[Audio] WASAPI loopback [{dev_idx}]: "
                      f"{self._dev_name} (16kHz mono) {tag}")
                return True
            except Exception:
                continue

        # ── 后端 2: 虚拟录制设备（CABLE Output / Stereo Mix 等）──
        input_candidates: list[tuple[int, str, int]] = []  # (prio, name, idx)
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] == 0:
                continue
            name_lower = dev['name'].lower()
            for j, pat in enumerate(self._LOOPBACK_INPUT_PATTERNS):
                if pat in name_lower:
                    # 低索引 = 高优先级（CABLE Output 排最前）
                    input_candidates.append((j, dev['name'], i))
                    break

        # 去重（同名设备可能出现在多个 hostapi 中，选 WASAPI 的）
        input_candidates.sort(key=lambda x: (x[0], x[2]))
        seen = set()
        unique_candidates = []
        for prio, name, idx in input_candidates:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                unique_candidates.append((prio, name, idx))

        for prio, name, idx in unique_candidates:
            dev = devices[idx]
            try:
                # 虚拟录制设备：普通 InputStream，无 loopback 标志
                self._stream = _sd.InputStream(
                    samplerate=self.sr, device=idx, channels=2,
                    dtype='int16', latency='low',
                    callback=self._callback,
                )
                self._stream.start()
                self._dev_name = name
                print(f"[Audio] Recording device [{idx}]: "
                      f"{name} (16kHz mono)")
                return True
            except Exception:
                continue

        print(f"[Audio] No audio capture device available "
              f"(loopback tried {len(loopback_candidates)}, "
              f"input tried {len(unique_candidates)})")
        print("[Audio] 要录制系统音频，请尝试以下之一：")
        print("[Audio]   1. 设置 VB-Cable Input 为默认播放设备")
        print("[Audio]      Win+R → mmsys.cpl → 播放 → CABLE Input → 设为默认")
        print("[Audio]   2. 启用立体声混音：")
        print("[Audio]      录制 → 右键 → 显示禁用的设备 → 启用立体声混音")
        print("[Audio]   3. 暂不录制音频，record.py 将继续采集画面+键鼠")
        return False

    def _callback(self, indata, frames, time_info, status):
        """立体声 → 单声道 → 追加到 chunk 列表。"""
        if indata.shape[1] >= 2:
            mono = indata[:, 0:2].mean(axis=1).astype(np.int16)
        else:
            mono = indata[:, 0].astype(np.int16)
        with self._lock:
            self._chunks.append(mono.copy())  # copy: indata 会被复用

    @property
    def total_samples(self) -> int:
        """当前已录制的样本数（不加锁的近似值，用于实时显示）。"""
        with self._lock:
            return sum(len(c) for c in self._chunks)

    def get_audio(self, expected_samples: int | None = None) -> np.ndarray:
        """返回全部录制音频，可选重采样到 expected_samples 以与帧对齐。

        重采样用线性插值，适用于时钟偏差很小的场景（±几十样本/分钟）。
        偏差过大时不做重采样，直接返回原始数据。
        """
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.int16)
            audio_flat = np.concatenate(self._chunks)

        if expected_samples is not None and len(audio_flat) != expected_samples:
            drift = len(audio_flat) - expected_samples
            if abs(drift) < expected_samples * 0.01:  # < 1% 偏差才重采样
                x_old = np.linspace(0, 1, len(audio_flat), endpoint=True)
                x_new = np.linspace(0, 1, expected_samples, endpoint=True)
                audio_flat = np.interp(x_new, x_old, audio_flat).astype(np.int16)
            # 否则直接截断或补零
            elif drift > 0:
                audio_flat = audio_flat[:expected_samples]
            else:
                pad = np.zeros(-drift, dtype=np.int16)
                audio_flat = np.concatenate([audio_flat, pad])

        return audio_flat

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


# ── 主循环 ────────────────────────────────────────────

def main() -> None:
    # 默认截主显示器
    with mss.MSS() as sct:
        monitor = sct.monitors[1]  # 1 = 主显示器

    w, h = monitor["width"], monitor["height"]
    print(f"Monitor: {w}x{h} at ({monitor['left']}, {monitor['top']})")

    # 脚本所在目录下创建 session 目录
    script_dir = Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = script_dir / f"session_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)

    events_path = session_dir / "events.jsonl"
    events_fh = open(events_path, "a", encoding="utf-8")

    # 事件队列 + 停止标志
    event_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    # 启动键鼠监听
    collector = EventCollector(event_queue, stop_event)
    collector.start()

    # ── 视频编码线程 ──
    video_path = session_dir / "video"
    video_writer = VideoWriterThread(str(video_path), fps=10.0,
                                     width=w, height=h)
    video_writer.start()

    # ── 音频采集 ──
    audio_cap = SystemAudioCapture(sample_rate=16000)
    if not audio_cap.start():
        audio_cap = None

    print(f"Session: {session_dir}")
    print(f"10Hz capture running...  Ctrl+C to stop.\n")

    frame_idx = 0
    interval = 1.0 / 10  # 100ms

    try:
        while not stop_event.is_set():
            loop_start = time.perf_counter()

            # 1. 截屏 + 光标合成 → 推入编码队列
            frame_bgr = capture_frame(monitor)
            video_writer.push(frame_bgr)

            # 2. 音频在后台线程独立录制，主循环不碰
            #    （避免 read() 零填充 → 音频割裂）

            # 3. 排空事件队列 → 写 JSONL
            while True:
                try:
                    ev = event_queue.get_nowait()
                    events_fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                except queue.Empty:
                    break

            frame_idx += 1

            # 4. 100ms 帧率控制
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        stop_event.set()
        collector.stop()

        # 先停视频编码线程（排空队列 → release writer）
        print("[Video] 正在关闭编码线程...")
        video_writer.stop()
        print(f"[Video] 共写入 {video_writer.frames_written} 帧, "
              f"丢弃 {video_writer.frames_dropped} 帧")
        if video_writer.actual_path:
            print(f"[Video] 文件: {video_writer.actual_path}")

        if audio_cap is not None:
            audio_cap.stop()

        # 排空剩余事件
        while True:
            try:
                ev = event_queue.get_nowait()
                events_fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
            except queue.Empty:
                break

        events_fh.flush()
        os.fsync(events_fh.fileno())
        events_fh.close()

        # ── 保存音频（从后台录制线程一次性取回，重采样对齐帧数）──
        if audio_cap is not None:
            expected = frame_idx * 1600
            audio_flat = audio_cap.get_audio(expected_samples=expected)
            audio_path = session_dir / "audio_raw.npy"
            audio_arr = audio_flat[:expected].reshape(-1, 1600)
            np.save(str(audio_path), audio_arr)
            print(f"Audio: {audio_arr.shape} int16 "
                  f"(drift={len(audio_flat) - expected:+d} samples) → {audio_path}")

        print(f"Frames captured: {frame_idx}")
        print(f"Output: {session_dir}")


if __name__ == "__main__":
    main()
