"""验证所有库兼容性 — 用合成数据复刻 preprocess.py 完整管线。

用法: python verify_env.py [--device cuda|cpu]

必须通过所有环节才能跑 preprocess.py。
"""
import sys
import tempfile
import os
from pathlib import Path

import numpy as np

failed = []


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  [OK] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed.append(name)


# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1: 库导入")
print("=" * 60)

check("torch", lambda: exec("import torch; global torch; torch=torch; print(f'      {torch.__version__}')"))
check("torchvision", lambda: exec("import torchvision; global torchvision; torchvision=torchvision; print(f'      {torchvision.__version__}')"))
check("torchaudio", lambda: exec("import torchaudio; global torchaudio; torchaudio=torchaudio; print(f'      {torchaudio.__version__}')"))
check("transformers", lambda: exec("import transformers; global transformers; transformers=transformers; print(f'      {transformers.__version__}')"))
check("numpy", lambda: exec("import numpy; print(f'      {numpy.__version__}')"))
check("cv2", lambda: exec("import cv2; global cv2; cv2=cv2; print(f'      {cv2.__version__}')"))
check("PIL", lambda: exec("from PIL import Image; print('      OK')"))
check("sounddevice", lambda: exec("import sounddevice; print(f'      {sounddevice.__version__}')"))
check("safetensors", lambda: exec("import safetensors; print('      OK')"))
check("packaging", lambda: exec("from packaging import version; print('      OK')"))

if failed:
    print(f"\n>>> {len(failed)} 个库导入失败: {failed}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 2: CUDA 可用性")
print("=" * 60)

cuda_ok = torch.cuda.is_available()
print(f"  torch.cuda.is_available(): {cuda_ok}")
if cuda_ok:
    print(f"  device count: {torch.cuda.device_count()}")
    print(f"  device name:  {torch.cuda.get_device_name(0)}")
    print(f"  mem:          {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  >>> CUDA 不可用！preprocess.py --device cuda 会失败")
    print("  >>> 请安装: pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124")

device = "cuda" if cuda_ok else "cpu"

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Step 3: torchvision/torch 算子兼容 (device={device})")
print("=" * 60)

check("torchvision ops", lambda: torchvision.ops.nms)

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Step 4: CLAP 加载 → {device}")
print("=" * 60)

# workaround: transformers is_torch_greater_or_equal("2.6") 对 torch 2.5 出错
import transformers.utils.import_utils as _tf_iu
_orig_check = _tf_iu.is_torch_greater_or_equal


def _patched(version, accept_dev=False):
    from packaging import version as _ver
    _tv = _ver.parse(torch.__version__.split("+")[0].replace("dev", ""))
    _rv = _ver.parse(version.replace("dev", ""))
    return _tv >= _rv


_tf_iu.is_torch_greater_or_equal = _patched

clap_model = None
clap_processor = None


def _load_clap():
    global clap_model, clap_processor
    from transformers import ClapModel, ClapProcessor
    from preprocess import CLAP_SNAPSHOT
    snap = str(CLAP_SNAPSHOT)
    clap_model = ClapModel.from_pretrained(snap).to(device).eval()
    clap_processor = ClapProcessor.from_pretrained(snap)
    print(f"      dim={clap_model.config.projection_dim}")


check("CLAP", _load_clap)

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Step 5: SigLIP + DINOv2 加载 → {device}")
print("=" * 60)

siglip = dinov2 = siglip_proc = dinov2_proc = None


def _load_vision():
    global siglip, dinov2, siglip_proc, dinov2_proc
    from transformers import SiglipVisionModel, Dinov2Model, AutoImageProcessor
    from preprocess import SIGLIP_PATH, DINOV2_PATH
    siglip = SiglipVisionModel.from_pretrained(str(SIGLIP_PATH)).to(device).eval()
    siglip_proc = AutoImageProcessor.from_pretrained(str(SIGLIP_PATH))
    print(f"      SigLIP hidden={siglip.config.hidden_size}")

    dinov2_new = Dinov2Model.from_pretrained(str(DINOV2_PATH)).to(device).eval()
    dinov2 = dinov2_new
    dinov2_proc = AutoImageProcessor.from_pretrained(str(DINOV2_PATH))
    print(f"      DINOv2 hidden={dinov2.config.hidden_size}")


check("SigLIP + DINOv2", _load_vision)

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Step 6: 合成数据跑完整管线 (device={device})")
print("=" * 60)

from preprocess import (
    encode_audio_clap, encode_visual, compute_actions,
    compute_gaze_pseudo_labels, compute_residual_targets,
    AUDIO_SR, CLAP_SR,
)
import torchaudio.transforms as T

# ── 辅助函数（必须在 check 调用前定义）──────────────────────

_fake_video = None


def _make_fake_video(tmp):
    global _fake_video
    _fake_video = tmp / "video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(_fake_video), fourcc, 10.0, (640, 480))
    for _ in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    print(f"      10 frames @ 640x480 → {_fake_video.name}")


def _enc_siglip(tmp):
    from transformers import SiglipVisionModel, AutoImageProcessor
    from preprocess import SIGLIP_PATH
    s = SiglipVisionModel.from_pretrained(str(SIGLIP_PATH)).to(device).eval()
    sp = AutoImageProcessor.from_pretrained(str(SIGLIP_PATH))
    cap = cv2.VideoCapture(str(_fake_video))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    from PIL import Image
    images = [Image.fromarray(f) for f in frames]
    si = sp(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        s_out = s(**si)
    s_vecs = s_out.last_hidden_state.cpu().numpy().astype(np.float16)
    np.save(str(tmp / "visual_siglip.npy"), s_vecs)
    print(f"      shape={s_vecs.shape}  dtype=float16")
    del s, sp


def _enc_dinov2(tmp):
    from transformers import Dinov2Model, AutoImageProcessor
    from preprocess import DINOV2_PATH
    d = Dinov2Model.from_pretrained(str(DINOV2_PATH)).to(device).eval()
    dp = AutoImageProcessor.from_pretrained(str(DINOV2_PATH))
    cap = cv2.VideoCapture(str(_fake_video))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    from PIL import Image
    images = [Image.fromarray(f) for f in frames]
    di = dp(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        d_out = d(**di)
    d_vecs = d_out.last_hidden_state.cpu().numpy().astype(np.float16)
    np.save(str(tmp / "visual_dinov2.npy"), d_vecs)
    print(f"      shape={d_vecs.shape}  dtype=float16")
    del d, dp


def _enc_audio():
    audio_raw = np.random.randint(-1000, 1000, (10, 1600), dtype=np.int16)
    resampler = T.Resample(orig_freq=AUDIO_SR, new_freq=CLAP_SR).to(device)
    out = encode_audio_clap(audio_raw, clap_model, clap_processor,
                            resampler, device, batch_size=2)
    assert out.shape == (10, 512), f"bad shape: {out.shape}"
    assert out.dtype == np.float16
    assert not np.any(np.isnan(out))
    print(f"      shape={out.shape}  mean={out.mean():.4f}  no_nan=True")


def _test_actions():
    act = compute_actions([], 0, 10, 640, 480)
    assert act.shape == (10, 178)
    assert act.dtype == np.float32
    print(f"      shape={act.shape}")


def _test_gaze(tmp):
    g = compute_gaze_pseudo_labels(_fake_video, 10, 640, 480)
    assert g.shape == (10, 2)
    assert g[0, 0] == 0.5 and g[0, 1] == 0.5
    assert np.all(g >= 0) and np.all(g <= 1)
    print(f"      shape={g.shape}  frame0=(0.5,0.5)  in_range=True")


def _test_residuals(tmp):
    sf_path = tmp / "visual_siglip.npy"
    shape, _ = compute_residual_targets(sf_path, tmp / "frame_targets_siglip.npy")
    tgt = np.load(str(tmp / "frame_targets_siglip.npy"), mmap_mode="r")
    assert np.all(tgt[-1] == 0), "last frame not zero"
    print(f"      shape={tgt.shape}  last_zeros=True")
    tgt._mmap.close()


# ── 执行测试 ─────────────────────────────────────────────

tmpdir = Path(tempfile.mkdtemp())
tmp = tmpdir

try:
    check("6a video gen", lambda: _make_fake_video(tmp))
    check("6a SigLIP enc", lambda: _enc_siglip(tmp))
    check("6a DINOv2 enc", lambda: _enc_dinov2(tmp))
    check("6b CLAP audio enc", lambda: _enc_audio())
    check("6c actions", lambda: _test_actions())
    check("6c gaze", lambda: _test_gaze(tmp))
    check("6d residuals", lambda: _test_residuals(tmp))
finally:
    if device == "cuda":
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    for f in list(tmp.glob("*.npy")) + list(tmp.glob("*.mp4")):
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass
    try:
        tmp.rmdir()
    except Exception:
        pass

# ── 清理 ──
if clap_model is not None:
    del clap_model, clap_processor
if siglip is not None:
    del siglip, dinov2, siglip_proc, dinov2_proc
if device == "cuda":
    torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Step 7: End-to-end synthetic pipeline (device={device})")
print("=" * 60)

from preprocess import preprocess_session


def _run_e2e():
    import tempfile, json
    tmp = Path(tempfile.mkdtemp())
    session = tmp / "session_test"
    session.mkdir()
    out_dir = tmp / "processed"

    # 合成 20 帧视频
    video_path = session / "video.mp4"
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 480))
    for _ in range(20):
        out.write(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    out.release()

    # 合成音频
    np.save(str(session / "audio_raw.npy"),
            np.random.randint(-1000, 1000, (20, 1600), dtype=np.int16))

    # 合成 events
    (session / "events.jsonl").write_text("\n".join([
        json.dumps({"type": "mouse_move", "x": 320, "y": 240, "ts_ns": 0}),
        json.dumps({"type": "mouse_move", "x": 330, "y": 250, "ts_ns": 500_000_000}),
    ]), encoding="utf-8")

    preprocess_session(session, out_dir, device=device, batch_size=2,
                       max_frames=20, legacy_audio=False)

    # 验证输出
    from preprocess import SIGLIP_PATH, DINOV2_PATH
    from transformers import SiglipVisionModel, Dinov2Model
    siglip_cfg = SiglipVisionModel.from_pretrained(str(SIGLIP_PATH)).config
    dinov2_cfg = Dinov2Model.from_pretrained(str(DINOV2_PATH)).config

    checks = {
        "visual_siglip.npy": (20, 196, 768),
        "visual_dinov2.npy": (20, 257, 768),
        "frame_targets_siglip.npy": (20, 196, 768),
        "frame_targets_dinov2.npy": (20, 257, 768),
        "audio_clap.npy": (20, 512),
        "gaze_pseudo.npy": (20, 2),
        "actions.npy": (20, 178),
        "timestamps.npy": (20,),
    }
    for fname, exp in checks.items():
        arr = np.load(str(out_dir / fname), mmap_mode="r")
        assert arr.shape == exp, f"{fname}: {arr.shape} != {exp}"

    gaze = np.load(str(out_dir / "gaze_pseudo.npy"))
    assert gaze[0, 0] == 0.5 and gaze[0, 1] == 0.5, "frame0 not center"

    tgt = np.load(str(out_dir / "frame_targets_siglip.npy"), mmap_mode="r")
    assert np.all(tgt[-1] == 0), "last frame not zero"

    # cleanup
    del siglip_cfg, dinov2_cfg
    import gc; gc.collect()
    for f in tmp.rglob("*"):
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass
    for d in sorted(tmp.rglob("*"), reverse=True):
        try:
            d.rmdir()
        except Exception:
            pass
    print("      E2E pipeline: 20 synthetic frames → all 8 files validated")


check("7  e2e synthetic", _run_e2e)

if device == "cuda":
    torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════
if failed:
    print(f"\n{'=' * 60}")
    print(f">>> {len(failed)} 个环节失败: {failed}")
    print(f"{'=' * 60}")
    sys.exit(1)
else:
    print(f"\n{'=' * 60}")
    print("全部通过 — 可以跑 preprocess.py --device cuda")
    print("=" * 60)
