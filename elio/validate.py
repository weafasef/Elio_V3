"""验证预处理输出数据：形状、NaN、gaze 范围、残差末帧、时序单调性等。"""
import numpy as np, json, sys
from pathlib import Path


def find_latest_processed(data_dir: Path) -> Path | None:
    """自动找 data/processed/ 下最新的 processed 目录。"""
    candidates = sorted(data_dir.glob("*/processed"), key=lambda p: p.parent.name)
    return candidates[-1] if candidates else None


def validate_processed(processed_dir: Path) -> bool:
    """返回 True 表示全部通过。"""
    if not processed_dir.exists():
        print(f"ERROR: directory not found: {processed_dir}", file=sys.stderr)
        return False

    spec = json.loads((processed_dir / "spec.json").read_text())
    N = spec["N"]
    print(f"N={N}  session={spec['session_id']}  "
          f"screen={spec['screen']['width']}x{spec['screen']['height']}")

    # ── 预期形状 ──
    expected_shapes = {
        "visual_siglip.npy": (N, 196, 768),
        "visual_dinov2.npy": (N, 257, 768),
        "frame_targets_siglip.npy": (N, 196, 768),
        "frame_targets_dinov2.npy": (N, 257, 768),
        "audio_clap.npy": (N, 512),
        "gaze_pseudo.npy": (N, 2),
        "actions.npy": (N, 178),
        "timestamps.npy": (N,),
    }

    all_ok = True

    for fname, exp_shape in expected_shapes.items():
        fp = processed_dir / fname
        if not fp.exists():
            print(f"  MISSING: {fname}")
            all_ok = False
            continue

        arr = np.load(str(fp), mmap_mode="r")
        shape_ok = arr.shape == exp_shape
        nan_head = np.any(np.isnan(arr[: min(100, arr.shape[0])]))
        print(f"  {'OK' if shape_ok else 'FAIL'} {fname:30s} "
              f"{str(arr.shape):25s} NaN_head={nan_head}")
        arr._mmap.close()
        if not shape_ok:
            all_ok = False

    # ── Gaze ──
    gaze = np.load(str(processed_dir / "gaze_pseudo.npy"))
    in_range = bool(np.all(gaze >= 0) and np.all(gaze <= 1))
    center = gaze[0, 0] == 0.5 and gaze[0, 1] == 0.5
    print(f"  gaze x∈[{gaze[:, 0].min():.4f}, {gaze[:, 0].max():.4f}]  "
          f"y∈[{gaze[:, 1].min():.4f}, {gaze[:, 1].max():.4f}]")
    print(f"  gaze[0]=({gaze[0, 0]:.4f}, {gaze[0, 1]:.4f})  "
          f"in_range={in_range}  center={center}")
    if not in_range or not center:
        all_ok = False

    # ── Residual last frame ──
    for name in ["frame_targets_siglip", "frame_targets_dinov2"]:
        tgt = np.load(str(processed_dir / f"{name}.npy"), mmap_mode="r")
        last_zero = bool(np.all(tgt[-1] == 0))
        nz_ratio = float((tgt[:100] != 0).sum() / tgt[:100].size)
        print(f"  {name}: last_zero={last_zero}  head_nz_ratio={nz_ratio:.4f}")
        tgt._mmap.close()
        if not last_zero:
            all_ok = False

    # ── Audio ──
    audio = np.load(str(processed_dir / "audio_clap.npy"), mmap_mode="r")
    print(f"  audio mean={audio[:].mean():.6f}  std={audio[:].std():.6f}")
    if audio.shape[1] != 512:
        print(f"  FAIL: CLAP dim={audio.shape[1]}, expected 512")
        all_ok = False
    audio._mmap.close()

    # ── Timestamps ──
    ts = np.load(str(processed_dir / "timestamps.npy"))
    monotonic = bool(np.all(np.diff(ts) > 0))
    print(f"  timestamps monotonic={monotonic}")
    if not monotonic:
        all_ok = False

    # ── Actions ──
    actions = np.load(str(processed_dir / "actions.npy"), mmap_mode="r")
    print(f"  actions nonzero={int((actions[:] != 0).sum()):,} of {int(actions.size):,}")
    actions._mmap.close()

    if all_ok:
        print("\nALL CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED")
    return all_ok


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Validate preprocessed data")
    p.add_argument("dir", nargs="?", default=None,
                   help="Path to processed/ directory (auto-find latest if omitted)")
    args = p.parse_args()

    if args.dir:
        target = Path(args.dir)
    else:
        # 自动搜项目根下的 data/processed/
        project = Path(__file__).resolve().parent.parent
        target = find_latest_processed(project / "data" / "processed")
        if target is None:
            print("No processed data found. Specify a directory.", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-selected: {target}")

    ok = validate_processed(target)
    sys.exit(0 if ok else 1)
