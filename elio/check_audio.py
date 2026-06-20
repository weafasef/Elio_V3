"""检查 session 的音频内容。用法: python -m elio.check_audio [session_dir]"""
import numpy as np
import sys
from pathlib import Path


def check_audio(session_dir: Path):
    audio = np.load(str(session_dir / "audio_raw.npy"))

    print(f"session: {session_dir.name}")
    print(f"shape={audio.shape}  dtype={audio.dtype}")
    print(f"min={audio.min()}  max={audio.max()}  mean={audio.mean():.2f}  std={audio.std():.2f}")
    print(f"zeros={(audio == 0).sum()}/{audio.size} ({(audio == 0).sum()/audio.size*100:.1f}%)")

    abs_max = np.abs(audio).max(axis=1)
    print(f"silent chunks (peak<10): {(abs_max < 10).sum()} / {audio.shape[0]}")
    print(f"loud chunks  (peak>100): {(abs_max > 100).sum()} / {audio.shape[0]}")
    print(f"first 5 rows peak: {abs_max[:5]}")
    print(f"last 5 rows peak:  {abs_max[-5:]}")
    print(f"peak distribution: min={abs_max.min()} p50={np.median(abs_max):.0f} "
          f"p90={np.percentile(abs_max, 90):.0f}")

    npy_mb = (session_dir / "audio_raw.npy").stat().st_size / 1024**2
    video_gb = (session_dir / "video.mp4").stat().st_size / 1024**3
    duration_sec = audio.shape[0] * 0.1
    print(f"file sizes: audio={npy_mb:.0f}MB  video={video_gb:.1f}GB  duration={duration_sec:.0f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check raw audio in a session directory")
    parser.add_argument("session", nargs="?", default=None,
                        help="Path to session directory (auto-find latest in data/raw/ if omitted)")
    args = parser.parse_args()

    if args.session:
        session = Path(args.session)
    else:
        project = Path(__file__).resolve().parent.parent
        data_raw = project / "data" / "raw"
        sessions = sorted(data_raw.glob("session_*"))
        if not sessions:
            print("No session directories found in data/raw/", file=sys.stderr)
            sys.exit(1)
        session = sessions[-1]
        print(f"Auto-selected: {session.name}")

    check_audio(session)
