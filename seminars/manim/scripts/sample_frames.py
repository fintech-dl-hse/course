#!/usr/bin/env python3
"""Extract 1-fps frames from a video, dedupe via imagehash.phash.

Usage:
    sample_frames.py <video> [--out-dir DIR] [--phash-threshold N]

Выход:
    <out-dir>/frame_0001.png, ...  — сохранённые кадры после dedupe
    <out-dir>/frames_index.json    — отчёт: pre/post counts + дропнутые кадры

По умолчанию out-dir = .out/frames/<video_stem>/ относительно seminars/manim/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _run_ffmpeg_extract(video: Path, tmp_dir: Path) -> None:
    """Извлечь кадры 1 fps через ffmpeg CLI (без зависимости от ffmpeg-python runtime)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        "fps=1",
        str(tmp_dir / "frame_%04d.png"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def _dedupe(frames: list[Path], threshold: int) -> tuple[list[Path], list[dict]]:
    """pHash-based dedupe, preserving the first and last frame.

    Returns:
        kept: сохранённые Path в исходном порядке.
        dropped: список отчётных записей {"frame", "hamming_to_retained"}.
    """
    import imagehash
    from PIL import Image

    if not frames:
        return [], []
    if len(frames) == 1:
        return list(frames), []

    retained: list[Path] = [frames[0]]
    retained_hashes = [imagehash.phash(Image.open(frames[0]))]
    dropped: list[dict] = []

    # Walk middle frames; force-keep the last one separately.
    middle = frames[1:-1]
    for f in middle:
        h = imagehash.phash(Image.open(f))
        dist = int(h - retained_hashes[-1])
        if dist <= threshold:
            dropped.append({"frame": f.name, "hamming_to_retained": dist})
        else:
            retained.append(f)
            retained_hashes.append(h)

    # Always keep the last frame, even if it collapses against the previous retained one.
    last = frames[-1]
    if retained[-1] != last:
        retained.append(last)

    return retained, dropped


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract and dedupe 1-fps video frames.")
    parser.add_argument("video", type=Path, help="Input video file (.mp4)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: .out/frames/<video_stem>/ relative to seminars/manim/",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=3,
        help="Hamming-distance threshold; drop frame if <= threshold to last retained (default: 3).",
    )
    args = parser.parse_args(argv)

    video: Path = args.video
    if not video.exists():
        print(f"[sample_frames] ERROR: video not found: {video}", file=sys.stderr)
        return 2

    # Resolve out-dir relative to seminars/manim/ (parent of scripts/).
    if args.out_dir is not None:
        out_dir: Path = args.out_dir
    else:
        base = Path(__file__).resolve().parent.parent  # seminars/manim/
        out_dir = base / ".out" / "frames" / video.stem

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_raw"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    _run_ffmpeg_extract(video.resolve(), tmp_dir)

    raw_frames = sorted(tmp_dir.glob("frame_*.png"))
    pre_count = len(raw_frames)

    kept, dropped = _dedupe(raw_frames, args.phash_threshold)

    # Move retained frames into out_dir with canonical indices.
    for existing in out_dir.glob("frame_*.png"):
        existing.unlink()
    for idx, src in enumerate(kept, start=1):
        dst = out_dir / f"frame_{idx:04d}.png"
        shutil.copyfile(src, dst)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    index = {
        "pre_dedupe_count": pre_count,
        "post_dedupe_count": len(kept),
        "phash_threshold": args.phash_threshold,
        "dropped": dropped,
    }
    (out_dir / "frames_index.json").write_text(json.dumps(index, indent=2))
    print(
        f"[sample_frames] {pre_count} -> {len(kept)} frames "
        f"(dropped {len(dropped)}); out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
