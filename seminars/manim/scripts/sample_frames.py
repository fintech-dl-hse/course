#!/usr/bin/env python3
"""Extract two frames from a rendered video — one at 50% duration, one near the end.

Rationale: the downstream ``manim-frame-critic`` vision pass only needs a
"mid-animation" sample and the final settled state. Two frames are enough.

Usage:
    sample_frames.py <video> [--out-dir DIR] [--end-offset SEC]

Выход:
    <out-dir>/frame_0001.png  — кадр на 50% длительности видео.
    <out-dir>/frame_0002.png  — кадр у самого конца (``duration - end-offset``).
    <out-dir>/frames_index.json — отчёт: длительность и таймкоды.

По умолчанию out-dir = .out/frames/<video_stem>/ относительно seminars/manim/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _probe_duration(video: Path) -> float:
    """Return the video duration in seconds via ``ffprobe``."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"ffprobe returned empty duration for {video}")
    try:
        return float(out)
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned non-numeric duration {out!r}") from exc


def _extract_frame(
    video: Path, t_seconds: float, out_path: Path, downscale: int = 2
) -> None:
    """Grab a single frame at ``t_seconds`` via ffmpeg fast-seek.

    ``downscale`` shrinks both dimensions by the given integer factor so the
    saved frames are smaller (default 2x — halves width and height, ~4x
    smaller file). The critic vision pass does not need full render resolution.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{t_seconds:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
    ]
    if downscale > 1:
        cmd += ["-vf", f"scale=iw/{downscale}:ih/{downscale}"]
    cmd += ["-update", "1", "-y", str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed at t={t_seconds}: {proc.stderr.strip()}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract two frames (midpoint + near-end) from a video."
    )
    parser.add_argument("video", type=Path, help="Input video file (.mp4)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: .out/frames/<video_stem>/ relative to seminars/manim/",
    )
    parser.add_argument(
        "--end-offset",
        type=float,
        default=0.05,
        # ffmpeg cannot seek past the final decoded frame; subtract a small
        # delta so `-ss` lands on (or just before) the last frame.
        help="Seconds subtracted from duration for the endpoint frame (default: 0.05).",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=2,
        help="Integer factor to shrink both width and height (default: 2). Use 1 to disable.",
    )
    args = parser.parse_args(argv)
    if args.downscale < 1:
        parser.error("--downscale must be >= 1")

    video: Path = args.video
    if not video.exists():
        print(f"[sample_frames] ERROR: video not found: {video}", file=sys.stderr)
        return 2

    base = Path(__file__).resolve().parent.parent  # seminars/manim/
    if args.out_dir is not None:
        out_dir: Path = args.out_dir
    else:
        out_dir = base / ".out" / "frames" / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wipe any stale frames from a previous run.
    for existing in out_dir.glob("frame_*.png"):
        existing.unlink()

    duration = _probe_duration(video.resolve())
    if duration <= 0:
        print(
            f"[sample_frames] ERROR: unusable duration {duration} for {video}",
            file=sys.stderr,
        )
        return 3

    midpoint = duration / 2.0
    endpoint = max(0.0, duration - args.end_offset)

    frame1 = out_dir / "frame_0001.png"
    frame2 = out_dir / "frame_0002.png"
    _extract_frame(video.resolve(), midpoint, frame1, downscale=args.downscale)
    _extract_frame(video.resolve(), endpoint, frame2, downscale=args.downscale)

    index = {
        "source": "midpoint_endpoint",
        "duration_seconds": duration,
        "end_offset_seconds": args.end_offset,
        "downscale": args.downscale,
        "frames": [
            {"index": 1, "t_seconds": midpoint, "role": "midpoint", "path": frame1.name},
            {"index": 2, "t_seconds": endpoint, "role": "endpoint", "path": frame2.name},
        ],
    }
    (out_dir / "frames_index.json").write_text(json.dumps(index, indent=2))
    print(
        f"[sample_frames] duration={duration:.2f}s -> 2 frames "
        f"(midpoint={midpoint:.2f}s, endpoint={endpoint:.2f}s, downscale={args.downscale}x); out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
