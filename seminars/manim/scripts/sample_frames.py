#!/usr/bin/env python3
"""Extract frames from a video, using Manim keyframe sidecar when available.

Primary path: if a keyframe sidecar exists at .out/keyframes/<video_stem>.json
and contains events with t_seconds > 0, extract one frame per keyframe event
via ffmpeg fast-seek (-ss before -i).

Fallback path: 1-fps extraction + imagehash pHash dedupe (original behaviour).

Usage:
    sample_frames.py <video> [--out-dir DIR] [--phash-threshold N] [--force-phash]

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


def _extract_by_keyframes(
    video: Path,
    sidecar: Path,
    out_dir: Path,
    min_dt: float = 0.0,
) -> tuple[list[Path], list[dict]]:
    """Extract frames at keyframe timestamps using ffmpeg fast-seek.

    Args:
        video: Path to the input video file.
        sidecar: Path to the keyframe sidecar JSON file.
        out_dir: Directory where extracted frames will be written.
        min_dt: Minimum spacing between consecutive kept timestamps, in seconds.
            Events closer than this to the previous kept timestamp are dropped
            as mid-animation transitions. Set to 0 to disable.

    Returns:
        kept_paths: List of extracted frame Paths in order.
        keyframe_records: List of dicts with index, t_seconds, kind, animation.
    """
    data = json.loads(sidecar.read_text())
    raw_events: list[dict] = data.get("events", [])

    # Keep only events with t_seconds > 0; dedupe consecutive identical timestamps
    # by keeping the first occurrence of each distinct timestamp. When min_dt>0,
    # also collapse clusters spaced closer than min_dt (keep the earliest of each
    # cluster; animation beats settle at the cluster boundary either way).
    last_kept: float = -1.0
    deduped: list[dict] = []
    for ev in raw_events:
        t = float(ev["t_seconds"])
        if t <= 0:
            continue
        if last_kept >= 0 and (t - last_kept) < max(min_dt, 1e-9):
            continue
        last_kept = t
        deduped.append(ev)

    out_dir.mkdir(parents=True, exist_ok=True)
    kept_paths: list[Path] = []
    keyframe_records: list[dict] = []

    for idx, ev in enumerate(deduped, start=1):
        t = float(ev["t_seconds"])
        out_path = out_dir / f"frame_{idx:04d}.png"
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-ss",
            str(t),
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-update",
            "1",
            "-y",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed at t={t}: {proc.stderr.strip()}"
            )
        kept_paths.append(out_path)
        keyframe_records.append(
            {
                "index": idx,
                "t_seconds": t,
                "kind": ev.get("kind", ""),
                "animation": ev.get("animation", ""),
            }
        )

    return kept_paths, keyframe_records


def _load_sidecar_if_usable(video: Path, out_dir_base: Path) -> Optional[Path]:
    """Return the sidecar path if it exists and has at least one event with t_seconds > 0.

    Args:
        video: Input video path (stem used to find sidecar).
        out_dir_base: seminars/manim/ base directory.

    Returns:
        Path to the sidecar JSON, or None if not usable.
    """
    sidecar = out_dir_base / ".out" / "keyframes" / f"{video.stem}.json"
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text())
        events = data.get("events", [])
        if any(float(ev["t_seconds"]) > 0 for ev in events):
            return sidecar
    except Exception:
        return None
    return None


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
    parser.add_argument(
        "--force-phash",
        action="store_true",
        default=False,
        help="Disable the hook/keyframe path even when a sidecar exists (for benchmarking).",
    )
    parser.add_argument(
        "--hook-min-dt",
        type=float,
        default=0.75,
        help=(
            "Minimum spacing (seconds) between consecutive kept keyframes in hook "
            "mode; tighter clusters collapse to the first timestamp. Default: 0.75."
        ),
    )
    args = parser.parse_args(argv)

    video: Path = args.video
    if not video.exists():
        print(f"[sample_frames] ERROR: video not found: {video}", file=sys.stderr)
        return 2

    # Resolve out-dir relative to seminars/manim/ (parent of scripts/).
    base = Path(__file__).resolve().parent.parent  # seminars/manim/
    if args.out_dir is not None:
        out_dir: Path = args.out_dir
    else:
        out_dir = base / ".out" / "frames" / video.stem

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Primary path: keyframe sidecar ---
    sidecar: Optional[Path] = None
    if not args.force_phash:
        sidecar = _load_sidecar_if_usable(video, base)

    if sidecar is not None:
        # Clean up any existing frames in out_dir before writing new ones.
        for existing in out_dir.glob("frame_*.png"):
            existing.unlink()

        data = json.loads(sidecar.read_text())
        scene_name: str = data.get("scene", video.stem)
        raw_event_count = len(data.get("events", []))

        kept_paths, keyframe_records = _extract_by_keyframes(
            video.resolve(), sidecar, out_dir, min_dt=args.hook_min_dt
        )

        index = {
            "source": "hook",
            "scene": scene_name,
            "pre_hook_event_count": raw_event_count,
            "post_dedupe_event_count": len(kept_paths),
            "keyframes": keyframe_records,
        }
        (out_dir / "frames_index.json").write_text(json.dumps(index, indent=2))
        print(
            f"[sample_frames] hook: {len(kept_paths)} keyframes extracted "
            f"(from {raw_event_count} events); out={out_dir}"
        )
        return 0

    # --- Fallback path: ffmpeg 1-fps + pHash dedupe ---
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
        "source": "phash_fallback",
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
