#!/usr/bin/env python3
"""Stream a video file through sha256 and print the 64-char hex digest.

Usage:
    hash_video.py <video>
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Optional

_CHUNK = 1 << 20  # 1 MiB


def hash_video(path: Path) -> str:
    """Compute sha256 of file content; return 64-char lowercase hex."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stream-hash a video file with sha256.")
    parser.add_argument("video", type=Path, help="Input video file")
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"[hash_video] ERROR: not found: {args.video}", file=sys.stderr)
        return 2

    print(hash_video(args.video))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
