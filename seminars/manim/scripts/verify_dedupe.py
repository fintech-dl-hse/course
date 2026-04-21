#!/usr/bin/env python3
"""Assert that two frames are NOT collapsed by pHash at a given threshold.

Usage:
    verify_dedupe.py <frame_a> <frame_b> --phash-threshold N

Exit 0 if Hamming distance > threshold (would be kept distinct).
Exit 1 if Hamming distance <= threshold (dedupe would collapse them).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify pHash dedupe preserves two frames.")
    parser.add_argument("frame_a", type=Path)
    parser.add_argument("frame_b", type=Path)
    parser.add_argument("--phash-threshold", type=int, required=True)
    args = parser.parse_args(argv)

    if not args.frame_a.exists() or not args.frame_b.exists():
        print(
            f"[verify_dedupe] ERROR: missing frame(s): {args.frame_a} / {args.frame_b}",
            file=sys.stderr,
        )
        return 2

    import imagehash
    from PIL import Image

    ha = imagehash.phash(Image.open(args.frame_a))
    hb = imagehash.phash(Image.open(args.frame_b))
    dist = int(ha - hb)
    print(f"hamming_distance={dist} threshold={args.phash_threshold}")
    if dist > args.phash_threshold:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
