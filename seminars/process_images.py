#!/usr/bin/env python3
"""
Process Jupyter notebook markdown cells: normalize images to HTML with max width=400.

- Remote URLs: keep as-is, wrap in <img src="..." width=400 />.
- Local paths (static/...): keep, ensure <img src="static/..." width=400 />.
- Base64 inline images: decode and save to ./static/<name>.(png|jpg|gif), then use
  <img src="static/<name>.<ext>" width=400 />.

Usage:
  python process_images.py path/to/notebook.ipynb [path/to/another.ipynb ...]
  python process_images.py --dry-run path/to/notebook.ipynb  # print changes, do not write
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path


# Max width for all images (HTML attribute).
MAX_WIDTH = 400

# Supported data URL MIME types → file extension for static files.
MIME_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "png": "png",
    "jpeg": "jpg",
    "jpg": "jpg",
    "gif": "gif",
}


def _ensure_static_dir(static_dir: Path) -> Path:
    static_dir.mkdir(parents=True, exist_ok=True)
    return static_dir


def _save_base64_image(static_dir: Path, data: bytes, mime: str, counter: int) -> str:
    """Save decoded image to static_dir; return path for src (e.g. 'static/img_0.png')."""
    ext = MIME_TO_EXT.get(mime.lower(), "png")
    name = f"img_{counter}.{ext}"
    path = static_dir / name
    path.write_bytes(data)
    return f"static/{name}"


def process_cell_source(
    source: list[str],
    static_dir: Path,
    image_counter: list[int],
    dry_run: bool = False,
) -> tuple[list[str], bool]:
    """
    Process markdown source: replace image markdown and ensure img tags have width=400.

    Returns (new_source_lines, changed).
    """
    text = "".join(source)
    changed = False

    # 1) Markdown image with data URL: ![alt](data:image/xxx;base64,...)
    # Regex with DOTALL so base64 can span newlines; .+? stops at first ")" (base64 has no ")").
    def replace_data_url(match: re.Match) -> str:
        nonlocal changed
        mime = match.group(2).strip().lower()
        b64_raw = match.group(3)
        ext = MIME_TO_EXT.get(mime)
        if not ext:
            print("Unknown mime type:", mime)
            return match.group(0)
        b64_clean = b64_raw.replace("\n", "").replace("\r", "").strip()
        idx = image_counter[0]
        image_counter[0] += 1
        if dry_run:
            changed = True
            return f'<img src="static/img_{idx}.{ext}" width={MAX_WIDTH} />'
        try:
            data = base64.b64decode(b64_clean)
        except Exception:
            print("Failed to decode base64:", b64_clean)
            return match.group(0)
        src_path = _save_base64_image(static_dir, data, mime, idx)
        changed = True
        return f'<img src="{src_path}" width={MAX_WIDTH} />'

    _data_url_re = re.compile(
        r"!\[[^\]]*\]\((data:image/([^;]+);base64,(.+?))\)",
        re.DOTALL,
    )
    new_text = _data_url_re.sub(replace_data_url, text)
    if new_text != text:
        changed = True
        text = new_text

    # 2) Markdown image with remote URL: ![alt](https?://...)
    def replace_remote_url(match: re.Match) -> str:
        nonlocal changed
        url = match.group(1)
        if url.strip().lower().startswith("data:"):
            return match.group(0)
        changed = True
        return f'<img src="{url}" width={MAX_WIDTH} />'

    text = re.sub(
        r"!\[[^\]]*\]\((https?://[^)]+)\)",
        replace_remote_url,
        text,
    )

    # 3) Markdown image with local path: ![alt](static/...) or ![alt](./static/...)
    def replace_local_md(match: re.Match) -> str:
        nonlocal changed
        path = match.group(1).strip()
        if path.startswith("./"):
            path = path[2:]
        if not path.startswith("static/"):
            path = "static/" + path.lstrip("/")
        changed = True
        return f'<img src="{path}" width={MAX_WIDTH} />'

    text = re.sub(
        r"!\[[^\]]*\]\(((?:\./)?static/[^)]+)\)",
        replace_local_md,
        text,
    )

    # 4) Existing <img ... src="..." ...>: ensure width=400 (add or replace)
    def fix_img_tag(match: re.Match) -> str:
        nonlocal changed
        full = match.group(0)
        if re.search(r"\bwidth\s*=", full, re.IGNORECASE):
            new_full = re.sub(r"\bwidth\s*=\s*[\"']?\d+[\"']?", f"width={MAX_WIDTH}", full, flags=re.IGNORECASE)
        else:
            # Insert width before > or />
            new_full = re.sub(r"(\s*)(/?>)", f" width={MAX_WIDTH}\\1\\2", full, count=1)
        if new_full != full:
            changed = True
        return new_full

    text = re.sub(
        r"<img\s[^>]*src\s*=\s*[\"'][^\"']+[\"'][^>]*/?>",
        fix_img_tag,
        text,
    )

    if not changed:
        return source, False

    # Split back into lines (preserve trailing newline per line like Jupyter)
    lines = text.split("\n")
    new_source = [line + "\n" for line in lines[:-1]]
    if lines:
        new_source.append(lines[-1] if lines[-1] else "")
    return new_source, True


def process_notebook(notebook_path: Path, dry_run: bool = False) -> bool:
    """
    Process a single notebook: extract base64 images to static/, normalize all images to width=400.

    Returns True if notebook was modified.
    """
    notebook_path = notebook_path.resolve()
    if not notebook_path.suffix == ".ipynb":
        return False
    static_dir = notebook_path.parent / "static"
    if not dry_run:
        _ensure_static_dir(static_dir)

    with open(notebook_path, encoding="utf-8") as f:
        nb = json.load(f)

    modified = False
    image_counter = [0]

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", [])
        if not src:
            continue
        new_src, cell_changed = process_cell_source(
            src, static_dir, image_counter, dry_run=dry_run
        )
        if cell_changed:
            cell["source"] = new_src
            modified = True

    if modified and not dry_run:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")

    return modified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize notebook images to HTML with width=400; extract base64 to static/."
    )
    parser.add_argument(
        "notebooks",
        nargs="+",
        type=Path,
        help="Paths to .ipynb files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be changed, do not write files or create static/",
    )
    args = parser.parse_args()

    for path in args.notebooks:
        if not path.exists():
            print(f"Skip (not found): {path}", file=sys.stderr)
            continue
        try:
            changed = process_notebook(path, dry_run=args.dry_run)
            status = "would be modified" if args.dry_run and changed else ("modified" if changed else "unchanged")
            print(f"{path}: {status}")
        except Exception as e:
            print(f"{path}: error — {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
