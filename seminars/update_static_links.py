#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

BASE_REPO_URL = "https://github.com/fintech-dl-hse/course/raw/refs/heads/main/seminars"
SRC_RE = re.compile(r'(\bsrc\s*=\s*["\'])static/([^"\']+)')
MD_RE = re.compile(r'(\]\()static/([^)]+)\)')


def update_line(line: str, base_url: str) -> tuple[str, bool]:
    changed = False

    def replace_src(match: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        return f"{match.group(1)}{base_url}/{match.group(2)}"

    def replace_md(match: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        return f"{match.group(1)}{base_url}/{match.group(2)})"

    updated = SRC_RE.sub(replace_src, line)
    updated = MD_RE.sub(replace_md, updated)
    return updated, changed


def process_notebook(path: Path, seminars_dir: Path) -> bool:
    rel_dir = path.parent.relative_to(seminars_dir).as_posix()
    base_url = f"{BASE_REPO_URL}/{rel_dir}/static"

    with path.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    updated = False
    for cell in notebook.get("cells", []):
        source = cell.get("source")
        if not isinstance(source, list):
            continue
        new_source: list[str] = []
        for line in source:
            if not isinstance(line, str):
                new_source.append(line)
                continue
            updated_line, changed = update_line(line, base_url)
            updated = updated or changed
            new_source.append(updated_line)
        cell["source"] = new_source

    if updated:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(notebook, handle, ensure_ascii=False, indent=1)
            handle.write("\n")

    return updated


def main() -> int:
    seminars_dir = Path(__file__).resolve().parent
    updated_files = []
    for notebook in seminars_dir.rglob("*.ipynb"):
        if "ipynb_checkpoints" in notebook.parts:
            continue
        if process_notebook(notebook, seminars_dir):
            updated_files.append(notebook)

    if updated_files:
        for notebook in updated_files:
            print(f"Updated {notebook.relative_to(seminars_dir)}")
    else:
        print("No notebooks needed updates.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

