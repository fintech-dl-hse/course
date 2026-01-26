#!/usr/bin/env python3
from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path

COLAB_BADGE_URL = "https://colab.research.google.com/assets/colab-badge.svg"
COLAB_NOTEBOOK_BASE = (
    "https://colab.research.google.com/github/fintech-dl-hse/course/blob/main/seminars"
)


def build_badge_lines(notebook_rel_path: str) -> list[str]:
    notebook_url = f"{COLAB_NOTEBOOK_BASE}/{notebook_rel_path}"
    return [
        f'<a target="_blank" href="{notebook_url}">\n',
        f'  <img src="{COLAB_BADGE_URL}" alt="Open In Colab"/>\n',
        "</a>\n",
    ]


def extract_text(source: object) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(item for item in source if isinstance(item, str))
    return ""


def notebook_has_badge(notebook: dict) -> bool:
    for cell in notebook.get("cells", []):
        text = extract_text(cell.get("source"))
        if COLAB_BADGE_URL in text or COLAB_NOTEBOOK_BASE in text:
            return True
    return False


def next_unique_id(existing_ids: set[str], base: str) -> str:
    if base not in existing_ids:
        return base
    suffix = 2
    while f"{base}-{suffix}" in existing_ids:
        suffix += 1
    return f"{base}-{suffix}"


def create_badge_cell(badge_lines: list[str], existing_ids: set[str]) -> dict:
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": badge_lines,
    }
    if existing_ids:
        cell["id"] = next_unique_id(existing_ids, "colab-badge")
    return cell


def process_notebook(path: Path, seminars_dir: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            notebook = json.load(handle)
    except JSONDecodeError:
        print(f"Skipping invalid notebook: {path.relative_to(seminars_dir)}")
        return False

    if notebook_has_badge(notebook):
        return False

    rel_path = path.relative_to(seminars_dir).as_posix()
    badge_lines = build_badge_lines(rel_path)
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return False

    existing_ids = {
        cell.get("id") for cell in cells if isinstance(cell, dict) and "id" in cell
    }
    existing_ids = {value for value in existing_ids if isinstance(value, str)}
    badge_cell = create_badge_cell(badge_lines, existing_ids)

    notebook["cells"] = [badge_cell] + cells
    with path.open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    return True


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
            print(f"Added badge to {notebook.relative_to(seminars_dir)}")
    else:
        print("No notebooks needed badge updates.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

