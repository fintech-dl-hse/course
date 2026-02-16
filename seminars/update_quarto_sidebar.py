#!/usr/bin/env python3
"""
Update seminars/_quarto.yml sidebar with the current list of seminar notebooks.

Discovers direct subdirs of seminars/ that contain a main .ipynb (same stem as
the dir name, or the first .ipynb in the dir), sorts them by dir name, and
rewrites the "Семинары" section contents in _quarto.yml.

Run from repo root or from seminars/; used by the auto_update_seminars workflow.
"""
from __future__ import annotations

import re
from pathlib import Path


def discover_seminar_notebooks(seminars_dir: Path) -> list[str]:
    """
    Return sorted list of relative paths to main seminar notebooks.

    Each direct subdir of seminars_dir that looks like a seminar folder
    (contains at least one .ipynb) contributes one path. Prefer the .ipynb
    whose stem matches the dir name; else one with same numeric prefix and
    "seminar" in the name; else the first .ipynb by name.
    """
    entries: list[tuple[str, str]] = []
    for path in sorted(seminars_dir.iterdir()):
        if not path.is_dir() or path.name.startswith(("_", ".")):
            continue
        ipynbs = sorted(path.glob("*.ipynb"))
        if not ipynbs:
            continue
        # Prefer notebook with same stem as dir name
        preferred = path / f"{path.name}.ipynb"
        if preferred.exists():
            nb = preferred
        else:
            # Prefer main seminar notebook: same numeric prefix as dir + "seminar" in name
            prefix = path.name.split("_")[0] + "_" if path.name else ""
            seminar_nbs = [p for p in ipynbs if p.stem.startswith(prefix) and "seminar" in p.stem.lower()]
            nb = seminar_nbs[0] if seminar_nbs else ipynbs[0]
        rel = nb.relative_to(seminars_dir).as_posix()
        entries.append((path.name, rel))
    return [rel for _, rel in sorted(entries, key=lambda x: x[0])]


def update_quarto_yml(seminars_dir: Path, notebook_paths: list[str]) -> bool:
    """
    Rewrite _quarto.yml so the sidebar "Семинары" contents match notebook_paths.

    Returns True if the file was changed.
    """
    config_path = seminars_dir / "_quarto.yml"
    text = config_path.read_text(encoding="utf-8")

    # Replace the list under "Семинары" / contents: (only the - dir/file.ipynb lines)
    # Pattern: section "Семинары" then contents: then indented - path lines until next key
    new_list = "\n".join(f"          - {p}" for p in notebook_paths)
    # Match from "section: Семинары" through the last "          - ...ipynb"
    pattern = re.compile(
        r"(- section: [\"']Семинары[\"']\s*\n\s+contents:\s*\n)(?:\s+-\s+[\w/]+\.ipynb\s*\n)+",
        re.MULTILINE,
    )
    new_text = pattern.sub(r"\g<1>" + new_list + "\n", text)
    if new_text == text:
        return False
    config_path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    seminars_dir = Path(__file__).resolve().parent
    notebook_paths = discover_seminar_notebooks(seminars_dir)
    if not notebook_paths:
        print("No seminar notebooks found.")
        return 0
    print("Seminars:", ", ".join(notebook_paths))
    if update_quarto_yml(seminars_dir, notebook_paths):
        print("Updated _quarto.yml sidebar.")
    else:
        print("_quarto.yml sidebar already matches discovered notebooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
