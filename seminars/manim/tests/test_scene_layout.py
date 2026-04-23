"""Static layout check for every scene under ``scenes/``.

Autodiscovers scene files (one ``class <ClassName>(Scene)`` per file),
imports each with animations monkey-patched to no-ops, and asserts
``LayoutLinter`` reports no ``high`` issues. Runs in seconds — no rendering.

This test requires a working manim environment (LaTeX for ``MathTex``).
When manim isn't importable, every case is marked ``skip`` rather than
``fail`` so the test file remains usable outside the dedicated venv.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterator

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # seminars/manim
SCENES_DIR = REPO / "scenes"
SCRIPTS_DIR = REPO / "scripts"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS_DIR))


CLASS_DECL = re.compile(r"^class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(\s*Scene\s*\)\s*:", re.MULTILINE)


def _discover_scenes() -> list[tuple[str, Path]]:
    """Yield (ClassName, scene_file) for every Scene subclass under scenes/."""
    found: list[tuple[str, Path]] = []
    if not SCENES_DIR.exists():
        return found
    for py in SCENES_DIR.rglob("*.py"):
        if py.name == "__init__.py":
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in CLASS_DECL.finditer(src):
            found.append((m.group("name"), py))
    return found


SCENE_CASES = _discover_scenes()


def _cases() -> Iterator[pytest.param]:
    if not SCENE_CASES:
        yield pytest.param(None, None, id="no-scenes-found", marks=pytest.mark.skip(reason="no Scene subclasses discovered"))
        return
    for scene, path in SCENE_CASES:
        yield pytest.param(scene, path, id=scene)


@pytest.mark.parametrize("scene,path", list(_cases()))
def test_scene_layout_clean(scene: str | None, path: Path | None) -> None:
    if scene is None:
        pytest.skip("no scenes discovered")
    manim = pytest.importorskip("manim", reason="manim not available in this environment")
    # Import the lint harness lazily: it in turn imports shared.layout_check,
    # which imports manim — so skip cleanly above if manim is missing.
    from lint_scene import lint_scene  # noqa: E402

    try:
        issues = lint_scene(scene, path)
    except Exception as e:
        if "latex" in str(e).lower() or "tex" in type(e).__name__.lower():
            pytest.skip(f"LaTeX unavailable: {e!r}")
        raise

    high = [i for i in issues if i.severity == "high"]
    assert not high, (
        f"{scene} has {len(high)} high-severity layout issue(s):\n"
        + "\n".join(f"  - {i}" for i in high)
    )
