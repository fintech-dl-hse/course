"""Run the static layout linter against a ManimCE scene without rendering video.

Usage:
    python scripts/lint_scene.py --scene LSTMGates
    python scripts/lint_scene.py scenes/lstm/LSTMGates.py

The script imports the scene module, monkey-patches `Scene.play`/`wait`
so construct() populates the mobject tree without animating, then runs
`shared.layout_check.run_all` **at every keyframe** (after each `play()`
and `wait()` call) and deduplicates issues. Exit code:

* 0 — no issues
* 1 — one or more `high` issues
* 2 — render-time error (e.g. LaTeX failure)
* 3 — usage/discovery error

The expected caller is the manim-visualizer agent's inner loop: run lint
first, only spend a full `make render` once lint is clean.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # seminars/manim
sys.path.insert(0, str(REPO))

from shared.layout_check import Issue, format_report, run_all  # noqa: E402


def _find_scene_file(scene: str) -> Path | None:
    """Mirror the Makefile's render-target lookup: filename first, then class grep."""
    scenes_dir = REPO / "scenes"
    by_name = list(scenes_dir.rglob(f"{scene}.py"))
    if by_name:
        return by_name[0]
    pattern = re.compile(rf"^class\s+{re.escape(scene)}\s*\(", re.MULTILINE)
    for py in scenes_dir.rglob("*.py"):
        try:
            if pattern.search(py.read_text(encoding="utf-8")):
                return py
        except OSError:
            continue
    return None


def _import_scene_module(path: Path) -> Any:
    """Import the scene .py as a standalone module."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _LintCollector:
    """Collects lint issues at every keyframe during construct()."""

    def __init__(self) -> None:
        self.step: int = 0
        self.all_issues: list[tuple[int, Issue]] = []
        # Dedup key: (code, tuple of mobject names)
        self._seen: set[tuple[str, tuple[str, ...]]] = set()

    def check(self, mobjects: list[Any]) -> None:
        """Run all lint checks on current mobjects, collect new issues."""
        self.step += 1
        issues = run_all(mobjects)
        for issue in issues:
            key = (issue.code, issue.mobjects)
            if key not in self._seen:
                self._seen.add(key)
                self.all_issues.append((self.step, issue))

    def get_issues(self) -> list[Issue]:
        """Return deduplicated issues, annotated with first-seen step."""
        result: list[Issue] = []
        for step, issue in self.all_issues:
            annotated = Issue(
                severity=issue.severity,
                code=issue.code,
                message=f"[step {step}] {issue.message}",
                mobjects=issue.mobjects,
            )
            result.append(annotated)
        return result


def _patch_scene_animations(collector: _LintCollector) -> None:
    """Make `Scene.play`/`wait` no-ops that register mobjects and lint.

    After each `play()` call, mobjects from animations are added to the scene
    and then ``run_all`` checks the current visible state. ``wait()`` also
    triggers a lint pass. This catches issues at every keyframe — not just
    the final frame.
    """
    from manim import Scene  # imported lazily so manim only loads when lint is run

    def _collect(args: Iterable[Any]) -> list[Any]:
        out: list[Any] = []
        for a in args:
            m = getattr(a, "mobject", None)
            if m is not None:
                out.append(m)
                continue
            ms = getattr(a, "mobjects", None)
            if ms:
                out.extend(ms)
                continue
            # Fallback: the user passed a bare Mobject (rare but legal).
            if hasattr(a, "get_center"):
                out.append(a)
        return out

    def _play(self: Scene, *animations: Any, **_: Any) -> None:
        for m in _collect(animations):
            if m not in self.mobjects:
                self.add(m)
        collector.check(list(self.mobjects))

    def _wait(self: Scene, *_: Any, **__: Any) -> None:
        collector.check(list(self.mobjects))

    Scene.play = _play  # type: ignore[method-assign]
    Scene.wait = _wait  # type: ignore[method-assign]
    # Keep `add` / `remove` / `clear` working — they already mutate `mobjects`.


def lint_scene(scene: str, scene_file: Path) -> list[Issue]:
    """Import the scene, run construct() with per-keyframe linting."""
    collector = _LintCollector()
    _patch_scene_animations(collector)
    mod = _import_scene_module(scene_file)
    cls = getattr(mod, scene, None)
    if cls is None:
        raise AttributeError(f"class {scene} not found in {scene_file}")

    instance = cls()
    instance.construct()

    # Also lint the final state (in case construct ends without play/wait)
    collector.check(list(instance.mobjects))

    return collector.get_issues()


def _suggest_venv_run(argv: list[str]) -> None:
    """Print a hint to re-run inside the manim venv when imports fail."""
    print(
        "[lint_scene] manim import failed. Try running under the venv:\n"
        f"    cd seminars/manim && source bin/activate_env.sh && "
        f".venv/bin/python scripts/lint_scene.py {' '.join(argv)}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", nargs="?", help="scene class name or path to .py file")
    p.add_argument("--scene", help="scene class name (alternative to positional)")
    args = p.parse_args(argv)

    target = args.scene or args.target
    if not target:
        p.print_usage(sys.stderr)
        return 3

    if target.endswith(".py") and Path(target).exists():
        scene_file = Path(target).resolve()
        scene = scene_file.stem
    else:
        scene = target
        found = _find_scene_file(scene)
        if found is None:
            print(f"[lint_scene] no scene file found for '{scene}'", file=sys.stderr)
            return 3
        scene_file = found

    print(f"[lint_scene] linting {scene} ({scene_file.relative_to(REPO)})")

    try:
        issues = lint_scene(scene, scene_file)
    except ImportError as e:
        if "manim" in str(e).lower():
            _suggest_venv_run(argv or sys.argv[1:])
        else:
            traceback.print_exc()
        return 2
    except Exception:
        traceback.print_exc()
        return 2

    if not issues:
        print("[lint_scene] OK — 0 issues")
        return 0

    print(format_report(issues))
    high = [i for i in issues if i.severity == "high"]
    print(
        f"[lint_scene] {len(issues)} issue(s): "
        f"{len(high)} high, "
        f"{sum(1 for i in issues if i.severity == 'med')} med, "
        f"{sum(1 for i in issues if i.severity == 'low')} low"
    )
    return 1 if high else 0


if __name__ == "__main__":
    sys.exit(main())
