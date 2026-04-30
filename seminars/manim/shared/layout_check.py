"""Static layout linter for ManimCE scenes.

Runs **before** a scene is rendered. Inspects mobject bounding boxes that
manim has already computed (via LaTeX for `MathTex`, geometric for `Circle` /
`RoundedRectangle`, etc.) and flags issues that would otherwise only surface
on a vision pass over a rendered frame. A few seconds of linting replaces
30–60 s of render + ffmpeg sample + vision read.

The checks are conservative: they flag geometric facts (BB outside frame,
BB overlap, arrow segment piercing a non-endpoint mobject, MathTex below a
size threshold). They do not attempt to second-guess semantics.

Public entry point: :func:`run_all`. Returned `Issue` list can be filtered
by severity; callers typically fail on any `high` issue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np
from manim import Arrow, Mobject, Tex, VMobject

Severity = Literal["high", "med", "low"]

# ManimCE `-qm` / 720p camera: frame_width=14.22, frame_height=8.
FRAME_X_HALF = 7.11
FRAME_Y_HALF = 4.0


@dataclass
class Issue:
    """A single linter finding."""

    severity: Severity
    code: str
    message: str
    mobjects: tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        tag = f"[{self.severity.upper()}] {self.code}"
        refs = f" ({', '.join(self.mobjects)})" if self.mobjects else ""
        return f"{tag}{refs}: {self.message}"


def _aabb(mobj: Mobject) -> tuple[float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax) of a mobject's axis-aligned bounding box."""
    left = float(mobj.get_left()[0])
    right = float(mobj.get_right()[0])
    bottom = float(mobj.get_bottom()[1])
    top = float(mobj.get_top()[1])
    return left, right, bottom, top


def _point_in_aabb(p: np.ndarray, aabb: tuple[float, float, float, float]) -> bool:
    xmin, xmax, ymin, ymax = aabb
    return xmin <= float(p[0]) <= xmax and ymin <= float(p[1]) <= ymax


def _aabbs_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _name_of(mobj: Mobject) -> str:
    """Best-effort human label for a mobject."""
    label = getattr(mobj, "label_tex", None)
    if label is not None:
        text = getattr(label, "tex_string", None) or getattr(label, "get_tex_string", lambda: None)()
        if text:
            return f"{type(mobj).__name__}({text})"
    return type(mobj).__name__


def _walk(mobjects: Iterable[Mobject]) -> list[Mobject]:
    """Flatten the mobject tree (pre-order)."""
    out: list[Mobject] = []
    for m in mobjects:
        out.append(m)
        out.extend(_walk(m.submobjects))
    return out


def check_in_frame(
    mobj: Mobject,
    name: str | None = None,
    margin: float = 0.05,
) -> list[Issue]:
    """`high` if BB is outside the frame; `med` if within `margin` of an edge."""
    issues: list[Issue] = []
    if not _has_geometry(mobj):
        return issues
    xmin, xmax, ymin, ymax = _aabb(mobj)
    nm = name or _name_of(mobj)

    if xmin < -FRAME_X_HALF or xmax > FRAME_X_HALF or ymin < -FRAME_Y_HALF or ymax > FRAME_Y_HALF:
        issues.append(
            Issue(
                "high",
                "offscreen",
                f"{nm} BB=({xmin:.2f},{xmax:.2f},{ymin:.2f},{ymax:.2f}) "
                f"exits frame ±({FRAME_X_HALF},{FRAME_Y_HALF})",
                (nm,),
            )
        )
        return issues
    if (
        xmin < -FRAME_X_HALF + margin
        or xmax > FRAME_X_HALF - margin
        or ymin < -FRAME_Y_HALF + margin
        or ymax > FRAME_Y_HALF - margin
    ):
        issues.append(
            Issue(
                "med",
                "near-edge",
                f"{nm} BB within {margin:.2f} of frame edge",
                (nm,),
            )
        )
    return issues


def _has_geometry(mobj: Mobject) -> bool:
    """True if the mobject has a meaningful bounding box we can measure."""
    try:
        _ = mobj.get_left()[0]
        _ = mobj.get_right()[0]
    except Exception:
        return False
    # Some containers have degenerate BBs (all zeros). Treat as geometry-less.
    try:
        xmin, xmax, ymin, ymax = _aabb(mobj)
        if (xmax - xmin) < 1e-6 and (ymax - ymin) < 1e-6:
            return False
    except Exception:
        return False
    return True


def check_min_label_scale(
    mobjects: Iterable[Mobject],
    min_height: float = 0.18,
    low_height: float = 0.14,
) -> list[Issue]:
    """Flag `Tex` / `MathTex` / `Text` labels with rendered height below thresholds.

    Default ManimCE `MathTex("f").height` is ≈ 0.36. Scale 0.5 → 0.18.
    Below 0.4 (height ≈ 0.14) is hard to read at 720p.
    """
    issues: list[Issue] = []
    for m in _walk(list(mobjects)):
        if not isinstance(m, Tex):
            continue
        h = float(m.height)
        if h <= 0:
            continue
        text = getattr(m, "tex_string", "") or ""
        nm = f"{type(m).__name__}({text[:40]})" if text else type(m).__name__
        if h < low_height:
            issues.append(
                Issue("high", "label-too-small", f"{nm} height={h:.3f} < {low_height}", (nm,))
            )
        elif h < min_height:
            issues.append(
                Issue("med", "label-small", f"{nm} height={h:.3f} < {min_height}", (nm,))
            )
    return issues


def check_pair_no_overlap(
    a: Mobject,
    b: Mobject,
    name_a: str | None = None,
    name_b: str | None = None,
) -> list[Issue]:
    """`high` if `a` and `b` have overlapping axis-aligned bounding boxes."""
    if not (_has_geometry(a) and _has_geometry(b)):
        return []
    na = name_a or _name_of(a)
    nb = name_b or _name_of(b)
    if _aabbs_overlap(_aabb(a), _aabb(b)):
        return [
            Issue(
                "high",
                "overlap",
                f"{na} and {nb} have overlapping bounding boxes",
                (na, nb),
            )
        ]
    return []


def check_arrow_path_clear(
    arrow: Arrow,
    obstacles: Sequence[Mobject],
    samples: int = 20,
    endpoint_tolerance: float = 0.35,
) -> list[Issue]:
    """`high` if the arrow segment passes through a non-endpoint obstacle.

    Arrow endpoints are identified by proximity of an obstacle's center to
    the arrow's start or end (within `endpoint_tolerance`). Those are skipped
    — only obstacles the arrow "flies over" count as violations.
    """
    issues: list[Issue] = []
    try:
        start = np.asarray(arrow.get_start(), dtype=float)
        end = np.asarray(arrow.get_end(), dtype=float)
    except Exception:
        return issues
    if np.linalg.norm(end - start) < 1e-6:
        return issues

    sampled = [start + (end - start) * t for t in np.linspace(0.05, 0.95, samples)]

    for ob in obstacles:
        if ob is arrow or not _has_geometry(ob):
            continue
        try:
            center = np.asarray(ob.get_center(), dtype=float)
        except Exception:
            continue
        # Skip arrow's own endpoints
        if (
            np.linalg.norm(center[:2] - start[:2]) < endpoint_tolerance
            or np.linalg.norm(center[:2] - end[:2]) < endpoint_tolerance
        ):
            continue
        aabb = _aabb(ob)
        if any(_point_in_aabb(p, aabb) for p in sampled):
            nm = _name_of(ob)
            issues.append(
                Issue(
                    "high",
                    "arrow-clips",
                    f"arrow from ({start[0]:.2f},{start[1]:.2f}) to "
                    f"({end[0]:.2f},{end[1]:.2f}) passes through {nm} "
                    f"AABB=({aabb[0]:.2f},{aabb[1]:.2f},{aabb[2]:.2f},{aabb[3]:.2f})",
                    (nm,),
                )
            )
    return issues


def _collect_arrows_and_obstacles(
    mobjects: Iterable[Mobject],
) -> tuple[list[Arrow], list[Mobject]]:
    """Split scene tree into arrows vs. other "obstacle" mobjects.

    Obstacles are top-level mobjects that are not arrows. We intentionally
    keep the granularity coarse (use the top-level mobjects, not their
    children) so a `Neuron` (Circle+MathTex VGroup) counts as one obstacle.
    """
    arrows: list[Arrow] = []
    obstacles: list[Mobject] = []
    for m in mobjects:
        if isinstance(m, Arrow):
            arrows.append(m)
        elif _has_geometry(m) and not _is_tex_only(m):
            obstacles.append(m)
    return arrows, obstacles


def _is_tex_only(m: Mobject) -> bool:
    """True if `m` is (or is a container of only) Tex labels — not a solid shape.

    We don't want title equations to count as obstacles for arrow paths.
    """
    if isinstance(m, Tex):
        return True
    if isinstance(m, VMobject) and m.submobjects:
        return all(_is_tex_only(s) for s in m.submobjects)
    return False


def check_labeled_group_centering(
    mobjects: Iterable[Mobject],
    tolerance: float = 0.08,
) -> list[Issue]:
    """Flag VGroups where a label causes the cells/core to be off-center.

    Catches the common jitter bug: a TensorColumn (or similar) with a label
    has its bounding-box center shifted away from the geometric center of
    the core shapes.  When such an object replaces a label-less ghost in an
    animation, the core shapes visibly jump.

    Works on any VGroup that has a ``cells`` attribute (list of submobjects)
    and a ``label_tex`` attribute.
    """
    issues: list[Issue] = []
    for m in _walk(list(mobjects)):
        cells = getattr(m, "cells", None)
        label_tex = getattr(m, "label_tex", None)
        if cells is None or label_tex is None or not cells:
            continue
        # Compute cells-only center vs VGroup center
        cells_center = np.mean([c.get_center() for c in cells], axis=0)
        group_center = np.asarray(m.get_center(), dtype=float)
        offset = float(np.linalg.norm(cells_center[:2] - group_center[:2]))
        if offset > tolerance:
            nm = _name_of(m)
            issues.append(
                Issue(
                    "high",
                    "label-centering-jitter",
                    f"{nm} cells center offset {offset:.3f} > {tolerance:.3f} "
                    f"from group center — will cause jitter in animations",
                    (nm,),
                )
            )
    return issues


def run_all(
    mobjects: Sequence[Mobject],
    *,
    frame_margin: float = 0.05,
    min_label_height: float = 0.18,
    check_arrows: bool = True,
) -> list[Issue]:
    """Run the standard check suite against a scene's top-level mobjects."""
    issues: list[Issue] = []

    for m in mobjects:
        issues.extend(check_in_frame(m, margin=frame_margin))

    issues.extend(check_min_label_scale(mobjects, min_height=min_label_height))
    issues.extend(check_labeled_group_centering(mobjects))

    if check_arrows:
        arrows, obstacles = _collect_arrows_and_obstacles(mobjects)
        for arr in arrows:
            issues.extend(check_arrow_path_clear(arr, obstacles))

    return issues


def format_report(issues: Sequence[Issue]) -> str:
    """Human-readable multi-line summary. Empty string if `issues` is empty."""
    if not issues:
        return ""
    by_sev: dict[Severity, list[Issue]] = {"high": [], "med": [], "low": []}
    for i in issues:
        by_sev[i.severity].append(i)
    lines: list[str] = []
    for sev in ("high", "med", "low"):
        bucket = by_sev[sev]  # type: ignore[index]
        if not bucket:
            continue
        lines.append(f"-- {sev.upper()} ({len(bucket)}) --")
        for i in bucket:
            lines.append(f"  {i}")
    return "\n".join(lines)
