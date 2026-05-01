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
from manim import Arrow, MathTex, Mobject, Tex, VMobject

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
        if not isinstance(m, (Tex, MathTex)):
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
            has_fix = hasattr(m, "move_cells_to")
            fix_hint = (
                " Fix: use .move_cells_to([x, y, 0]) instead of .move_to()"
                if has_fix
                else " Fix: position based on cells center, not group center"
            )
            issues.append(
                Issue(
                    "high",
                    "label-centering-jitter",
                    f"{nm} cells center offset {offset:.3f} > {tolerance:.3f} "
                    f"from group center — will cause jitter when swapping "
                    f"with label-less ghost.{fix_hint}",
                    (nm,),
                )
            )
    return issues


def check_label_arrow_overlap(
    mobjects: Iterable[Mobject],
    padding: float = 0.06,
) -> list[Issue]:
    """Flag Tex/MathTex labels whose bounding box overlaps an arrow.

    Common case: a label placed next to a tensor column gets clipped by
    an arrow going out of that column. The check inflates the label AABB
    by ``padding`` to catch near-misses.
    """
    issues: list[Issue] = []
    all_mobs = _walk(list(mobjects))
    arrows = [m for m in all_mobs if isinstance(m, Arrow)]
    labels = [m for m in all_mobs if isinstance(m, (Tex, MathTex)) and _has_geometry(m)]

    for lab in labels:
        lx0, lx1, ly0, ly1 = _aabb(lab)
        # inflate
        lx0 -= padding
        lx1 += padding
        ly0 -= padding
        ly1 += padding
        lab_aabb = (lx0, lx1, ly0, ly1)
        text = getattr(lab, "tex_string", "") or ""
        nm = f"{type(lab).__name__}({text[:30]})" if text else type(lab).__name__

        for arr in arrows:
            if not _has_geometry(arr):
                continue
            try:
                start = np.asarray(arr.get_start(), dtype=float)
                end = np.asarray(arr.get_end(), dtype=float)
            except Exception:
                continue
            if np.linalg.norm(end - start) < 1e-6:
                continue
            # Check: does the arrow line pass through the inflated label AABB?
            hit = False
            for t in np.linspace(0.0, 1.0, 25):
                p = start + (end - start) * t
                if _point_in_aabb(p, lab_aabb):
                    hit = True
                    break
            # Also check if the arrow AABB overlaps with label AABB
            if not hit:
                arr_aabb = _aabb(arr)
                hit = _aabbs_overlap(lab_aabb, arr_aabb)
                # Refine: AABB overlap doesn't mean the line actually passes through.
                # For straight arrows this is conservative enough.
            if hit:
                issues.append(
                    Issue(
                        "high",
                        "label-arrow-overlap",
                        f"{nm} overlaps with arrow from "
                        f"({start[0]:.2f},{start[1]:.2f}) to "
                        f"({end[0]:.2f},{end[1]:.2f}). "
                        f"Fix: increase buff or change label position "
                        f"(e.g. UP/DOWN instead of LEFT/RIGHT)",
                        (nm,),
                    )
                )

    return issues


def check_label_label_overlap(
    mobjects: Iterable[Mobject],
    padding: float = 0.02,
) -> list[Issue]:
    """Flag pairs of Tex/MathTex labels whose bounding boxes overlap.

    Common case: a label placed next to a matrix row overlaps with another
    label (e.g. matrix title "A" overlaps with row label "bites").
    """
    issues: list[Issue] = []
    all_mobs = _walk(list(mobjects))
    labels = [m for m in all_mobs if isinstance(m, (Tex, MathTex)) and _has_geometry(m)]

    seen: set[tuple[int, int]] = set()
    for i, lab_a in enumerate(labels):
        ax0, ax1, ay0, ay1 = _aabb(lab_a)
        ax0 -= padding; ax1 += padding; ay0 -= padding; ay1 += padding
        aabb_a = (ax0, ax1, ay0, ay1)
        text_a = getattr(lab_a, "tex_string", "") or ""
        nm_a = f"{type(lab_a).__name__}({text_a[:30]})" if text_a else type(lab_a).__name__

        for j, lab_b in enumerate(labels):
            if j <= i:
                continue
            pair = (id(lab_a), id(lab_b))
            if pair in seen:
                continue
            seen.add(pair)

            bx0, bx1, by0, by1 = _aabb(lab_b)
            bx0 -= padding; bx1 += padding; by0 -= padding; by1 += padding
            aabb_b = (bx0, bx1, by0, by1)

            if _aabbs_overlap(aabb_a, aabb_b):
                text_b = getattr(lab_b, "tex_string", "") or ""
                nm_b = f"{type(lab_b).__name__}({text_b[:30]})" if text_b else type(lab_b).__name__
                issues.append(
                    Issue(
                        "high",
                        "label-label-overlap",
                        f"{nm_a} overlaps with {nm_b}. "
                        f"Fix: increase spacing or reposition labels",
                        (nm_a, nm_b),
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
    issues.extend(check_label_arrow_overlap(mobjects))
    issues.extend(check_label_label_overlap(mobjects))

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
