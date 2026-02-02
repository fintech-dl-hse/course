"""
Visualization utilities for Manim Deep Learning animations.
"""

import numpy as np
from sklearn.svm import LinearSVC

# Import Manim classes - will be available when running with manimgl
try:
    from manim_imports_ext import *
except ImportError:
    # Fallback for type checking
    pass


def value_to_color(
    value,
    low_positive_color=None,
    high_positive_color=None,
    low_negative_color=None,
    high_negative_color=None,
    min_value=0.0,
    max_value=10.0
):
    """Map a numerical value to a color based on its sign and magnitude."""
    # Default colors if not provided
    if low_positive_color is None:
        from manim_imports_ext import BLUE_E
        low_positive_color = BLUE_E
    if high_positive_color is None:
        from manim_imports_ext import BLUE_B
        high_positive_color = BLUE_B
    if low_negative_color is None:
        from manim_imports_ext import RED_E
        low_negative_color = RED_E
    if high_negative_color is None:
        from manim_imports_ext import RED_B
        high_negative_color = RED_B

    from manim_imports_ext import clip, inverse_interpolate, interpolate_color_by_hsl

    alpha = clip(float(inverse_interpolate(min_value, max_value, abs(value))), 0, 1)
    if value >= 0:
        colors = (low_positive_color, high_positive_color)
    else:
        colors = (low_negative_color, high_negative_color)
    return interpolate_color_by_hsl(*colors, alpha)


def create_decision_boundary_mobject(axes, xx, yy, Z, colors=None, opacity=0.5, step=1):
    """Create a Manim mobject representing the decision boundary.

    Uses fine-grained polygons for smooth boundaries.
    """
    from manim_imports_ext import VGroup, Polygon, BLUE_E, RED_E

    if colors is None:
        colors = [BLUE_E, RED_E]

    positive_regions = VGroup()
    negative_regions = VGroup()

    for i in range(0, len(xx) - step, step):
        for j in range(0, len(xx[0]) - step, step):
            try:
                corners = [
                    axes.c2p(xx[i, j], yy[i, j]),
                    axes.c2p(xx[i+step, j], yy[i+step, j]),
                    axes.c2p(xx[i+step, j+step], yy[i+step, j+step]),
                    axes.c2p(xx[i, j+step], yy[i, j+step]),
                ]
                center_val = Z[i, j]
                if center_val > 0:
                    poly = Polygon(*corners, fill_opacity=opacity, stroke_width=0)
                    poly.set_fill(colors[0])
                    positive_regions.add(poly)
                else:
                    poly = Polygon(*corners, fill_opacity=opacity, stroke_width=0)
                    poly.set_fill(colors[1])
                    negative_regions.add(poly)
            except (IndexError, ValueError):
                continue

    return VGroup(positive_regions, negative_regions)


def create_data_points(axes, X, y, colors=None, radius=0.08):
    """Create Manim dots for data points."""
    from manim_imports_ext import VGroup, Dot, BLUE, RED

    if colors is None:
        colors = [BLUE, RED]

    dots = VGroup()
    for point, label in zip(X, y):
        dot = Dot(axes.c2p(point[0], point[1]), radius=radius)
        dot.set_color(colors[label])
        dots.add(dot)
    return dots


def fit_linear_separator(X, y):
    """Fit a linear separator (2D) and return (w, b) for w^T x + b = 0."""
    clf = LinearSVC(C=1.0, max_iter=50_000, random_state=0)
    clf.fit(X, y)
    w = clf.coef_[0].astype(float)
    b = float(clf.intercept_[0])
    return w, b


def count_separator_errors(X, y, w, b):
    """Count mistakes of separator sign(w^T x + b) over labels y in {0,1}."""
    scores = X @ w + b
    y_pred = (scores > 0).astype(int)
    return int(np.sum(y_pred != y))


def separator_to_line_on_axes(axes, w, b, color=None, stroke_width=6):
    """Create a Line mobject for w^T x + b = 0 within current axes ranges."""
    from manim_imports_ext import Line, YELLOW

    if color is None:
        color = YELLOW

    x_min, x_max = float(axes.x_range[0]), float(axes.x_range[1])
    y_min, y_max = float(axes.y_range[0]), float(axes.y_range[1])

    w1, w2 = float(w[0]), float(w[1])
    eps = 1e-8

    if abs(w2) > eps:
        x1, x2 = x_min, x_max
        y1 = -(w1 * x1 + b) / w2
        y2 = -(w1 * x2 + b) / w2
    else:
        x1 = x2 = -b / (w1 if abs(w1) > eps else eps)
        y1, y2 = y_min, y_max

    p1 = axes.c2p(x1, y1)
    p2 = axes.c2p(x2, y2)
    line = Line(p1, p2, color=color, stroke_width=stroke_width)
    return line
