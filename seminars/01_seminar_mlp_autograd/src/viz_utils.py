"""Visualization utilities for Manim animations."""

from manim_imports_ext import *


def create_decision_boundary_mobject(axes, xx, yy, Z, colors=[BLUE_E, RED_E], opacity=0.5, step=1):
    """Create a Manim mobject representing the decision boundary.

    Uses fine-grained polygons for smooth boundaries.
    """
    positive_regions = VGroup()
    negative_regions = VGroup()

    # Use step=1 for maximum smoothness (no skipping cells)
    for i in range(0, len(xx) - step, step):
        for j in range(0, len(xx[0]) - step, step):
            # Get four corners of the cell
            try:
                corners = [
                    axes.c2p(xx[i, j], yy[i, j]),
                    axes.c2p(xx[i+step, j], yy[i+step, j]),
                    axes.c2p(xx[i+step, j+step], yy[i+step, j+step]),
                    axes.c2p(xx[i, j+step], yy[i, j+step]),
                ]
                # Determine which region based on center value
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


def create_data_points(axes, X, y, colors=[BLUE, RED], radius=0.08):
    """Create Manim dots for data points."""
    dots = VGroup()
    for point, label in zip(X, y):
        dot = Dot(axes.c2p(point[0], point[1]), radius=radius)
        dot.set_color(colors[label])
        dots.add(dot)
    return dots

