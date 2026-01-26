"""
Shared utilities for Manim Deep Learning visualizations.
"""

from .models import MLP1Hidden, train_model, train_width_model, get_or_train_width_model
from .data import get_moons_data, get_decision_boundary
from .visualization import (
    create_decision_boundary_mobject,
    create_data_points,
    fit_linear_separator,
    count_separator_errors,
    separator_to_line_on_axes,
    value_to_color,
)
from .autograd import Value, Neuron, Layer, SimpleMLP
