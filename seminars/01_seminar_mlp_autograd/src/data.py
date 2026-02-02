"""
Data utilities for Manim Deep Learning visualizations.
"""

import numpy as np
import torch
from sklearn.datasets import make_moons


def get_moons_data(n_samples=100, noise=0.1, random_state=1):
    """Generate moons dataset for classification."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def get_decision_boundary(model, X, h=0.05, x_range=None, y_range=None):
    """Get decision boundary mesh for visualization.

    Uses fine-grained grid (h=0.05) for smooth boundaries.
    If x_range and y_range are provided, uses those instead of data bounds.
    """
    if x_range is not None and y_range is not None:
        x_min, x_max = x_range
        y_min, y_max = y_range
    else:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    Xmesh = np.c_[xx.ravel(), yy.ravel()]

    with torch.no_grad():
        Xbatch = torch.tensor(Xmesh).float()
        scores = model.forward(Xbatch)
        Z = scores.numpy()

    Z = Z.reshape(xx.shape)
    return xx, yy, Z
