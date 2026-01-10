"""
MLP (Multi-Layer Perceptron) Visualization for Deep Learning Course
Educational animations demonstrating MLP concepts from basic linear transformations
to deep networks and universal approximation.
"""

import sys
sys.path.append('/Users/d.tarasov/workspace/hse/fintech-dl-hse/videos')

from manim_imports_ext import *
from _2024.transformers.helpers import NeuralNetwork, WeightMatrix, value_to_color

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_moons


# ============================================================================
# Model Definition and Training Utilities
# ============================================================================

class MLP(nn.Module):
    """MLP model for moons dataset classification."""
    def __init__(self, activation_cls=nn.ReLU):
        super().__init__()
        self.input_layer = nn.Linear(2, 100)
        self.hidden_layer = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, 1)
        self.activation = activation_cls()

    def forward(self, x_coordinates):
        # x_coordinates ~ [ batch_size, 2 ]
        latents = self.activation(self.input_layer(x_coordinates))  # [ batch_size, 100 ]
        latents = self.activation(self.hidden_layer(latents))  # [ batch_size, 100 ]
        scores = self.output_layer(latents)  # [ batch_size, 1 ]
        scores = scores[:, 0]  # [ batch_size ]
        return scores


def loss_function(model, Xbatch, ybatch):
    """Compute loss for the model."""
    Xbatch = torch.tensor(Xbatch).float()
    ybatch = torch.tensor(ybatch).float().unsqueeze(-1)
    model_prediction = model.forward(Xbatch).unsqueeze(-1)
    losses = F.relu(1 - ybatch * model_prediction)
    loss = losses.mean()
    alpha = 1e-4
    reg_loss = alpha * sum((p * p).sum() for p in model.parameters())
    total_loss = loss + reg_loss
    accuracy = ((ybatch > 0) == (model_prediction > 0)).float().mean()
    return total_loss, accuracy


def train_model(model, learning_rate=0.05, n_steps=500):
    """Train the MLP model."""
    Xbatch, ybatch = make_moons(n_samples=100, noise=0.1, random_state=1)
    ybatch = ybatch * 2 - 1  # make y be -1 or 1

    for k in range(n_steps):
        model.zero_grad()
        total_loss, acc = loss_function(model, Xbatch, ybatch)
        total_loss.backward()
        for p in model.parameters():
            p.data = p.data - learning_rate * p.grad

    return model


def get_decision_boundary(model, X, h=0.1):
    """Get decision boundary mesh for visualization."""
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


# ============================================================================
# Visualization Utilities
# ============================================================================

def create_decision_boundary_mobject(axes, xx, yy, Z, colors=[BLUE_E, RED_E], opacity=0.5, step=2):
    """Create a Manim mobject representing the decision boundary."""
    # Use a more efficient approach: sample the grid and create polygons
    positive_regions = VGroup()
    negative_regions = VGroup()

    # Sample every 'step' cells to reduce computation
    for i in range(0, len(xx) - 1, step):
        for j in range(0, len(xx[0]) - 1, step):
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


# ============================================================================
# Main Scene
# ============================================================================

class MLPVisualization(InteractiveScene):
    """Main scene for MLP visualization."""
    def construct(self):
        # Introduction
        self.introduction()

        # Prepare data
        X, y = make_moons(n_samples=100, noise=0.1, random_state=1)
        y_binary = y * 2 - 1  # Convert to -1, 1

        # Train models
        print("Training MLP with ReLU...")
        model_relu = MLP(activation_cls=nn.ReLU)
        model_relu = train_model(model_relu, learning_rate=0.05, n_steps=500)

        print("Training MLP without nonlinearity...")
        model_linear = MLP(activation_cls=nn.Identity)
        model_linear = train_model(model_linear, learning_rate=0.05, n_steps=500)

        # Scene 1: Linear Transformation
        self.scene1_linear_transformation(X, y)

        # Scene 2: Dimensionality Expansion
        self.scene2_dimensionality_expansion(X, y)

        # Scene 3: MLP with Nonlinearity
        self.scene3_mlp_nonlinearity(X, y, model_relu)

        # Scene 4: Comparison
        self.scene4_comparison(X, y, model_relu, model_linear)

        # Conclusion
        self.conclusion()

    def introduction(self):
        """Introduction scene."""
        title = Text("From Linear Transformations to MLPs", font_size=64)
        title.to_edge(UP, buff=1)

        subtitle = Text("Understanding Multi-Layer Perceptrons", font_size=40)
        subtitle.next_to(title, DOWN, buff=0.5)

        self.play(Write(title))
        self.wait(0.5)
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(2)

        self.play(
            FadeOut(title),
            FadeOut(subtitle)
        )
        self.wait(0.5)

    def conclusion(self):
        """Conclusion scene."""
        conclusion_text = Text("Key Takeaways", font_size=60)
        conclusion_text.to_edge(UP, buff=0.5)

        takeaways = VGroup(
            Text("1. Linear transformations preserve linear separability", font_size=32),
            Text("2. Adding dimensions provides more flexibility", font_size=32),
            Text("3. Nonlinearity enables learning complex boundaries", font_size=32),
            Text("4. Composition of linear layers = single linear layer", font_size=32)
        )
        takeaways.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        takeaways.center()

        self.play(Write(conclusion_text))
        self.wait(0.5)
        self.play(LaggedStartMap(FadeIn, takeaways, lag_ratio=0.3, shift=UP))
        self.wait(3)

        self.play(
            FadeOut(conclusion_text),
            FadeOut(takeaways)
        )
        self.wait(0.5)

    def scene1_linear_transformation(self, X, y):
        """Scene 1: Show linear transformation on 2D plane."""
        # Title
        title = Text("Linear Transformation", font_size=60)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(1)

        # Create axes
        axes = Axes(
            x_range=[-2, 3, 0.5],
            y_range=[-2, 2, 0.5],
            width=10,
            height=6,
            axis_config={"include_tip": True}
        )
        axes.center().shift(0.5 * DOWN)

        # Show data points
        dots = create_data_points(axes, X, y)
        self.play(FadeIn(axes), LaggedStartMap(FadeIn, dots, lag_ratio=0.02))
        self.wait(1)

        # Label the data
        data_label = Text("Moons Dataset", font_size=32)
        data_label.next_to(axes, DOWN, buff=0.3)
        self.play(Write(data_label))
        self.wait(0.5)

        # Show linear transformation formula
        formula = Tex("f(x) = Wx + b", font_size=48)
        formula.to_edge(RIGHT).shift(UP * 2.5)
        self.play(Write(formula))
        self.wait(1)

        # Create a simple 2x2 transformation matrix
        W = np.array([[1.2, 0.3], [-0.2, 0.8]])
        b = np.array([0.1, -0.1])

        # Show matrix representation
        matrix_text = Tex("W = \\begin{bmatrix} 1.2 & 0.3 \\\\ -0.2 & 0.8 \\end{bmatrix}", font_size=36)
        matrix_text.next_to(formula, DOWN, buff=0.5)
        self.play(Write(matrix_text))
        self.wait(1)

        # Apply transformation
        X_transformed = (X @ W.T) + b

        # Transform dots
        dots_transformed = create_data_points(axes, X_transformed, y)
        dots_transformed.set_opacity(0)

        # Animate transformation
        self.play(
            Transform(dots, dots_transformed, path_arc=PI/4),
            FadeOut(data_label),
            run_time=2
        )
        self.wait(1)

        # Show that it's still linearly separable
        text = Text("Linear transformations preserve\nlinear separability", font_size=36)
        text.to_edge(DOWN, buff=0.5)
        self.play(Write(text))
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(formula),
            FadeOut(matrix_text),
            FadeOut(text),
            FadeOut(axes),
            FadeOut(dots)
        )
        self.wait(0.5)

    def scene2_dimensionality_expansion(self, X, y):
        """Scene 2: Show dimensionality expansion."""
        # Title
        title = Text("Expanding Dimensions", font_size=60)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(1)

        # Create 2D axes
        axes_2d = Axes(
            x_range=[-2, 3, 0.5],
            y_range=[-2, 2, 0.5],
            width=6,
            height=6,
            axis_config={"include_tip": True}
        )
        axes_2d.to_edge(LEFT).shift(0.5 * DOWN)

        # Show 2D data
        dots_2d = create_data_points(axes_2d, X, y)
        self.play(FadeIn(axes_2d), LaggedStartMap(FadeIn, dots_2d, lag_ratio=0.02))
        self.wait(1)

        # Arrow showing expansion
        arrow = Arrow(RIGHT, RIGHT * 2, buff=0.5)
        arrow.move_to(ORIGIN)
        expansion_text = Text("2D → Higher Dim", font_size=36)
        expansion_text.next_to(arrow, UP)

        self.play(
            GrowArrow(arrow),
            Write(expansion_text)
        )
        self.wait(1)

        # Create 3D visualization
        axes_3d = ThreeDAxes(
            x_range=[-2, 3, 0.5],
            y_range=[-2, 2, 0.5],
            z_range=[-1, 1, 0.5],
            width=6,
            height=6,
            depth=4
        )
        axes_3d.to_edge(RIGHT).shift(0.5 * DOWN)

        # Project to 3D (add a third dimension based on distance from origin)
        X_3d = np.column_stack([X, np.linalg.norm(X, axis=1) * 0.3])

        # Create 3D dots
        dots_3d = VGroup()
        for point, label in zip(X_3d, y):
            dot = Dot(
                axes_3d.c2p(point[0], point[1], point[2]),
                radius=0.08
            )
            dot.set_color(BLUE if label == 0 else RED)
            dots_3d.add(dot)

        # Set camera for 3D
        self.frame.reorient(20, -70, 0)
        self.add(axes_3d)

        # Animate transformation to 3D
        dots_3d_copy = dots_3d.copy()
        dots_3d_copy.set_opacity(0)
        self.play(
            Transform(dots_2d.copy(), dots_3d_copy, path_arc=PI/3),
            run_time=2
        )
        self.remove(dots_3d_copy)
        self.add(dots_3d)
        self.wait(2)

        # Text about more flexibility
        text = Text("More dimensions provide\nmore flexibility", font_size=36)
        text.to_edge(DOWN, buff=0.5)
        self.play(Write(text))
        self.wait(2)

        # Clean up
        self.frame.reorient(0, 0, 0)
        self.play(
            FadeOut(title),
            FadeOut(axes_2d),
            FadeOut(dots_2d),
            FadeOut(arrow),
            FadeOut(expansion_text),
            FadeOut(axes_3d),
            FadeOut(dots_3d),
            FadeOut(text)
        )
        self.wait(0.5)

    def scene3_mlp_nonlinearity(self, X, y, model):
        """Scene 3: Show MLP with nonlinearity."""
        # Title
        title = Text("MLP with Nonlinearity", font_size=60)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(1)

        # Show MLP architecture
        network = NeuralNetwork([2, 10, 10, 1])
        network.scale(0.8)
        network.to_edge(LEFT).shift(UP * 0.5)

        self.play(LaggedStartMap(FadeIn, network.layers, lag_ratio=0.3))
        self.play(LaggedStartMap(ShowCreation, network.lines, lag_ratio=0.01, run_time=2))
        self.wait(1)

        # Label layers
        layer_labels = VGroup(
            Text("Input\n(2)", font_size=24),
            Text("Hidden\n(100)", font_size=24),
            Text("Hidden\n(100)", font_size=24),
            Text("Output\n(1)", font_size=24)
        )
        for i, label in enumerate(layer_labels):
            label.next_to(network.layers[i], DOWN, buff=0.3)

        self.play(LaggedStartMap(Write, layer_labels, lag_ratio=0.3))
        self.wait(1)

        # Show ReLU activation
        relu_text = Tex("\\text{ReLU}(x) = \\max(0, x)", font_size=36)
        relu_text.to_edge(RIGHT).shift(UP * 2)
        self.play(Write(relu_text))
        self.wait(1)

        # Move network to side and show decision boundary
        self.play(
            network.animate.scale(0.6).to_corner(UL, buff=0.5),
            LaggedStartMap(FadeOut, layer_labels),
            FadeOut(relu_text)
        )

        # Create axes for decision boundary
        axes = Axes(
            x_range=[-2, 3, 0.5],
            y_range=[-2, 2, 0.5],
            width=10,
            height=6,
            axis_config={"include_tip": True}
        )
        axes.center().shift(0.5 * DOWN)

        # Get decision boundary
        xx, yy, Z = get_decision_boundary(model, X, h=0.1)

        # Create decision boundary visualization
        boundary = create_decision_boundary_mobject(axes, xx, yy, Z, opacity=0.4)
        dots = create_data_points(axes, X, y)

        self.play(FadeIn(axes))
        self.play(FadeIn(boundary))
        self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.02))
        self.wait(1)

        # Highlight nonlinear regions
        text = Text("Nonlinearity enables learning\ncomplex decision boundaries", font_size=36)
        text.to_edge(DOWN, buff=0.5)
        self.play(Write(text))
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(network),
            FadeOut(axes),
            FadeOut(boundary),
            FadeOut(dots),
            FadeOut(text)
        )
        self.wait(0.5)

    def scene4_comparison(self, X, y, model_relu, model_linear):
        """Scene 4: Compare MLP with and without nonlinearity."""
        # Title
        title = Text("With vs Without Nonlinearity", font_size=60)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(1)

        # Create two side-by-side axes
        axes_left = Axes(
            x_range=[-2, 3, 0.5],
            y_range=[-2, 2, 0.5],
            width=6,
            height=5,
            axis_config={"include_tip": True}
        )
        axes_left.to_edge(LEFT, buff=1).shift(0.3 * DOWN)

        axes_right = Axes(
            x_range=[-2, 3, 0.5],
            y_range=[-2, 2, 0.5],
            width=6,
            height=5,
            axis_config={"include_tip": True}
        )
        axes_right.to_edge(RIGHT, buff=1).shift(0.3 * DOWN)

        # Labels
        label_relu = Text("With ReLU", font_size=40)
        label_relu.next_to(axes_left, UP)
        label_linear = Text("Without Nonlinearity", font_size=40)
        label_linear.next_to(axes_right, UP)

        # Get decision boundaries
        xx_relu, yy_relu, Z_relu = get_decision_boundary(model_relu, X, h=0.1)
        xx_linear, yy_linear, Z_linear = get_decision_boundary(model_linear, X, h=0.1)

        # Create visualizations
        boundary_relu = create_decision_boundary_mobject(axes_left, xx_relu, yy_relu, Z_relu, opacity=0.4)
        boundary_linear = create_decision_boundary_mobject(axes_right, xx_linear, yy_linear, Z_linear, opacity=0.4)

        dots_left = create_data_points(axes_left, X, y)
        dots_right = create_data_points(axes_right, X, y)

        # Show left side
        self.play(FadeIn(axes_left), Write(label_relu))
        self.play(FadeIn(boundary_relu))
        self.play(LaggedStartMap(FadeIn, dots_left, lag_ratio=0.02))
        self.wait(1)

        # Show right side
        self.play(FadeIn(axes_right), Write(label_linear))
        self.play(FadeIn(boundary_linear))
        self.play(LaggedStartMap(FadeIn, dots_right, lag_ratio=0.02))
        self.wait(1)

        # Key insight
        insight = Text("Composition of linear = linear", font_size=42)
        insight.to_edge(DOWN, buff=0.5)
        self.play(Write(insight))
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(axes_left),
            FadeOut(axes_right),
            FadeOut(boundary_relu),
            FadeOut(boundary_linear),
            FadeOut(dots_left),
            FadeOut(dots_right),
            FadeOut(label_relu),
            FadeOut(label_linear),
            FadeOut(insight)
        )
        self.wait(0.5)
