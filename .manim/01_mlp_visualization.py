"""
MLP (Multi-Layer Perceptron) Visualization for Deep Learning Course
Educational animations demonstrating MLP concepts from basic linear transformations
to deep networks and universal approximation.
"""

import sys
import os
sys.path.append('/Users/d.tarasov/workspace/hse/fintech-dl-hse/videos')

from manim_imports_ext import *
from _2024.transformers.helpers import NeuralNetwork, WeightMatrix, value_to_color

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import LinearSVC


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


# ============================================================================
# Model Loading/Saving Utilities
# ============================================================================

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATHS = {
    "relu": os.path.join(MODEL_DIR, "mlp_relu.pth"),
    "linear": os.path.join(MODEL_DIR, "mlp_linear.pth"),
}

WIDTH_MODEL_PATHS = {
    2: os.path.join(MODEL_DIR, "mlp_relu_width_2.pth"),
    4: os.path.join(MODEL_DIR, "mlp_relu_width_4.pth"),
    8: os.path.join(MODEL_DIR, "mlp_relu_width_8.pth"),
    16: os.path.join(MODEL_DIR, "mlp_relu_width_16.pth"),
}


def get_or_train_model(activation_cls=nn.ReLU, model_key="relu"):
    """Lazily load or train and save MLP model."""
    model_path = MODEL_PATHS[model_key]

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = MLP(activation_cls=activation_cls)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    else:
        print(f"Training model (not found at {model_path})...")
        model = MLP(activation_cls=activation_cls)
        model = train_model(model, learning_rate=0.05, n_steps=500)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        model.eval()
        return model


class MLP1Hidden(nn.Module):
    """Simple 1-hidden-layer MLP for width comparison (cached per width)."""

    def __init__(self, hidden_dim, activation_cls=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(2, int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), 1)
        self.activation = activation_cls()

    def forward(self, x_coordinates):
        h = self.activation(self.fc1(x_coordinates))
        scores = self.fc2(h)
        return scores[:, 0]


def train_width_model(model, learning_rate=0.02, n_steps=2500):
    """Train a width-model on moons dataset (binary classification)."""
    X, y = make_moons(n_samples=100, noise=0.1, random_state=1)
    X_t = torch.tensor(X).float()
    y_t = torch.tensor(y).float()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(n_steps):
        opt.zero_grad()
        logits = model.forward(X_t)
        loss = F.binary_cross_entropy_with_logits(logits, y_t)
        alpha = 1e-4
        reg = alpha * sum((p * p).sum() for p in model.parameters())
        (loss + reg).backward()
        opt.step()
    model.eval()
    return model


def get_or_train_width_model(hidden_dim, activation_cls=nn.ReLU):
    """Lazily load or train and save MLP1Hidden model for a given hidden width."""
    if hidden_dim not in WIDTH_MODEL_PATHS:
        raise ValueError(f"Unsupported hidden_dim={hidden_dim}, expected one of {sorted(WIDTH_MODEL_PATHS.keys())}")
    model_path = WIDTH_MODEL_PATHS[hidden_dim]
    if os.path.exists(model_path):
        print(f"Loading width-model from {model_path}")
        model = MLP1Hidden(hidden_dim=hidden_dim, activation_cls=activation_cls)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model

    print(f"Training width-model (not found at {model_path})...")
    model = MLP1Hidden(hidden_dim=hidden_dim, activation_cls=activation_cls)
    model = train_width_model(model, learning_rate=0.02, n_steps=2500)
    torch.save(model.state_dict(), model_path)
    print(f"Width-model saved to {model_path}")
    model.eval()
    return model


def get_decision_boundary(model, X, h=0.05):
    """Get decision boundary mesh for visualization.

    Uses fine-grained grid (h=0.05) for smooth boundaries.
    """
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


def separator_to_line_on_axes(axes, w, b, color=YELLOW, stroke_width=6):
    """Create a Line mobject for w^T x + b = 0 within current axes ranges."""
    x_min, x_max = float(axes.x_range[0]), float(axes.x_range[1])
    y_min, y_max = float(axes.y_range[0]), float(axes.y_range[1])

    w1, w2 = float(w[0]), float(w[1])
    eps = 1e-8

    if abs(w2) > eps:
        # y = -(w1 x + b) / w2
        x1, x2 = x_min, x_max
        y1 = -(w1 * x1 + b) / w2
        y2 = -(w1 * x2 + b) / w2
    else:
        # Vertical-ish line: x = -b / w1
        x1 = x2 = -b / (w1 if abs(w1) > eps else eps)
        y1, y2 = y_min, y_max

    p1 = axes.c2p(x1, y1)
    p2 = axes.c2p(x2, y2)
    line = Line(p1, p2, color=color, stroke_width=stroke_width)
    return line


# ============================================================================
# Individual Scene Classes
# ============================================================================

class IntroductionScene(InteractiveScene):
    """Introduction scene."""
    def construct(self):
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


class LinearTransformationScene(InteractiveScene):
    """Scene 1: Show linear transformation on 2D plane."""
    def construct(self):
        # Prepare data (keep the same dataset)
        X0, y = make_moons(n_samples=100, noise=0.1, random_state=1)

        # Fit one "best" linear separator: w^T x + b = 0
        w_base, b_base = fit_linear_separator(X0, y)
        errors0 = count_separator_errors(X0, y, w_base, b_base)

        # Title
        title = Text("Linear transforms: what do they do to the dataset?", font_size=48)
        title.to_edge(UP, buff=0.35)
        self.play(Write(title))
        self.wait(0.4)

        # Layout: axes on the left, formulas on the right
        axes = Axes(
            x_range=[-4.0, 4.0, 0.5],
            y_range=[-3.2, 3.2, 0.5],
            width=7.2,
            height=6.0,
            axis_config={"include_tip": True},
        )
        axes.to_edge(LEFT, buff=0.7).shift(0.25 * DOWN)

        panel = Rectangle(width=5.6, height=6.2)
        panel.to_edge(RIGHT, buff=0.7).shift(0.25 * DOWN)
        panel.set_stroke(WHITE, width=2)
        panel.set_fill(BLACK, opacity=0.72)

        panel_title = Text("Formulas", font_size=36)
        panel_title.move_to(panel.get_top() + DOWN * 0.38)
        panel_title.align_to(panel, LEFT).shift(RIGHT * 0.35)

        eq_tex = Tex(r"x' = A x + b", font_size=44)
        eq_tex.next_to(panel_title, DOWN, buff=0.35)
        eq_tex.align_to(panel_title, LEFT)

        A_tex = Tex(r"A = I", font_size=34)
        A_tex.next_to(eq_tex, DOWN, buff=0.35)
        A_tex.align_to(panel_title, LEFT)

        b_tex = Tex(r"b = \begin{bmatrix} 0 \\ 0 \end{bmatrix}", font_size=34)
        b_tex.next_to(A_tex, DOWN, buff=0.25)
        b_tex.align_to(panel_title, LEFT)

        step_label = Text("Original data", font_size=28)
        step_label.next_to(b_tex, DOWN, buff=0.35)
        step_label.align_to(panel_title, LEFT)

        err_panel = Text(f"Linear separator errors: {errors0} / {len(X0)}", font_size=28)
        err_panel.next_to(step_label, DOWN, buff=0.35)
        err_panel.align_to(panel_title, LEFT)

        invariant = Text("Errors stay the same\nunder affine transforms", font_size=26)
        invariant.next_to(err_panel, DOWN, buff=0.35)
        invariant.align_to(panel_title, LEFT)

        # Data + separator (baseline)
        dots = create_data_points(axes, X0, y, radius=0.075)
        sep_line = separator_to_line_on_axes(axes, w_base, b_base, color=YELLOW)

        self.play(FadeIn(axes), FadeIn(panel))
        self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.02))
        self.play(
            ShowCreation(sep_line),
            FadeIn(panel_title),
            FadeIn(eq_tex),
            FadeIn(A_tex),
            FadeIn(b_tex),
            FadeIn(step_label),
            FadeIn(err_panel),
            FadeIn(invariant),
        )
        self.wait(0.8)

        def update_panel(A_str, b_str, label_str, param_font_size=34):
            A_new = Tex(A_str, font_size=param_font_size)
            A_new.move_to(A_tex)
            A_new.align_to(A_tex, LEFT)

            b_new = Tex(b_str, font_size=param_font_size)
            b_new.move_to(b_tex)
            b_new.align_to(b_tex, LEFT)

            label_new = Text(label_str, font_size=28)
            label_new.move_to(step_label)
            label_new.align_to(step_label, LEFT)

            self.play(
                Transform(A_tex, A_new),
                Transform(b_tex, b_new),
                Transform(step_label, label_new),
                run_time=0.7,
            )

        def demo_linear(A, name, param_str, param_font_size=36):
            A = np.array(A, dtype=float)

            # Forward transform x' = A x
            X1 = X0 @ A.T
            w1 = np.linalg.inv(A).T @ w_base
            b1 = b_base
            errors1 = count_separator_errors(X1, y, w1, b1)

            # This should be invariant if we transform separator consistently
            if errors1 != errors0:
                print("Warning: errors changed:", errors0, "->", errors1)

            dots_1 = create_data_points(axes, X1, y, radius=0.075)
            sep_1 = separator_to_line_on_axes(axes, w1, b1, color=YELLOW)

            update_panel(
                param_str,
                r"b = \begin{bmatrix} 0 \\ 0 \end{bmatrix}",
                name,
                param_font_size,
            )
            self.play(
                Transform(dots, dots_1, path_arc=PI / 10),
                Transform(sep_line, sep_1),
                run_time=2.0,
            )
            self.wait(0.5)

            # Return to baseline before the next demo
            dots_back = create_data_points(axes, X0, y, radius=0.075)
            sep_back = separator_to_line_on_axes(axes, w_base, b_base, color=YELLOW)
            self.play(
                Transform(dots, dots_back, path_arc=-PI / 10),
                Transform(sep_line, sep_back),
                run_time=0.6,
            )
            self.wait(0.1)

        def demo_bias(t, name, param_str, param_font_size=34):
            t = np.array(t, dtype=float)

            # Forward transform x' = x + t, separator becomes w^T x' + (b - w^T t) = 0
            X1 = X0 + t
            w1 = w_base
            b1 = b_base - float(w_base @ t)
            errors1 = count_separator_errors(X1, y, w1, b1)

            if errors1 != errors0:
                print("Warning: errors changed:", errors0, "->", errors1)

            dots_1 = create_data_points(axes, X1, y, radius=0.075)
            sep_1 = separator_to_line_on_axes(axes, w1, b1, color=YELLOW)

            update_panel(
                r"A = I",
                param_str,
                name,
                param_font_size,
            )
            self.play(
                Transform(dots, dots_1, path_arc=PI / 10),
                Transform(sep_line, sep_1),
                run_time=2.0,
            )
            self.wait(0.5)

            dots_back = create_data_points(axes, X0, y, radius=0.075)
            sep_back = separator_to_line_on_axes(axes, w_base, b_base, color=YELLOW)
            self.play(
                Transform(dots, dots_back, path_arc=-PI / 10),
                Transform(sep_line, sep_back),
                run_time=0.6,
            )
            self.wait(0.1)

        # 1) Stretch (anisotropic scaling)
        demo_linear(
            [[1.7, 0.0], [0.0, 0.65]],
            "Stretch (anisotropic scaling)",
            r"A = \begin{bmatrix} 1.7 & 0 \\ 0 & 0.65 \end{bmatrix}",
            param_font_size=32,
        )

        # 3) Rotation
        theta = 45 * DEGREES
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
            dtype=float,
        )
        demo_linear(
            R,
            "Rotation",
            r"A = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix},\ \theta=45^\circ",
            param_font_size=26,
        )

        # 4) Bias / translation
        demo_bias(
            [1.1, -0.75],
            "Translation (bias)",
            r"b = \begin{bmatrix} 1.1 \\ -0.75 \end{bmatrix}",
            param_font_size=32,
        )

        # Final takeaway
        takeaway = Text(
            "Moons are NOT linearly separable.\nAffine transforms do not fix separability.",
            font_size=34,
        )
        takeaway.to_edge(DOWN, buff=0.3)
        self.play(Write(takeaway))
        self.wait(1.8)

        self.play(
            FadeOut(takeaway),
            FadeOut(title),
            FadeOut(panel_title),
            FadeOut(eq_tex),
            FadeOut(A_tex),
            FadeOut(b_tex),
            FadeOut(step_label),
            FadeOut(err_panel),
            FadeOut(invariant),
            FadeOut(panel),
            FadeOut(axes),
            FadeOut(dots),
            FadeOut(sep_line),
        )
        self.wait(0.3)


class DimensionalityExpansionScene(InteractiveScene):
    """Scene 2: Show dimensionality expansion."""
    def construct(self):
        # Prepare data
        X, y = make_moons(n_samples=100, noise=0.1, random_state=1)

        title = Text("Dimensionality expansion via a linear map", font_size=52)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.wait(0.4)

        # Left: original 2D space
        axes_2d = Axes(
            x_range=[-2, 3, 0.5],
            y_range=[-2, 2, 0.5],
            width=6.6,
            height=6.0,
            axis_config={"include_tip": True},
        )
        axes_2d.to_edge(LEFT, buff=0.7).shift(0.25 * DOWN)
        dots_2d = create_data_points(axes_2d, X, y, radius=0.075)

        label_2d = Text(r"Input space: $\mathbb{R}^2$", font_size=30)
        label_2d.next_to(axes_2d, UP, buff=0.2)

        # Right: higher-dimensional space (3D)
        axes_3d = ThreeDAxes(
            x_range=[-2, 3, 0.5],
            y_range=[-2, 2, 0.5],
            z_range=[-3, 3, 1],
            width=6.0,
            height=6.0,
            depth=4.2,
        )
        axes_3d.to_edge(RIGHT, buff=0.7).shift(0.25 * DOWN)

        label_3d = Text(r"Output space: $\mathbb{R}^3$", font_size=30)
        label_3d.next_to(axes_3d, UP, buff=0.2)

        # Middle: generic affine form (no parameter values shown)
        arrow = Arrow(axes_2d.get_right(), axes_3d.get_left(), buff=0.25)
        map_tex = Tex(r"x' = A x + b", font_size=44)
        map_tex.next_to(arrow, UP, buff=0.2)

        map_caption = Text("Linear map into a higher-dimensional space", font_size=26)
        map_caption.next_to(map_tex, UP, buff=0.15)

        self.play(
            FadeIn(axes_2d),
            FadeIn(axes_3d),
            FadeIn(label_2d),
            FadeIn(label_3d),
        )
        self.play(LaggedStartMap(FadeIn, dots_2d, lag_ratio=0.02))
        self.play(GrowArrow(arrow), FadeIn(map_tex), FadeIn(map_caption))
        self.wait(0.6)

        # A concrete 2D -> 3D linear embedding (values not shown on screen)
        A = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.85, -0.35],
            ],
            dtype=float,
        )
        b = np.array([0.0, 0.0, 0.0], dtype=float)

        X3 = X @ A.T + b

        # Visualize that the image of R^2 under a linear map sits on a plane in R^3
        u_min, u_max = -2.5, 3.0
        v_min, v_max = -2.0, 2.0

        def lift(u, v):
            p = np.array([u, v], dtype=float)
            q = A @ p + b
            return axes_3d.c2p(float(q[0]), float(q[1]), float(q[2]))

        plane = Polygon(
            lift(u_min, v_min),
            lift(u_max, v_min),
            lift(u_max, v_max),
            lift(u_min, v_max),
            stroke_width=2,
        )
        plane.set_fill(BLUE_E, opacity=0.12)
        plane.set_stroke(BLUE_E, opacity=0.6)

        dots_3d = VGroup()
        for point, label in zip(X3, y):
            dot = Dot(axes_3d.c2p(point[0], point[1], point[2]), radius=0.075)
            dot.set_color(BLUE if label == 0 else RED)
            dots_3d.add(dot)

        group_3d = VGroup(axes_3d, plane, dots_3d)

        self.play(FadeIn(plane))

        # Animate "lifting": keep original dots, move a copy into the 3D embedding
        travel = dots_2d.copy()
        self.add(travel)
        self.play(
            Transform(travel, dots_3d.copy(), path_arc=PI / 6),
            run_time=2.0,
        )
        self.remove(travel)
        self.add(dots_3d)
        self.wait(0.4)

        # Show depth without changing the camera (prevents tilting 2D elements)
        self.play(
            Rotate(group_3d, angle=22 * DEGREES, axis=UP),
            run_time=1.2,
        )
        self.play(
            Rotate(group_3d, angle=-16 * DEGREES, axis=RIGHT),
            run_time=1.2,
        )
        self.wait(0.5)

        takeaway = Text(
            "Even in higher dimensions, this is still linear.\nA linear map cannot fix non-linearity by itself.",
            font_size=30,
        )
        takeaway.to_edge(DOWN, buff=0.35)
        self.play(Write(takeaway))
        self.wait(2.0)

        self.play(
            FadeOut(title),
            FadeOut(label_2d),
            FadeOut(label_3d),
            FadeOut(arrow),
            FadeOut(map_tex),
            FadeOut(map_caption),
            FadeOut(axes_2d),
            FadeOut(dots_2d),
            FadeOut(plane),
            FadeOut(dots_3d),
            FadeOut(axes_3d),
            FadeOut(takeaway),
        )
        self.wait(0.4)


class MLPNonlinearityScene(InteractiveScene):
    """Scene 3: Show MLP with nonlinearity."""
    def construct(self):
        # Prepare data
        X, y = make_moons(n_samples=100, noise=0.1, random_state=1)

        # Lazily load or train models of different widths
        widths = [2, 4, 8, 16]
        models = {w: get_or_train_width_model(hidden_dim=w, activation_cls=nn.ReLU) for w in widths}

        # Title
        title = Text("MLP with Nonlinearity", font_size=60)
        title.to_edge(UP, buff=0.5)
        self.add(title)
        self.wait(1)

        # Formula block (full MLP equations for 1-hidden-layer architecture)
        mlp_formula = Tex(
            r"""
            \begin{aligned}
            h &= \text{ReLU}(W_1 x + b_1) \\
            y &= W_2 h + b_2
            \end{aligned}
            """,
            font_size=34,
        )
        mlp_formula.to_edge(RIGHT).shift(UP * 2.0)
        self.add(mlp_formula)
        self.wait(0.1)

        # Create axes for decision boundary (left side)
        axes = Axes(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            width=6,
            height=6,
            axis_config={"include_tip": True}
        )
        axes.to_edge(LEFT, buff=0.8).shift(0.5 * DOWN)

        # Show MLP architecture (schematic) - right bottom
        network = NeuralNetwork([2, widths[0], 1])
        network.scale(0.6)
        network.to_edge(RIGHT, buff=0.8).shift(DOWN * 2.0)

        # Label for hidden neurons count (below network)
        width_label = Text(f"Hidden neurons: {widths[0]}", font_size=32)
        width_label.next_to(network, DOWN, buff=0.3)

        # Start with width=2
        w0 = widths[0]
        xx, yy, Z = get_decision_boundary(models[w0], X, h=0.08)
        boundary = create_decision_boundary_mobject(axes, xx, yy, Z, opacity=0.35, step=2)
        dots = create_data_points(axes, X, y)

        self.play(FadeIn(axes))
        self.play(FadeIn(boundary))
        self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.02))
        self.play(
            LaggedStartMap(FadeIn, network.layers, lag_ratio=0.03),
            FadeIn(width_label)
        )
        self.play(LaggedStartMap(ShowCreation, network.lines, lag_ratio=0.01, run_time=0.5))
        self.wait(1)

        # Cycle through widths and show how predictions change
        for w in widths[1:]:
            label_new = Text(f"Hidden neurons: {w}", font_size=32)
            label_new.move_to(width_label)
            xx, yy, Z = get_decision_boundary(models[w], X, h=0.08)
            boundary_new = create_decision_boundary_mobject(axes, xx, yy, Z, opacity=0.35, step=2)
            # Update network visualization to show current width
            network_new = NeuralNetwork([2, w, 1])
            network_new.scale(0.6)
            network_new.move_to(network)
            self.play(
                FadeOut(boundary),
                FadeIn(boundary_new),
                Transform(network, network_new),
                Transform(width_label, label_new),
                run_time=1.6,
            )
            boundary = boundary_new
            # Flash highlight to show the update
            self.play(
                FlashAround(width_label, color=YELLOW, time_width=0.8, run_time=0.8),
                FlashAround(network, color=YELLOW, time_width=0.8, run_time=0.8),
            )
            self.wait(0.6)

        text = Text("Wider hidden layers\n→ more flexible decision boundaries", font_size=36)
        text.to_edge(DOWN, buff=0.5)
        self.play(Write(text))
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(network),
            FadeOut(mlp_formula),
            FadeOut(axes),
            FadeOut(boundary),
            FadeOut(dots),
            FadeOut(width_label),
            FadeOut(text)
        )
        self.wait(0.5)


class ComparisonScene(InteractiveScene):
    """Scene 4: Compare MLP with and without nonlinearity."""
    def construct(self):
        # Prepare data
        X, y = make_moons(n_samples=100, noise=0.1, random_state=1)

        # Lazily load or train models
        model_relu = get_or_train_model(activation_cls=nn.ReLU, model_key="relu")
        model_linear = get_or_train_model(activation_cls=nn.Identity, model_key="linear")

        # Title
        title = Text("With vs Without Nonlinearity", font_size=60)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(1)

        # Create two side-by-side axes (smaller size)
        axes_left = Axes(
            x_range=[-2, 3, 0.5],
            y_range=[-2, 2, 0.5],
            width=5,
            height=4,
            axis_config={"include_tip": True}
        )
        axes_left.to_edge(LEFT, buff=1).shift(0.5 * DOWN)

        axes_right = Axes(
            x_range=[-2, 3, 0.5],
            y_range=[-2, 2, 0.5],
            width=5,
            height=4,
            axis_config={"include_tip": True}
        )
        axes_right.to_edge(RIGHT, buff=1).shift(0.5 * DOWN)

        # Labels (smaller font)
        label_relu = Text("With ReLU", font_size=36)
        label_relu.next_to(axes_left, UP, buff=0.2)
        label_linear = Text("Without Nonlinearity", font_size=36)
        label_linear.next_to(axes_right, UP, buff=0.2)

        # Formulas showing full network architecture (smaller font)
        formula_relu = Tex(
            R"""
            \begin{aligned}
            h_1 &= \text{ReLU}(W_1 x + b_1) \\
            h_2 &= \text{ReLU}(W_2 h_1 + b_2) \\
            y &= W_3 h_2 + b_3
            \end{aligned}
            """,
            font_size=22
        )
        formula_relu.next_to(axes_left, DOWN, buff=0.15)

        formula_identity = Tex(
            R"""
            \begin{aligned}
            h_1 &= W_1 x + b_1 \\
            h_2 &= W_2 h_1 + b_2 \\
            y   &= W_3 h_2 + b_3 \\
                &= W_3 W_2 W_1 x + \text{const}
            \end{aligned}
            """,
            font_size=22
        )
        formula_identity.next_to(axes_right, DOWN, buff=0.15)

        # Get decision boundaries (using default fine-grained grid)
        xx_relu, yy_relu, Z_relu = get_decision_boundary(model_relu, X)
        xx_linear, yy_linear, Z_linear = get_decision_boundary(model_linear, X)

        # Create visualizations
        boundary_relu = create_decision_boundary_mobject(axes_left, xx_relu, yy_relu, Z_relu, opacity=0.4)
        boundary_linear = create_decision_boundary_mobject(axes_right, xx_linear, yy_linear, Z_linear, opacity=0.4)

        dots_left = create_data_points(axes_left, X, y)
        dots_right = create_data_points(axes_right, X, y)

        # Show left side
        self.play(FadeIn(axes_left), Write(label_relu))
        self.play(Write(formula_relu))
        self.play(FadeIn(boundary_relu))
        self.play(LaggedStartMap(FadeIn, dots_left, lag_ratio=0.02))
        self.wait(1)

        # Show right side
        self.play(FadeIn(axes_right), Write(label_linear))
        self.play(Write(formula_identity))
        note_identity = Text("No matter how many linear layers you stack,\nyou still get a single linear layer", font_size=22)
        note_identity.next_to(formula_identity, DOWN, buff=0.15)
        self.play(Write(note_identity))
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
            FadeOut(formula_relu),
            FadeOut(formula_identity),
            FadeOut(note_identity),
            FadeOut(insight)
        )
        self.wait(0.5)


class ConclusionScene(InteractiveScene):
    """Conclusion scene."""
    def construct(self):
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


# ============================================================================
# Main Scene (combines all scenes)
# ============================================================================

class MLPVisualization(InteractiveScene):
    """Main scene combining all MLP visualization scenes.

    This scene runs all individual scenes in sequence.
    Each scene can also be rendered independently by calling its class directly.

    Usage:
        # Render all scenes together:
        manimgl 01_mlp_visualization.py MLPVisualization

        # Render individual scenes:
        manimgl 01_mlp_visualization.py IntroductionScene
        manimgl 01_mlp_visualization.py LinearTransformationScene
        manimgl 01_mlp_visualization.py MLPNonlinearityScene
        manimgl 01_mlp_visualization.py ComparisonScene
        etc.
    """
    def construct(self):
        # Create scene instances with shared camera and frame
        scenes = [
            IntroductionScene(),
            LinearTransformationScene(),
            DimensionalityExpansionScene(),
            MLPNonlinearityScene(),
            ComparisonScene(),
            ConclusionScene(),
        ]

        # Share camera, frame, and other scene attributes across all scenes
        for scene in scenes:
            scene.camera = self.camera
            scene.frame = self.frame
            # Copy other important attributes
            if hasattr(self, 'file_writer'):
                scene.file_writer = self.file_writer
            if hasattr(self, 'renderer'):
                scene.renderer = self.renderer
            scene.construct()
