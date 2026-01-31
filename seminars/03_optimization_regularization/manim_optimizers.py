"""
Manim CE visualizations for PyTorch optimizer comparison (SGD, SGD+momentum, Adam).

Focus:
- Optimizer state and GPU memory
- How moments are computed (m_t, v_t)
- Why moving average and how it works
- Intuition for first vs second moment

Data is generated and cached; the main scene loads from cache.

Render (one main scene, like lth.py):
  ~/miniconda3/envs/manim-ce/bin/manim -qm ./seminars/03_optimization_regularization/manim_optimizers.py OptimizerComparison

Generate cache only (no Manim):
  python manim_optimizers.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Paths and cache
# -----------------------------------------------------------------------------

def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _course_root() -> Path:
    """Course repo root (parent of seminars)."""
    return _script_dir().parent.parent


def cache_dir() -> Path:
    d = _course_root() / ".cache" / "optimizer_viz"
    d.mkdir(parents=True, exist_ok=True)
    return d


# -----------------------------------------------------------------------------
# Optimizer state size (for memory scene)
# -----------------------------------------------------------------------------

def _tiny_mlp() -> nn.Module:
    """Small MLP for deterministic state size demo."""
    return nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )


def optimizer_state_size_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for v in state.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total


def model_params_size_bytes(model: nn.Module) -> int:
    return sum(
        p.numel() * p.element_size()
        for p in model.parameters()
        if p.requires_grad
    )


def init_optimizer_state(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    x = torch.randn(32, 2)
    y = torch.randn(32, 1)
    optimizer.zero_grad()
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    loss.backward()
    optimizer.step()


def get_optimizer_state_sizes(seed: int = 0) -> dict[str, Any]:
    """
    Returns dict: params_kb, sgd_kb, sgd_momentum_kb, adam_kb.
    Uses a tiny MLP so numbers are stable and fast to compute.
    """
    cache_path = cache_dir() / "optimizer_state_sizes.pt"
    meta = {"seed": seed, "version": 1}
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("meta") == meta:
            return payload["data"]

    model_sgd = _tiny_mlp()
    model_sgd_mom = _tiny_mlp()
    model_adam = _tiny_mlp()

    sgd = torch.optim.SGD(model_sgd.parameters(), lr=1e-2)
    sgd_mom = torch.optim.SGD(model_sgd_mom.parameters(), lr=1e-2, momentum=0.9)
    adam = torch.optim.Adam(model_adam.parameters(), lr=1e-3)

    init_optimizer_state(sgd, model_sgd, seed=seed)
    init_optimizer_state(sgd_mom, model_sgd_mom, seed=seed)
    init_optimizer_state(adam, model_adam, seed=seed)

    params_b = model_params_size_bytes(model_sgd)
    data = {
        "params_kb": round(params_b / 1024, 2),
        "sgd_kb": round(optimizer_state_size_bytes(sgd) / 1024, 2),
        "sgd_momentum_kb": round(optimizer_state_size_bytes(sgd_mom) / 1024, 2),
        "adam_kb": round(optimizer_state_size_bytes(adam) / 1024, 2),
    }
    torch.save({"data": data, "meta": meta}, cache_path)
    return data


# -----------------------------------------------------------------------------
# Trajectories (2D objective)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Trajectory:
    name: str
    xy: torch.Tensor  # [T, 2]
    loss: torch.Tensor  # [T]


def make_objective_rotated_quadratic(
    *,
    a: float = 5.0,
    b: float = 1.0,
    theta: float = 0.8,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """f(xy) = 0.5 * (a*u^2 + b*v^2), [u,v] = R^T [x,y]."""
    c = float(torch.cos(torch.tensor(theta)))
    s = float(torch.sin(torch.tensor(theta)))
    r_t = torch.tensor([[c, s], [-s, c]], dtype=torch.float32)

    def f(xy: torch.Tensor) -> torch.Tensor:
        uv = r_t @ xy
        return 0.5 * (a * uv[0] ** 2 + b * uv[1] ** 2)

    return f


def make_objective_with_local_minima(
    *,
    global_min: tuple[float, float] = (0.0, 0.0),
    bump_scale: float = 0.3,
    bump_freq: float = 2.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Quadratic bowl with sinusoidal bumps creating local minima.
    f(xy) = 0.5 * (x^2 + y^2) + bump_scale * (1 - cos(bump_freq * x)) * (1 - cos(bump_freq * y))

    This creates a landscape where SGD can get stuck in local minima,
    but momentum can escape due to accumulated velocity.
    """
    gx, gy = global_min

    def f(xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[0] - gx, xy[1] - gy
        quadratic = 0.5 * (x ** 2 + y ** 2)
        bumps = bump_scale * (1 - torch.cos(bump_freq * x)) * (1 - torch.cos(bump_freq * y))
        return quadratic + bumps

    return f


def generate_trajectory(
    *,
    name: str,
    optim_ctor: Callable[[list[torch.Tensor]], torch.optim.Optimizer],
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: tuple[float, float] = (2.5, 2.0),
    steps: int = 120,
    seed: int = 0,
) -> Trajectory:
    torch.manual_seed(seed)
    xy = torch.tensor(list(x0), dtype=torch.float32, requires_grad=True)
    opt = optim_ctor([xy])
    xs: list[torch.Tensor] = []
    ls: list[torch.Tensor] = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = f(xy)
        loss.backward()
        opt.step()
        xs.append(xy.detach().clone())
        ls.append(loss.detach().clone())
    return Trajectory(name=name, xy=torch.stack(xs, dim=0), loss=torch.stack(ls, dim=0))


# -----------------------------------------------------------------------------
# Adam moments per step (for "how moments are computed" intuition)
# -----------------------------------------------------------------------------

def generate_adam_moments_1d(
    *,
    steps: int = 50,
    seed: int = 0,
    lr: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.999),
) -> dict[str, Any]:
    """
    Single scalar parameter; we record g_t, m_t, v_t each step.
    Uses a simple loss that yields non-trivial gradients.
    """
    cache_path = cache_dir() / "adam_moments_1d.pt"
    meta = {"steps": steps, "seed": seed, "lr": lr, "betas": betas, "version": 1}
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("meta") == meta:
            return payload["data"]

    torch.manual_seed(seed)
    theta = torch.tensor(1.5, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr, betas=betas)

    grads: list[float] = []
    m_list: list[float] = []
    v_list: list[float] = []

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = (theta - 0.2) ** 2 + 0.01 * (theta ** 4)
        loss.backward()
        g = theta.grad.item() if theta.grad is not None else 0.0
        grads.append(g)
        opt.step()
        state = opt.state[theta]
        m_list.append(state["exp_avg"].item())
        v_list.append(state["exp_avg_sq"].item())

    data = {
        "grads": torch.tensor(grads, dtype=torch.float32),
        "m": torch.tensor(m_list, dtype=torch.float32),
        "v": torch.tensor(v_list, dtype=torch.float32),
        "steps": steps,
    }
    torch.save({"data": data, "meta": meta}, cache_path)
    return data


# -----------------------------------------------------------------------------
# Moving average demo (raw + EMA curves)
# -----------------------------------------------------------------------------

def generate_moving_average_demo(
    *,
    length: int = 100,
    betas: tuple[float, ...] = (0.5, 0.9, 0.99),
    seed: int = 0,
) -> dict[str, Any]:
    """Noisy signal + EMA curves for different beta (same as notebook demo)."""
    cache_path = cache_dir() / "moving_average_demo.pt"
    meta = {"length": length, "betas": betas, "seed": seed, "version": 1}
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("meta") == meta:
            return payload["data"]

    torch.manual_seed(seed)
    raw = torch.rand(length) + torch.sqrt(torch.arange(length, dtype=torch.float32)) / 10
    raw[55] += 0.5  # outlier
    raw = raw / 10

    emas = torch.zeros(len(betas), length)
    emas[:, 0] = raw[0]
    for i in range(1, length):
        for j, beta in enumerate(betas):
            emas[j, i] = beta * emas[j, i - 1] + (1 - beta) * raw[i]

    data = {
        "raw": raw,
        "emas": emas,
        "betas": list(betas),
        "length": length,
    }
    torch.save({"data": data, "meta": meta}, cache_path)
    return data


# -----------------------------------------------------------------------------
# Load or generate trajectory comparison
# -----------------------------------------------------------------------------

def get_trajectories(
    *,
    steps: int = 120,
    seed: int = 0,
    x0: tuple[float, float] = (2.5, 2.0),
) -> list[dict[str, Any]]:
    """Load or generate SGD, SGD+momentum, Adam trajectories."""
    cache_path = cache_dir() / "optimizer_trajectories.pt"
    meta = {"steps": steps, "seed": seed, "x0": x0, "version": 1}
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("meta") == meta:
            return payload["trajectories"]

    f = make_objective_rotated_quadratic(a=6.0, b=1.0, theta=0.9)
    trajs = [
        generate_trajectory(
            name="SGD",
            optim_ctor=lambda p: torch.optim.SGD(p, lr=0.01),
            f=f,
            x0=x0,
            steps=steps,
            seed=seed,
        ),
        generate_trajectory(
            name="SGD + momentum",
            optim_ctor=lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
            f=f,
            x0=x0,
            steps=steps,
            seed=seed,
        ),
        generate_trajectory(
            name="Adam",
            optim_ctor=lambda p: torch.optim.Adam(p, lr=0.01),
            f=f,
            x0=x0,
            steps=steps,
            seed=seed,
        ),
    ]
    payload = {
        "meta": meta,
        "trajectories": [
            {"name": t.name, "xy": t.xy.cpu(), "loss": t.loss.cpu()}
            for t in trajs
        ],
    }
    torch.save(payload, cache_path)
    return payload["trajectories"]


def get_trajectories_bumpy(
    *,
    steps: int = 200,
    seed: int = 0,
    x0: tuple[float, float] = (3.0, 2.5),
) -> list[dict[str, Any]]:
    """Load or generate trajectories on bumpy landscape with local minima."""
    cache_path = cache_dir() / "optimizer_trajectories_bumpy.pt"
    meta = {"steps": steps, "seed": seed, "x0": x0, "version": 2}
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("meta") == meta:
            return payload["trajectories"]

    f = make_objective_with_local_minima(bump_scale=0.4, bump_freq=2.5)
    trajs = [
        generate_trajectory(
            name="SGD",
            optim_ctor=lambda p: torch.optim.SGD(p, lr=0.05),
            f=f,
            x0=x0,
            steps=steps,
            seed=seed,
        ),
        generate_trajectory(
            name="SGD + momentum",
            optim_ctor=lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
            f=f,
            x0=x0,
            steps=steps,
            seed=seed,
        ),
    ]
    payload = {
        "meta": meta,
        "trajectories": [
            {"name": t.name, "xy": t.xy.cpu(), "loss": t.loss.cpu()}
            for t in trajs
        ],
    }
    torch.save(payload, cache_path)
    return payload["trajectories"]


# -----------------------------------------------------------------------------
# Main: generate all caches when run as script
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Manim CE scenes (loaded only when run by manim CLI)
# -----------------------------------------------------------------------------

if __name__ != "__main__":
    from manim import *
    import numpy as np

    # -------------------------------------------------------------------------
    # Shared visualization helpers
    # -------------------------------------------------------------------------

    def create_filled_contours(
        axes: Axes,
        loss_fn: Callable[[float, float], float],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        levels: list[float],
        color_low: str,
        color_high: str,
        resolution: int = 50,
    ) -> VGroup:
        """
        Create filled contour visualization (like matplotlib contourf).

        Returns a VGroup of filled polygons representing loss regions.
        """
        x_min, x_max = x_range
        y_min, y_max = y_range

        # Create grid of loss values
        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)
        loss_grid = np.zeros((resolution, resolution))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                loss_grid[j, i] = loss_fn(x, y)

        filled_regions = VGroup()

        # Create filled regions from high to low (so lower values are on top)
        for level_idx in range(len(levels) - 1, -1, -1):
            level = levels[level_idx]
            t_color = level_idx / (len(levels) - 1) if len(levels) > 1 else 0
            color = interpolate_color(color_low, color_high, t_color)

            # Find all grid cells below this level
            for i in range(resolution - 1):
                for j in range(resolution - 1):
                    # Check if any corner is below level
                    corners = [
                        loss_grid[j, i],
                        loss_grid[j, i + 1],
                        loss_grid[j + 1, i],
                        loss_grid[j + 1, i + 1],
                    ]
                    if min(corners) < level:
                        # Create a small filled square
                        x0, x1 = xs[i], xs[i + 1]
                        y0, y1 = ys[j], ys[j + 1]
                        square = Polygon(
                            axes.coords_to_point(x0, y0),
                            axes.coords_to_point(x1, y0),
                            axes.coords_to_point(x1, y1),
                            axes.coords_to_point(x0, y1),
                            fill_color=color,
                            fill_opacity=0.6,
                            stroke_width=0,
                        )
                        filled_regions.add(square)

        return filled_regions

    def create_heatmap_background(
        axes: Axes,
        loss_fn: Callable[[float, float], float],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        color_low: str,
        color_high: str,
        resolution: int = 40,
        max_loss: float | None = None,
    ) -> VGroup:
        """
        Create a heatmap-style background showing loss values.
        """
        x_min, x_max = x_range
        y_min, y_max = y_range

        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)

        # Compute all loss values
        loss_values = []
        for x in xs:
            for y in ys:
                loss_values.append(loss_fn(x, y))

        if max_loss is None:
            max_loss = max(loss_values)
        min_loss = min(loss_values)

        heatmap = VGroup()
        idx = 0
        for i, x in enumerate(xs[:-1]):
            for j, y in enumerate(ys[:-1]):
                loss = loss_fn((x + xs[i + 1]) / 2, (y + ys[j + 1]) / 2)
                t = (loss - min_loss) / (max_loss - min_loss + 1e-8)
                t = min(1.0, max(0.0, t))
                color = interpolate_color(color_low, color_high, t)

                square = Polygon(
                    axes.coords_to_point(x, y),
                    axes.coords_to_point(xs[i + 1], y),
                    axes.coords_to_point(xs[i + 1], ys[j + 1]),
                    axes.coords_to_point(x, ys[j + 1]),
                    fill_color=color,
                    fill_opacity=0.7,
                    stroke_width=0,
                )
                heatmap.add(square)
                idx += 1

        return heatmap

    def draw_trajectory_scene(
        scene: Scene,
        axes: Axes,
        trajs: list[dict],
        colors: list[str],
        start_x: float,
        start_y: float,
        title: str = "Trajectories",
    ) -> tuple[VGroup, VGroup, VGroup]:
        """
        Draw optimizer trajectories on given axes.
        Returns (paths, end_dots, legend_items).
        """
        names = [t["name"] for t in trajs]
        stroke_widths = [2.5 + 0.5 * i for i in range(len(trajs))]
        faded_opacity = 0.25
        final_opacities = [0.5 + 0.5 * i / (len(trajs) - 1) for i in range(len(trajs))]

        # Prepare legend
        legend_items = VGroup()
        for name, color in zip(names, colors):
            line_sample = Line(ORIGIN, RIGHT * 0.4, color=color, stroke_width=3)
            t = Text(name, font_size=22, color=color)
            t.next_to(line_sample, RIGHT, buff=0.1)
            legend_items.add(VGroup(line_sample, t))
        legend_items.arrange(RIGHT, buff=0.8)
        legend_items.to_edge(DOWN, buff=0.5)

        paths = []
        end_dots = []

        for idx, (traj, color) in enumerate(zip(trajs, colors)):
            xy = traj["xy"]

            start_point = axes.coords_to_point(start_x, start_y)
            path_points = [start_point] + [
                axes.coords_to_point(float(xy[i, 0]), float(xy[i, 1]))
                for i in range(xy.shape[0])
            ]

            path = VMobject(color=color, stroke_width=stroke_widths[idx])
            path.set_points_smoothly(path_points)
            paths.append(path)

            end_dot = Dot(
                axes.coords_to_point(float(xy[-1, 0]), float(xy[-1, 1])),
                radius=0.1,
                color=color,
            )
            end_dots.append(end_dot)

            # Fade previous paths
            if idx > 0:
                fade_anims = []
                for prev_path in paths[:-1]:
                    fade_anims.append(prev_path.animate.set_stroke(opacity=faded_opacity))
                for prev_dot in end_dots[:-1]:
                    fade_anims.append(prev_dot.animate.set_opacity(faded_opacity))
                scene.play(*fade_anims, run_time=0.3)

            scene.play(
                Create(path),
                FadeIn(legend_items[idx]),
                run_time=2.5,
                rate_func=linear,
            )
            scene.play(FadeIn(end_dot), run_time=0.3)
            scene.wait(0.3)

        # Restore to final opacities
        restore_anims = []
        for idx, (path, end_dot) in enumerate(zip(paths, end_dots)):
            restore_anims.append(path.animate.set_stroke(opacity=final_opacities[idx]))
            restore_anims.append(end_dot.animate.set_opacity(final_opacities[idx]))
        scene.play(*restore_anims, run_time=0.5)

        return VGroup(*paths), VGroup(*end_dots), legend_items

    # -------------------------------------------------------------------------
    # Main scene
    # -------------------------------------------------------------------------

    class OptimizerComparison(Scene):
        """Main animation scene: SGD, SGD+momentum, Adam — state, moments, trajectories."""

        def construct(self):
            self.scene1_title()
            self.scene2_state_memory()
            self.scene3_moving_average()
            self.scene4_moments_and_intuition()
            self.scene5_trajectory_comparison()
            self.scene6_momentum_local_minima()

        def scene1_title(self):
            """Title card."""
            title = Text(
                "PyTorch Optimizers: SGD, Momentum, Adam",
                font_size=48,
                color=WHITE,
            )
            subtitle = Text(
                "State · Moments · Moving average · Trajectories",
                font_size=28,
                color=GRAY_B,
            ).next_to(title, DOWN, buff=0.5)

            self.play(FadeIn(title, shift=UP * 0.3), run_time=1)
            self.play(FadeIn(subtitle), run_time=0.8)
            self.wait(1.5)
            self.play(FadeOut(title), FadeOut(subtitle), run_time=0.8)
            self.wait(0.3)

        def scene2_state_memory(self):
            """Optimizer state and GPU memory (SGD, SGD+momentum, Adam)."""
            section_title = Text(
                "1. Optimizer state & GPU memory",
                font_size=36,
                color=GRAY_B,
            )
            section_title.to_edge(UP, buff=0.6)
            self.play(FadeIn(section_title), run_time=0.5)

            # Show model parameters box first
            model_box = Rectangle(
                width=2.5,
                height=1.2,
                fill_color=BLUE,
                fill_opacity=0.7,
                stroke_color=WHITE,
                stroke_width=2,
            )
            model_label = Text("Model\nParameters", font_size=24)
            model_label.move_to(model_box)
            model_group = VGroup(model_box, model_label)
            model_group.shift(UP * 2 + LEFT * 4)

            self.play(FadeIn(model_group))
            self.wait(0.5)

            # Memory indicator for model params
            data = get_optimizer_state_sizes()
            params_kb = data["params_kb"]

            mem_text = Text(f"1× ({params_kb:.1f} KB)", font_size=20, color=YELLOW)
            mem_text.next_to(model_box, DOWN, buff=0.15)
            self.play(FadeIn(mem_text))
            self.wait(0.5)

            # Create optimizer state visualizations
            optimizers_info = [
                ("SGD", GRAY, 0, "0× extra"),
                ("SGD + momentum", GREEN, 1, "+1× (velocity)"),
                ("Adam", TEAL, 2, "+2× (m, v)"),
            ]

            opt_groups = VGroup()
            arrows = VGroup()
            start_x = -1.5
            spacing = 3.5

            for i, (name, color, multiplier, extra_label) in enumerate(optimizers_info):
                # Optimizer box
                opt_box = Rectangle(
                    width=2.2,
                    height=0.9,
                    fill_color=color,
                    fill_opacity=0.6,
                    stroke_color=WHITE,
                    stroke_width=2,
                )
                opt_label = Text(name, font_size=22)
                opt_label.move_to(opt_box)
                opt_group = VGroup(opt_box, opt_label)
                opt_group.move_to(RIGHT * (start_x + i * spacing) + DOWN * 0.3)

                # Arrow from model to optimizer
                arrow = Arrow(
                    model_box.get_right() + DOWN * 0.3,
                    opt_box.get_top(),
                    buff=0.15,
                    color=WHITE,
                    stroke_width=2,
                )

                # Extra memory boxes below optimizer
                extra_group = VGroup()
                if multiplier > 0:
                    for j in range(multiplier):
                        extra_box = Rectangle(
                            width=2.0,
                            height=0.6,
                            fill_color=color,
                            fill_opacity=0.4,
                            stroke_color=color,
                            stroke_width=1.5,
                        )
                        extra_box.next_to(
                            opt_box if j == 0 else extra_group[-1],
                            DOWN,
                            buff=0.1,
                        )
                        extra_group.add(extra_box)

                # Memory label
                total_mult = 1 + multiplier
                mem_label = Text(
                    f"{total_mult}× total",
                    font_size=18,
                    color=YELLOW,
                )
                if multiplier > 0:
                    mem_label.next_to(extra_group, DOWN, buff=0.15)
                else:
                    mem_label.next_to(opt_box, DOWN, buff=0.15)

                extra_label_text = Text(extra_label, font_size=16, color=GRAY_B)
                extra_label_text.next_to(mem_label, DOWN, buff=0.08)

                full_group = VGroup(opt_group, extra_group, mem_label, extra_label_text)
                opt_groups.add(full_group)
                arrows.add(arrow)

            # Animate optimizers appearing
            for arrow, opt_group in zip(arrows, opt_groups):
                self.play(
                    GrowArrow(arrow),
                    FadeIn(opt_group),
                    run_time=0.8,
                )
                self.wait(0.3)

            self.wait(1)

            # Summary note
            note = Text(
                "Each moment buffer = same size as model parameters",
                font_size=24,
                color=GRAY_B,
            )
            note.to_edge(DOWN, buff=0.5)
            self.play(FadeIn(note))
            self.wait(2)

            self.play(
                FadeOut(section_title),
                FadeOut(model_group),
                FadeOut(mem_text),
                FadeOut(opt_groups),
                FadeOut(arrows),
                FadeOut(note),
                run_time=0.6,
            )

        def scene3_moving_average(self):
            """Why moving average; how EMA smooths noisy signal."""
            section_title = Text(
                "2. Exponential moving average (EMA)",
                font_size=36,
                color=GRAY_B,
            )
            section_title.to_edge(UP, buff=0.6)
            self.play(FadeIn(section_title), run_time=0.5)

            # Show EMA formula first
            ema_formula = MathTex(
                r"\text{EMA}_t = \beta \cdot \text{EMA}_{t-1} + (1 - \beta) \cdot x_t",
                font_size=40,
            )
            ema_formula.next_to(section_title, DOWN, buff=0.5)
            self.play(Write(ema_formula))
            self.wait(1)

            data = generate_moving_average_demo()
            raw = data["raw"]
            emas = data["emas"]
            betas = data["betas"]

            axes = Axes(
                x_range=[0, len(raw), 20],
                y_range=[float(raw.min()) - 0.02, float(raw.max()) + 0.02, 0.05],
                x_length=9,
                y_length=4,
                tips=False,
                axis_config={"include_numbers": False},
            )
            axes.shift(DOWN * 0.8)
            self.play(FadeIn(axes))

            x_vals = list(range(len(raw)))
            y_vals_raw = [float(raw[i]) for i in range(len(raw))]
            raw_line = axes.plot_line_graph(
                x_values=x_vals,
                y_values=y_vals_raw,
                line_color=GRAY,
                add_vertex_dots=False,
                stroke_width=2,
            )

            # Prepare legend items (position them first, then animate with lines)
            colors = [YELLOW, GREEN, TEAL]
            legend_labels = ["raw signal"] + [f"β = {b}" for b in betas]
            legend_colors = [GRAY] + colors

            legend_items = VGroup()
            for label, color in zip(legend_labels, legend_colors):
                line_sample = Line(ORIGIN, RIGHT * 0.5, color=color, stroke_width=3)
                text = Text(label, font_size=20, color=color)
                text.next_to(line_sample, RIGHT, buff=0.15)
                item = VGroup(line_sample, text)
                legend_items.add(item)

            legend_items.arrange(RIGHT, buff=0.6)
            legend_items.to_edge(DOWN, buff=0.5)

            # Draw raw line with its legend item
            self.play(Create(raw_line), FadeIn(legend_items[0]))
            self.wait(0.5)

            # Draw EMA lines with their legend items
            ema_lines = []
            for j, (beta, color) in enumerate(zip(betas, colors)):
                y_vals = [float(emas[j, i]) for i in range(emas.shape[1])]
                line = axes.plot_line_graph(
                    x_values=x_vals,
                    y_values=y_vals,
                    line_color=color,
                    add_vertex_dots=False,
                    stroke_width=2.5,
                )
                ema_lines.append(line)
                # Legend item index is j+1 (0 is raw signal)
                self.play(Create(line), FadeIn(legend_items[j + 1]), run_time=1.0)
                self.wait(0.2)

            # Note about beta
            note = Text(
                "Larger β → smoother (more history, slower to adapt)",
                font_size=22,
                color=GRAY_B,
            )
            note.next_to(ema_formula, DOWN, buff=0.3)
            self.play(FadeIn(note))
            self.wait(2)

            to_fade = [section_title, ema_formula, axes, raw_line, legend_items, note] + ema_lines
            self.play(*[FadeOut(m) for m in to_fade], run_time=0.6)

        def scene4_moments_and_intuition(self):
            """Combined scene: Adam moments formulas + intuition."""
            section_title = Text(
                "3. Adam: moments & intuition",
                font_size=36,
                color=GRAY_B,
            )
            section_title.to_edge(UP, buff=0.6)
            self.play(FadeIn(section_title), run_time=0.5)

            # Adam update rule - shown first on the left, visible throughout
            update_title = Text("Adam update:", font_size=35, color=GRAY_B)
            update = MathTex(
                r"\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}",
                font_size=40,
            )
            bias_note_m = MathTex(
                r"\hat{m}_t = \frac{m_t}{1 - \beta_1^t}",
                font_size=32,
                color=GRAY_B,
            )

            bias_note_v = MathTex(
                r"\hat{v}_t = \frac{v_t}{1 - \beta_2^t}",
                font_size=32,
                color=GRAY_B,
            )

            update_group = VGroup(update_title, update, bias_note_m, bias_note_v)
            update_group.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
            update_group.to_edge(LEFT, buff=0.5)
            # update_group.shift(DOWN * 0.5)

            # Draw box around update rule
            update_box = SurroundingRectangle(
                update_group,
                color=GRAY,
                buff=0.2,
                stroke_width=1,
            )

            self.play(Write(update), FadeIn(update_title), FadeIn(bias_note_m), FadeIn(bias_note_v))
            self.play(Create(update_box))
            self.wait(0.5)

            # First moment formula and intuition (right side)
            m_formula = MathTex(
                r"m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t",
                font_size=36,
            )
            m_formula.shift(UP * 1.5 + RIGHT * 2)

            m_intuition = VGroup(
                Text("First moment", font_size=26, color=YELLOW),
                Text("= EMA of gradients", font_size=22),
                Text("→ gradient direction", font_size=20, color=GRAY_B),
                Text("→ like momentum", font_size=20, color=GRAY_B),
            )
            m_intuition.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            m_intuition.next_to(m_formula, DOWN, buff=0.35)
            m_intuition.align_to(m_formula, LEFT)

            self.play(Write(m_formula))
            self.wait(0.3)
            self.play(FadeIn(m_intuition))
            self.wait(1)

            # Second moment formula and intuition (below, right side)
            v_formula = MathTex(
                r"v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2",
                font_size=36,
            )
            v_formula.shift(DOWN * 1.2 + RIGHT * 2)

            v_intuition = VGroup(
                Text("Second moment", font_size=26, color=TEAL),
                Text("= EMA of squared gradients", font_size=22),
                Text("→ gradient variance", font_size=20, color=GRAY_B),
                Text("→ adaptive learning rate", font_size=20, color=GRAY_B),
            )
            v_intuition.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            v_intuition.next_to(v_formula, DOWN, buff=0.35)
            v_intuition.align_to(v_formula, LEFT)

            self.play(Write(v_formula))
            self.wait(0.3)
            self.play(FadeIn(v_intuition))
            self.wait(2.5)

            self.play(
                FadeOut(section_title),
                FadeOut(update_group),
                FadeOut(update_box),
                FadeOut(m_formula),
                FadeOut(m_intuition),
                FadeOut(v_formula),
                FadeOut(v_intuition),
                run_time=0.6,
            )

        def _create_trajectory_scene(
            self,
            trajs: list[dict],
            colors: list[str],
            loss_fn: Callable[[float, float], float],
            x_bounds: tuple[float, float],
            y_bounds: tuple[float, float],
            start_xy: tuple[float, float],
            title: str,
            color_low: str,
            color_high: str,
            final_note: str,
        ) -> None:
            """
            Shared helper for trajectory visualization scenes.
            Creates heatmap background, draws trajectories, and cleans up.
            """
            section_title = Text(title, font_size=36, color=GRAY_B)
            section_title.to_edge(UP, buff=0.6)
            self.play(FadeIn(section_title), run_time=0.5)

            x_min, x_max = x_bounds
            y_min, y_max = y_bounds
            start_x, start_y = start_xy

            axes = Axes(
                x_range=[x_min, x_max, 0.5],
                y_range=[y_min, y_max, 0.5],
                x_length=8,
                y_length=6,
                tips=True,
                axis_config={"include_numbers": False, "tip_length": 0.2},
            )
            axes.shift(DOWN * 0.3)

            # Axis labels
            x_label = MathTex(r"\theta_1", font_size=28)
            x_label.next_to(axes.x_axis, RIGHT, buff=0.1)
            y_label = MathTex(r"\theta_2", font_size=28)
            y_label.next_to(axes.y_axis, UP, buff=0.1)

            # Create heatmap background (contourf-like)
            heatmap = create_heatmap_background(
                axes=axes,
                loss_fn=loss_fn,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max),
                color_low=color_low,
                color_high=color_high,
                resolution=35,
            )

            # Show heatmap and axes
            self.play(FadeIn(heatmap), run_time=0.8)
            self.play(FadeIn(axes), FadeIn(x_label), FadeIn(y_label))
            self.wait(0.3)

            # Mark the minimum
            min_dot = Dot(axes.coords_to_point(0, 0), radius=0.1, color=WHITE)
            min_label = Text("minimum", font_size=18, color=WHITE)
            min_label.next_to(min_dot, DOWN, buff=0.1)
            self.play(FadeIn(min_dot), FadeIn(min_label))

            # Mark starting point
            start_dot = Dot(
                axes.coords_to_point(start_x, start_y),
                radius=0.12,
                color=YELLOW,
            )
            start_label = Text("start", font_size=18, color=YELLOW)
            start_label.next_to(start_dot, UP, buff=0.1)
            self.play(FadeIn(start_dot), FadeIn(start_label))
            self.wait(1)

            self.play(FadeOut(min_label), FadeOut(start_label))

            # Draw trajectories using shared helper
            paths, end_dots, legend_items = draw_trajectory_scene(
                scene=self,
                axes=axes,
                trajs=trajs,
                colors=colors,
                start_x=start_x,
                start_y=start_y,
            )
            self.wait(1)

            # Final note
            note = Text(final_note, font_size=22, color=GRAY_B)
            note.next_to(legend_items, UP, buff=0.3)
            self.play(FadeIn(note))
            self.wait(2)

            # Clean up all objects
            all_objects = [
                section_title, axes, x_label, y_label, heatmap,
                min_dot, start_dot, paths, end_dots, legend_items, note,
            ]
            self.play(*[FadeOut(obj) for obj in all_objects], run_time=0.6)

        def scene5_trajectory_comparison(self):
            """Compare SGD, SGD+momentum, Adam on 2D loss with landscape."""
            trajs = get_trajectories(steps=120)

            # Reorder: Adam, SGD+momentum, SGD
            traj_order = [2, 1, 0]
            ordered_trajs = [trajs[i] for i in traj_order]
            colors = [BLUE, GREEN, RED]

            # Loss function for rotated quadratic
            a, b, theta = 6.0, 1.0, 0.9
            c = float(torch.cos(torch.tensor(theta)))
            s = float(torch.sin(torch.tensor(theta)))

            def loss_fn(x: float, y: float) -> float:
                u = c * x + s * y
                v = -s * x + c * y
                return 0.5 * (a * u**2 + b * v**2)

            # Compute bounds from trajectories
            all_xy = torch.cat([t["xy"] for t in trajs], dim=0)
            x_min = float(all_xy[:, 0].min()) - 0.5
            x_max = float(all_xy[:, 0].max()) + 0.5
            y_min = float(all_xy[:, 1].min()) - 0.5
            y_max = float(all_xy[:, 1].max()) + 0.5

            self._create_trajectory_scene(
                trajs=ordered_trajs,
                colors=colors,
                loss_fn=loss_fn,
                x_bounds=(x_min, x_max),
                y_bounds=(y_min, y_max),
                start_xy=(2.5, 2.0),
                title="4. Optimizer trajectories on 2D landscape",
                color_low=BLUE_E,
                color_high=BLUE_A,
                final_note="SGD oscillates; momentum smooths; Adam adapts per-coordinate",
            )

        def scene6_momentum_local_minima(self):
            """Demonstrate momentum escaping local minima on bumpy landscape."""
            trajs = get_trajectories_bumpy(steps=200)
            colors = [RED, GREEN]

            # Loss function with bumps
            bump_scale, bump_freq = 0.4, 2.5

            def loss_fn(x: float, y: float) -> float:
                quadratic = 0.5 * (x**2 + y**2)
                bumps = bump_scale * (1 - math.cos(bump_freq * x)) * (1 - math.cos(bump_freq * y))
                return quadratic + bumps

            # Compute bounds
            all_xy = torch.cat([t["xy"] for t in trajs], dim=0)
            x_min = min(float(all_xy[:, 0].min()), -0.5) - 0.5
            x_max = float(all_xy[:, 0].max()) + 0.5
            y_min = min(float(all_xy[:, 1].min()), -0.5) - 0.5
            y_max = float(all_xy[:, 1].max()) + 0.5

            self._create_trajectory_scene(
                trajs=trajs,
                colors=colors,
                loss_fn=loss_fn,
                x_bounds=(x_min, x_max),
                y_bounds=(y_min, y_max),
                start_xy=(3.0, 2.5),
                title="5. Momentum escapes local minima",
                color_low=PURPLE_E,
                color_high=PURPLE_A,
                final_note="Momentum accumulates velocity → escapes shallow local minima",
            )


if __name__ == "__main__":
    get_optimizer_state_sizes()
    generate_adam_moments_1d()
    generate_moving_average_demo()
    get_trajectories()
    get_trajectories_bumpy()
    print("Cache generated at", cache_dir())
