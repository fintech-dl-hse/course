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
            optim_ctor=lambda p: torch.optim.SGD(p, lr=0.08),
            f=f,
            x0=x0,
            steps=steps,
            seed=seed,
        ),
        generate_trajectory(
            name="SGD + momentum",
            optim_ctor=lambda p: torch.optim.SGD(p, lr=0.08, momentum=0.9),
            f=f,
            x0=x0,
            steps=steps,
            seed=seed,
        ),
        generate_trajectory(
            name="Adam",
            optim_ctor=lambda p: torch.optim.Adam(p, lr=0.08),
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

    class OptimizerComparison(Scene):
        """Main animation scene: SGD, SGD+momentum, Adam — state, moments, trajectories."""

        def construct(self):
            self.scene1_title()
            self.scene2_state_memory()
            self.scene3_moving_average()
            self.scene4_moments_and_intuition()
            self.scene5_trajectory_comparison()

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
            self.play(Create(raw_line))
            self.wait(0.5)

            colors = [YELLOW, GREEN, TEAL]
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
                self.play(Create(line), run_time=1.0)
                self.wait(0.2)

            # Create proper legend
            legend_items = VGroup()
            legend_labels = ["raw signal"] + [f"β = {b}" for b in betas]
            legend_colors = [GRAY] + colors

            for label, color in zip(legend_labels, legend_colors):
                line_sample = Line(ORIGIN, RIGHT * 0.5, color=color, stroke_width=3)
                text = Text(label, font_size=20, color=color)
                text.next_to(line_sample, RIGHT, buff=0.15)
                item = VGroup(line_sample, text)
                legend_items.add(item)

            legend_items.arrange(RIGHT, buff=0.6)
            legend_items.to_edge(DOWN, buff=0.5)
            self.play(FadeIn(legend_items))

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

            # First moment formula and intuition (left side)
            m_formula = MathTex(
                r"m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t",
                font_size=38,
            )
            m_formula.shift(UP * 1.8 + LEFT * 0.5)

            m_intuition = VGroup(
                Text("First moment", font_size=28, color=YELLOW),
                Text("= EMA of gradients", font_size=24),
                Text("→ gradient direction", font_size=22, color=GRAY_B),
                Text("→ like momentum", font_size=22, color=GRAY_B),
            )
            m_intuition.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            m_intuition.next_to(m_formula, DOWN, buff=0.4)
            m_intuition.align_to(m_formula, LEFT)

            self.play(Write(m_formula))
            self.wait(0.5)
            self.play(FadeIn(m_intuition))
            self.wait(1)

            # Second moment formula and intuition (below)
            v_formula = MathTex(
                r"v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2",
                font_size=38,
            )
            v_formula.shift(DOWN * 0.8 + LEFT * 0.5)

            v_intuition = VGroup(
                Text("Second moment", font_size=28, color=TEAL),
                Text("= EMA of squared gradients", font_size=24),
                Text("→ gradient variance", font_size=22, color=GRAY_B),
                Text("→ adaptive learning rate", font_size=22, color=GRAY_B),
            )
            v_intuition.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            v_intuition.next_to(v_formula, DOWN, buff=0.4)
            v_intuition.align_to(v_formula, LEFT)

            self.play(Write(v_formula))
            self.wait(0.5)
            self.play(FadeIn(v_intuition))
            self.wait(1.5)

            # Move formulas up and show update rule
            self.play(
                m_formula.animate.shift(UP * 0.5),
                m_intuition.animate.shift(UP * 0.5),
                v_formula.animate.shift(UP * 0.5),
                v_intuition.animate.shift(UP * 0.5),
            )

            # Adam update rule
            update_title = Text("Adam update rule:", font_size=26, color=GRAY_B)
            update_title.to_edge(DOWN, buff=1.8)

            update = MathTex(
                r"\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}",
                font_size=40,
            )
            update.next_to(update_title, DOWN, buff=0.3)

            bias_note = Text(
                "(with bias correction: m̂ = m/(1-β₁ᵗ), v̂ = v/(1-β₂ᵗ))",
                font_size=20,
                color=GRAY_B,
            )
            bias_note.next_to(update, DOWN, buff=0.2)

            self.play(FadeIn(update_title), Write(update))
            self.play(FadeIn(bias_note))
            self.wait(2.5)

            self.play(
                FadeOut(section_title),
                FadeOut(m_formula),
                FadeOut(m_intuition),
                FadeOut(v_formula),
                FadeOut(v_intuition),
                FadeOut(update_title),
                FadeOut(update),
                FadeOut(bias_note),
                run_time=0.6,
            )

        def scene5_trajectory_comparison(self):
            """Compare SGD, SGD+momentum, Adam on 2D loss with landscape."""
            section_title = Text(
                "4. Optimizer trajectories on 2D landscape",
                font_size=36,
                color=GRAY_B,
            )
            section_title.to_edge(UP, buff=0.6)
            self.play(FadeIn(section_title), run_time=0.5)

            trajs = get_trajectories(steps=120)
            all_xy = torch.cat([t["xy"] for t in trajs], dim=0)
            x_min, x_max = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
            y_min, y_max = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
            pad = 0.5
            x_range = [x_min - pad, x_max + pad, 0.5]
            y_range = [y_min - pad, y_max + pad, 0.5]

            axes = Axes(
                x_range=x_range,
                y_range=y_range,
                x_length=8,
                y_length=6,
                tips=False,
                axis_config={"include_numbers": False},
            )
            axes.shift(DOWN * 0.3)

            # Parameters for the rotated quadratic (must match get_trajectories)
            a, b, theta = 6.0, 1.0, 0.9
            c = float(torch.cos(torch.tensor(theta)))
            s = float(torch.sin(torch.tensor(theta)))

            # Draw contour ellipses for the rotated quadratic loss
            # Loss = 0.5 * (a*u^2 + b*v^2) where [u,v] = R^T [x,y]
            # Level set: a*u^2 + b*v^2 = 2*level
            # In principal coords: u^2/(2*level/a) + v^2/(2*level/b) = 1
            contour_lines = VGroup()
            loss_levels = [0.5, 1.5, 3.0, 6.0, 12.0, 20.0]
            num_points = 100

            for idx, level in enumerate(loss_levels):
                # Semi-axes in principal coordinates
                semi_a = (2 * level / a) ** 0.5
                semi_b = (2 * level / b) ** 0.5

                # Generate ellipse points in principal coords, then rotate
                angles = torch.linspace(0, 2 * 3.14159, num_points)
                u_pts = semi_a * torch.cos(angles)
                v_pts = semi_b * torch.sin(angles)

                # Rotate back to x,y: [x,y] = R @ [u,v]
                x_pts = c * u_pts - s * v_pts
                y_pts = s * u_pts + c * v_pts

                # Create path
                path_points = [
                    axes.coords_to_point(float(x_pts[i]), float(y_pts[i]))
                    for i in range(num_points)
                ]

                # Color interpolation from dark to light blue
                t_color = idx / (len(loss_levels) - 1) if len(loss_levels) > 1 else 0
                color = interpolate_color(BLUE_E, BLUE_A, t_color)

                contour = VMobject(
                    color=color,
                    stroke_width=1.5,
                    stroke_opacity=0.7,
                )
                contour.set_points_smoothly(path_points + [path_points[0]])
                contour_lines.add(contour)

            # Show landscape first
            self.play(FadeIn(axes))
            self.wait(0.3)

            landscape_label = Text(
                "Loss landscape (contour lines)",
                font_size=22,
                color=GRAY_B,
            )
            landscape_label.next_to(axes, UP, buff=0.2)
            self.play(
                LaggedStart(
                    *[Create(c) for c in contour_lines],
                    lag_ratio=0.1,
                ),
                FadeIn(landscape_label),
                run_time=1.5,
            )

            # Mark the minimum
            min_dot = Dot(
                axes.coords_to_point(0, 0),
                radius=0.08,
                color=WHITE,
            )
            min_label = Text("minimum", font_size=18, color=WHITE)
            min_label.next_to(min_dot, DOWN, buff=0.1)
            self.play(FadeIn(min_dot), FadeIn(min_label))
            self.wait(1)

            # Mark the starting point
            start_x, start_y = 2.5, 2.0
            start_dot = Dot(
                axes.coords_to_point(start_x, start_y),
                radius=0.1,
                color=YELLOW,
            )
            start_label = Text("start", font_size=18, color=YELLOW)
            start_label.next_to(start_dot, UP, buff=0.1)
            self.play(FadeIn(start_dot), FadeIn(start_label))
            self.wait(1)

            # Fade out labels before drawing trajectories
            self.play(
                FadeOut(landscape_label),
                FadeOut(min_label),
                FadeOut(start_label),
            )

            # Now draw optimizer trajectories
            colors = [RED, GREEN, BLUE]
            names = [t["name"] for t in trajs]
            paths = []
            trace_dots = []

            for traj, color, name in zip(trajs, colors, names):
                xy = traj["xy"]
                path_points = [
                    axes.coords_to_point(float(xy[i, 0]), float(xy[i, 1]))
                    for i in range(xy.shape[0])
                ]

                # Create path
                path = VMobject(color=color, stroke_width=3)
                path.set_points_smoothly(path_points)
                paths.append(path)

                # Create moving dot that traces the path
                trace_dot = Dot(
                    axes.coords_to_point(float(xy[0, 0]), float(xy[0, 1])),
                    radius=0.08,
                    color=color,
                )
                trace_dots.append(trace_dot)

                # Animate path creation with dot following
                self.play(
                    Create(path),
                    MoveAlongPath(trace_dot, path),
                    run_time=2.5,
                    rate_func=linear,
                )
                self.wait(0.3)

            # Legend
            legend = VGroup()
            for name, color in zip(names, colors):
                line_sample = Line(ORIGIN, RIGHT * 0.4, color=color, stroke_width=3)
                t = Text(name, font_size=22, color=color)
                t.next_to(line_sample, RIGHT, buff=0.1)
                legend.add(VGroup(line_sample, t))
            legend.arrange(RIGHT, buff=0.8)
            legend.to_edge(DOWN, buff=0.5)
            self.play(FadeIn(legend))
            self.wait(2)

            # Final note
            note = Text(
                "Adam adapts step size per coordinate → faster convergence",
                font_size=22,
                color=GRAY_B,
            )
            note.next_to(legend, UP, buff=0.3)
            self.play(FadeIn(note))
            self.wait(2)


if __name__ == "__main__":
    get_optimizer_state_sizes()
    generate_adam_moments_1d()
    generate_moving_average_demo()
    get_trajectories()
    print("Cache generated at", cache_dir())
