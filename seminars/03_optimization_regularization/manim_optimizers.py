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
            self.scene3_moments_formulas()
            self.scene4_moving_average()
            self.scene5_first_second_moment()
            self.scene6_trajectory_comparison()

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
                font_size=32,
                color=GRAY_B,
            )
            section_title.to_edge(UP, buff=0.4)
            self.play(FadeIn(section_title), run_time=0.5)

            data = get_optimizer_state_sizes()
            labels = ["Model params", "SGD", "SGD + momentum", "Adam"]
            values = [
                data["params_kb"],
                data["sgd_kb"],
                data["sgd_momentum_kb"],
                data["adam_kb"],
            ]
            max_v = max(values)
            bar_width = 0.6
            bar_max_height = 3.0

            bars = VGroup()
            baseline = DOWN * 1.5
            for i, (label, v) in enumerate(zip(labels, values)):
                height = bar_max_height * (v / max_v) if max_v > 0 else 0.1
                bar = Rectangle(
                    width=bar_width,
                    height=height,
                    fill_color=BLUE,
                    fill_opacity=0.8,
                    stroke_color=WHITE,
                )
                bar.move_to(baseline + (i - 1.5) * (bar_width + 0.4) * RIGHT)
                bar.shift(UP * (height / 2))
                txt = Text(f"{label}\n{v:.1f} KB", font_size=24)
                txt.next_to(bar, DOWN, buff=0.2)
                bars.add(VGroup(bar, txt))

            self.play(LaggedStart(*(FadeIn(b) for b in bars), lag_ratio=0.15))
            self.wait(1.5)

            note = Text(
                "SGD: no extra state. Momentum: 1 buffer (velocity). "
                "Adam: 2 buffers (m, v) → ~2× params.",
                font_size=26,
            )
            note.to_edge(DOWN)
            self.play(FadeIn(note))
            self.wait(2)

            self.play(
                FadeOut(section_title),
                FadeOut(bars),
                FadeOut(note),
                run_time=0.5,
            )

        def scene3_moments_formulas(self):
            """How moments are computed (m_t, v_t)."""
            section_title = Text(
                "2. Adam: how moments are computed",
                font_size=32,
                color=GRAY_B,
            )
            section_title.to_edge(UP, buff=0.4)
            self.play(FadeIn(section_title), run_time=0.5)

            m_formula = MathTex(
                r"m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t",
                font_size=44,
            )
            m_label = Text("First moment (gradient EMA)", font_size=28)
            m_label.next_to(m_formula, UP, buff=0.3)
            m_formula.shift(UP * 0.5)

            v_formula = MathTex(
                r"v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2",
                font_size=44,
            )
            v_formula.next_to(m_formula, DOWN, buff=0.8)
            v_label = Text("Second moment (squared gradient EMA)", font_size=28)
            v_label.next_to(v_formula, UP, buff=0.2)

            self.play(Write(m_label), Write(m_formula))
            self.wait(1.5)
            self.play(Write(v_label), Write(v_formula))
            self.wait(2)

            self.play(
                FadeOut(section_title),
                FadeOut(m_label),
                FadeOut(m_formula),
                FadeOut(v_label),
                FadeOut(v_formula),
                run_time=0.5,
            )

        def scene4_moving_average(self):
            """Why moving average; how EMA smooths noisy signal."""
            section_title = Text(
                "3. Exponential moving average",
                font_size=32,
                color=GRAY_B,
            )
            section_title.to_edge(UP, buff=0.4)
            self.play(FadeIn(section_title), run_time=0.5)

            data = generate_moving_average_demo()
            raw = data["raw"]
            emas = data["emas"]
            betas = data["betas"]

            axes = Axes(
                x_range=[0, len(raw), 20],
                y_range=[float(raw.min()) - 0.02, float(raw.max()) + 0.02, 0.02],
                x_length=10,
                y_length=5,
                tips=False,
            )
            axes.center()
            axes.shift(DOWN * 0.5)
            self.add(axes)

            x_vals = list(range(len(raw)))
            y_vals_raw = [float(raw[i]) for i in range(len(raw))]
            raw_line = axes.plot_line_graph(
                x_values=x_vals,
                y_values=y_vals_raw,
                line_color=GRAY,
                add_vertex_dots=False,
            )
            self.play(Create(raw_line))
            raw_label = Text("raw", font_size=24).set_color(GRAY).to_corner(UR)
            self.add(raw_label)
            self.wait(1)

            colors = [YELLOW, GREEN, TEAL]
            ema_lines = []
            for j, (beta, color) in enumerate(zip(betas, colors)):
                y_vals = [float(emas[j, i]) for i in range(emas.shape[1])]
                line = axes.plot_line_graph(
                    x_values=x_vals,
                    y_values=y_vals,
                    line_color=color,
                    add_vertex_dots=False,
                )
                ema_lines.append(line)
                self.play(Create(line), run_time=1.2)
                self.wait(0.3)
            legend = Text(
                f"β = {betas[0]}, {betas[1]}, {betas[2]}  (larger β → smoother)",
                font_size=26,
            )
            legend.to_edge(DOWN)
            self.play(FadeIn(legend))
            self.wait(2)

            to_fade = [section_title, axes, raw_line, raw_label, legend] + ema_lines
            self.play(*[FadeOut(m) for m in to_fade], run_time=0.5)

        def scene5_first_second_moment(self):
            """Intuition: first moment = direction; second = scale/variance."""
            section_title = Text(
                "4. First vs second moment intuition",
                font_size=32,
                color=GRAY_B,
            )
            section_title.to_edge(UP, buff=0.4)
            self.play(FadeIn(section_title), run_time=0.5)

            first = VGroup(
                Text("First moment m_t", font_size=32).set_color(YELLOW),
                Text("≈ average gradient direction", font_size=28),
                Text("→ where to step (like momentum)", font_size=26),
            )
            first.arrange(DOWN, aligned_edge=LEFT).shift(LEFT * 2.8 + UP * 0.2)
            second = VGroup(
                Text("Second moment v_t", font_size=32).set_color(TEAL),
                Text("≈ average squared gradient", font_size=28),
                Text("→ scale / variance per coordinate", font_size=26),
                Text("→ larger v → smaller step (adaptive lr)", font_size=26),
            )
            second.arrange(DOWN, aligned_edge=LEFT).shift(RIGHT * 2.8 + UP * 0.2)

            self.play(FadeIn(first))
            self.wait(1.5)
            self.play(FadeIn(second))
            self.wait(1.5)

            update = MathTex(
                r"\theta_{t+1} = \theta_t - \alpha \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}",
                font_size=36,
            )
            update.to_edge(DOWN)
            self.play(Write(update))
            self.wait(2)

            self.play(
                FadeOut(section_title),
                FadeOut(first),
                FadeOut(second),
                FadeOut(update),
                run_time=0.5,
            )

        def scene6_trajectory_comparison(self):
            """Compare SGD, SGD+momentum, Adam on 2D loss."""
            section_title = Text(
                "5. Optimizer comparison: same 2D objective",
                font_size=32,
                color=GRAY_B,
            )
            section_title.to_edge(UP, buff=0.4)
            self.play(FadeIn(section_title), run_time=0.5)

            trajs = get_trajectories(steps=120)
            all_xy = torch.cat([t["xy"] for t in trajs], dim=0)
            x_min, x_max = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
            y_min, y_max = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
            pad = 0.3
            x_range = [x_min - pad, x_max + pad, (x_max - x_min + 2 * pad) / 4]
            y_range = [y_min - pad, y_max + pad, (y_max - y_min + 2 * pad) / 4]

            axes = Axes(
                x_range=x_range,
                y_range=y_range,
                x_length=10,
                y_length=10,
                tips=False,
            )
            axes.center().shift(DOWN * 0.3)
            self.add(axes)

            colors = [RED, GREEN, BLUE]
            names = [t["name"] for t in trajs]
            paths_and_dots = []
            for traj, color in zip(trajs, colors):
                xy = traj["xy"]
                path_points = [
                    axes.coords_to_point(float(xy[i, 0]), float(xy[i, 1]))
                    for i in range(xy.shape[0])
                ]
                path = VMobject(color=color, stroke_width=3)
                path.set_points_smoothly(path_points)
                dot = Dot(
                    axes.coords_to_point(float(xy[0, 0]), float(xy[0, 1])),
                    radius=0.06,
                    color=color,
                )
                paths_and_dots.extend([path, dot])
                self.play(Create(path), FadeIn(dot), run_time=1.5)
                self.wait(0.2)

            legend = VGroup()
            for name, color in zip(names, colors):
                t = Text(name, font_size=24).set_color(color)
                legend.add(t)
            legend.arrange(RIGHT, buff=0.5).to_edge(DOWN)
            self.play(FadeIn(legend))
            self.wait(2)


if __name__ == "__main__":
    get_optimizer_state_sizes()
    generate_adam_moments_1d()
    generate_moving_average_demo()
    get_trajectories()
    print("Cache generated at", cache_dir())
