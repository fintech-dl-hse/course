## Reference templates (Manim CE + PyTorch)

Use these snippets as starting points. Adapt paths to the local seminar directory when integrating.

### Torch cache utilities

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def repo_root_from(file: str | Path, *, levels_up: int = 4) -> Path:
    """
    Heuristic: climb `levels_up` directories from the given file.

    Adjust `levels_up` if your script is nested differently.
    """
    p = Path(file).resolve()
    for _ in range(levels_up):
        p = p.parent
    return p


def cache_dir(file: str | Path) -> Path:
    d = repo_root_from(file) / ".cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass(frozen=True)
class TorchArtifact:
    state_dict: dict[str, Any]
    meta: dict[str, Any]


def save_artifact(path: Path, *, state_dict: dict[str, Any], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "meta": meta}, path)


def load_artifact(path: Path) -> TorchArtifact:
    payload = torch.load(path, map_location="cpu")
    return TorchArtifact(state_dict=payload["state_dict"], meta=payload["meta"])
```

### Optimizer trajectories (save runnable data)

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


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
    """
    Returns f(xy) for xy shape [2], a rotated quadratic bowl.

    f(x) = 0.5 * (a * u^2 + b * v^2) where [u, v] = R^T [x, y].
    """
    c = float(torch.cos(torch.tensor(theta)))
    s = float(torch.sin(torch.tensor(theta)))
    r_t = torch.tensor([[c, s], [-s, c]], dtype=torch.float32)

    def f(xy: torch.Tensor) -> torch.Tensor:
        uv = r_t @ xy
        return 0.5 * (a * uv[0] ** 2 + b * uv[1] ** 2)

    return f


def generate_optimizer_trajectory(
    *,
    name: str,
    optim_ctor: Callable[[list[torch.Tensor]], torch.optim.Optimizer],
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: tuple[float, float] = (2.5, 2.0),
    steps: int = 120,
    seed: int = 0,
) -> Trajectory:
    """
    Runs optimizer steps on a 2D objective and returns parameter + loss traces.

    Note: keep this CPU-friendly and deterministic.
    """
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


def save_trajectories(path: str, trajectories: list[Trajectory], *, meta: dict) -> None:
    payload = {
        "meta": meta,
        "trajectories": [
            {"name": t.name, "xy": t.xy.cpu(), "loss": t.loss.cpu()} for t in trajectories
        ],
    }
    torch.save(payload, path)
```

Example usage (build optimizers + save):

```python
import torch

f = make_objective_rotated_quadratic(a=6.0, b=1.0, theta=0.9)

trajs = [
    generate_optimizer_trajectory(
        name="SGD",
        optim_ctor=lambda params: torch.optim.SGD(params, lr=0.08),
        f=f,
        steps=160,
        seed=0,
    ),
    generate_optimizer_trajectory(
        name="SGD+Momentum",
        optim_ctor=lambda params: torch.optim.SGD(params, lr=0.08, momentum=0.9),
        f=f,
        steps=160,
        seed=0,
    ),
    generate_optimizer_trajectory(
        name="Adam",
        optim_ctor=lambda params: torch.optim.Adam(params, lr=0.08),
        f=f,
        steps=160,
        seed=0,
    ),
]

save_trajectories(
    "optimizer_trajectories.pt",
    trajs,
    meta={"objective": "rotated_quadratic", "seed": 0},
)
```

### Manim CE scene pattern: traced optimizer path

```python
from __future__ import annotations

from pathlib import Path

import torch
from manim import *


class OptimizerTrajectoryScene(Scene):
    def construct(self) -> None:
        payload = torch.load(Path("optimizer_trajectories.pt"), map_location="cpu")
        traj = payload["trajectories"][0]
        xy = traj["xy"]  # [T, 2]

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=8,
            y_length=8,
            tips=False,
        )
        self.add(axes)

        dot = Dot(axes.c2p(float(xy[0, 0]), float(xy[0, 1])), radius=0.06)
        trace = TracedPath(dot.get_center, stroke_color=YELLOW, stroke_width=4)
        self.add(trace, dot)

        title = Text(traj["name"]).scale(0.6).to_edge(UP)
        self.add(title)

        for t in range(1, xy.shape[0]):
            dot.move_to(axes.c2p(float(xy[t, 0]), float(xy[t, 1])))
            self.wait(0.01)
```
