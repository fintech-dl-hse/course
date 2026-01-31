---
name: manim-dl-visualizations
description: Creates deep learning visualizations using Manim Community Edition (manim) with runnable PyTorch data. Includes patterns for training tiny models when needed, caching torch models, and generating/saving optimizer trajectories for side-by-side comparisons. Use when the user mentions manim community edition, manim scenes, deep learning animations, optimization trajectories, or PyTorch-driven visualizations.
---

# Manim DL Visualizations (Community Edition)

Use this skill when creating or updating **Manim Community Edition** visualizations for the DL course repository, especially when the visualization should be backed by **real runnable PyTorch data** (e.g., optimizer trajectories, loss curves, tiny trained models).

## Defaults (do these unless explicitly told otherwise)

- **Manim version**: Use **Manim Community Edition** (`manim`), not ManimGL / 3b1b Manim.
- **Imports**: `from manim import *`
- **Execution**: Prefer short renders for iteration (`-ql`/`-qh`), bump quality only at the end.
- **Data**: Prefer **generated data from code** over hand-authored coordinates.
- **Training**: Only train **tiny models** (seconds to a couple minutes on CPU). Cache results.
- **Determinism**: Set seeds and keep runs reproducible.

## Recommended project layout (non-binding)

- **Scenes**: `seminars/<seminar_dir>/manim_ce/*.py`
- **Rendered media**: `seminars/<seminar_dir>/static/` (if committing videos/images)
- **Caches (not committed)**:
  - `.cache/torch_models/` for `torch.save(...)` model checkpoints
  - `.cache/trajectories/` for optimizer trajectories, loss curves, etc.

If `.cache/` is used and not yet ignored, add it to `.gitignore`.

## Workflow

### 1) Start from a “data first” design

Before writing animation code, implement a **pure function** that returns the data you want to animate:

- **Optimizer trajectories**: 2D/3D points, loss values, step indices
- **Training curves**: loss/accuracy arrays
- **Model behavior**: predictions over a grid, decision boundary samples, etc.

This function must be runnable without Manim (so it can be tested quickly).

### 2) Cache: load-or-generate, then animate

Prefer this pattern:

- Compute a stable cache key (hyperparams + seed + data version).
- If cache exists: load.
- Else: compute/train and save.
- Manim scenes should read cached artifacts and animate them.

Keep cached artifacts in `torch.save`-friendly formats (`.pt` / `.pth`), typically dicts of tensors and metadata.

### 3) Optimizer trajectory comparisons (preferred pattern)

When visualizing optimizer behavior, implement utilities that:

- Build optimizers via constructor functions (e.g., `torch.optim.SGD`, `Adam`, etc.)
- Run a fixed number of steps on a simple objective
- Save per-step trajectories (parameters, gradients optionally, losses)

Use a small 2D objective for pedagogy (quadratic bowl, rotated quadratic, Rosenbrock with care, etc.). Keep the function differentiable and stable.

### 4) Manim scene structure: single main class (like lth.py)

- **One main Scene class per file** whose `construct()` calls section methods in order (e.g. `scene1_title()`, `scene2_...()`, …). Do **not** split into many separate Scene classes when the result is one continuous narrative.
- **Reference**: `seminars/02_activations_initialization_dataloader_trainer/lth.py` → `LotteryTicketHypothesis(Scene)` with `scene1_title()`, `scene2_dense_network()`, etc.
- Each section method: show content, then `FadeOut(...)` of that section’s mobjects before the next (clean transitions).
- Move data generation into helpers; keep scene methods focused on building and animating mobjects.
- Make axes/scales explicit and label them.
- Prefer simple, readable animations: `Create`, `Transform`, `FadeIn`, `FadeOut`, `VMobject.set_points_smoothly(...)` for polyline paths.

```bash
~/miniconda3/envs/manim-ce/bin/manim -qm ./seminars/03_.../manim_optimizers.py OptimizerComparison
```

## Pitfalls and mistakes to avoid

- **Optimizer state**: Read optimizer state (e.g. Adam `exp_avg`, `exp_avg_sq`) **after** `opt.step()`, not before; PyTorch fills state during/after the step → `KeyError` if read before.
- **Axes API (Manim CE)**: Use `axes.coords_to_point(x, y)` for coordinate→point; `axes.plot_line_graph(x_values=..., y_values=...)` takes **x_values** and **y_values** (iterables), not `x_range`.
- **Cache vs Manim import**: If the same file does cache generation when run as script and defines Scene when loaded by Manim, use `if __name__ != "__main__":` before `from manim import *` and scene class(es), so `python file.py` does not require Manim.

## Commands (Manim CE)

Run the main scene (one class per file):

```bash
manim -qm path/to/file.py MainSceneName
```

Faster iteration: `-ql`. Higher quality: `-qh`. Use `--media_dir <path>` to send output to a custom directory (e.g. seminar `static/`).

## Quality checklist

- **Correctness**: data functions run standalone; no silent NaNs.
- **Reproducibility**: set seeds, document hyperparams.
- **Performance**: avoid heavy training during rendering; use cache.
- **Pedagogy**: labels, legends, and short on-screen text; avoid clutter.
- **Files**: no trailing spaces; don’t commit large caches.

## Templates and copy-paste snippets

See `reference.md` for ready-to-use templates:
- cache utilities for torch artifacts
- an optimizer trajectory generator that saves `.pt` files
- a trajectory scene pattern using `TracedPath`

**Do not** add a separate shell script to “render all scenes”; one main Scene class and one `manim ... MainSceneName` command is enough.
