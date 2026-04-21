# seminars/manim — ManimCE pilot

> Banner / context
>
> - This directory uses **ManimCE** (Manim Community Edition, `manim>=0.18,<0.20`). It is entirely isolated from the 3b1b-manim stack used elsewhere in this repo.
> - **Do NOT touch** `videos/prefill_decode/**` or `media/videos/manim_optimizers/**`. Those are pre-existing 3b1b / scratch artifacts outside this pilot.
> - The directory name `seminars/manim/` intentionally breaks the numbered-seminar convention (`NN_topic/`) because the `shared/` library needs cross-seminar scope. See the last section for the rationale.

## Env setup

This pilot ships its own venv and Makefile. All commands are run from `seminars/manim/`.

```bash
cd seminars/manim
make venv
# Produces:
#   .venv/                  — isolated Python with ManimCE + deps
#   bin/activate_env.sh     — prepends user-local ffmpeg/MacTeX to PATH when missing
```

The `make venv` recipe probes `ffmpeg` and `latex`, and if they are absent from the default `PATH` it falls back to:

- `$HOME/homebrew/bin/ffmpeg` (user-local Homebrew)
- `/usr/local/texlive/2024/bin/universal-darwin/latex` (MacTeX)

If neither the system binary nor the fallback is present, `make venv` aborts with an install hint.

## Add a new scene

Scenes live under `scenes/<topic>/<name>.py`. The Makefile resolves `SCENE=<ClassName>` by searching `scenes/**/<ClassName>.py`, so the **file stem must equal the scene class name**.

```bash
# Create a new scene
mkdir -p scenes/attention
cat > scenes/attention/AttentionDemo.py <<'PY'
from manim import *
from shared.neural import Neuron, LabeledBox, arrow_between

class AttentionDemo(Scene):
    def construct(self) -> None:
        q = Neuron(label="q").shift(LEFT * 2)
        k = Neuron(label="k").shift(RIGHT * 2)
        self.play(Create(q), Create(k))
        self.play(Create(arrow_between(q, k)))
        self.wait(1)
PY

# Render it
make render SCENE=AttentionDemo
# → .out/AttentionDemo.mp4
```

Import at least two helpers from `shared.neural` so the shared library stays exercised across scenes (principle 2 of the plan).

## Frame sampling

`scripts/sample_frames.py` turns a rendered `.mp4` into exactly **two PNGs** for the vision critic:

- `frame_0001.png` — frame at 50 % of the video duration (mid-animation check).
- `frame_0002.png` — frame at `duration − end-offset` (near-final settled state; default offset 0.05 s so ffmpeg fast-seek lands on the last decoded frame).

`ffprobe` reports the duration; `ffmpeg -ss <t> -frames:v 1` writes each frame. Any stale `frame_*.png` in the output directory is wiped first. A companion `frames_index.json` records `source: "midpoint_endpoint"`, duration, offset, and per-frame timestamps.

```bash
# From seminars/manim/ with the venv active:
python scripts/sample_frames.py .out/RNNUnroll.mp4
# → .out/frames/RNNUnroll/frame_0001.png (midpoint)
# → .out/frames/RNNUnroll/frame_0002.png (endpoint)
# → .out/frames/RNNUnroll/frames_index.json
```

Tune `--end-offset` only if your render drops the last few frames; the default is correct for 30 fps ManimCE output.

## Authoring new scenes — `manim-visualizer` subagent

`.claude/agents/manim-visualizer.md` defines a scene-author subagent (Sonnet, tools: Read/Write/Edit/Glob) that writes a new ManimCE `Scene` subclass from a short natural-language brief. Hard contract:

- The target file path looks like `scenes/<topic>/<ClassName>.py`; the produced class name equals the file stem (Makefile `SCENE=<ClassName>` routing).
- At least two primitives are imported from `shared.neural` (e.g. `Neuron`, `LabeledBox`, `arrow_between`).
- No edits outside the target file; no new pip deps; no imports from `videos/prefill_decode/**` or `media/videos/manim_optimizers/**`.

Ralph invokes it by delegating via the `Task` tool with `subagent_type="manim-visualizer"`, passing the brief and target path. Example brief: *"Visualize an LSTM cell showing forget/input/output gates and the cell state c_t across 3 timesteps at seminars/manim/scenes/lstm/LSTMGates.py."* The shipped `scenes/lstm/LSTMGates.py` is the reference output.

## Run the critic loop

The full render → midpoint+endpoint sample → vision critic → fix loop is driven by ralph via the PRD at `prd/manim-rnn.prd.json`. From the repo root:

```bash
# One-shot render/sample/hash (no critic):
make -C seminars/manim render SCENE=RNNUnroll
make -C seminars/manim sample VIDEO=.out/RNNUnroll.mp4

# Full loop, driven by ralph (it invokes the manim-frame-critic subagent):
/oh-my-claudecode:ralph seminars/manim/prd/manim-rnn.prd.json
```

Loop termination rules (enforced by `scripts/render_loop.py`):

- `approved=true` from the critic → exit 0.
- Hash-stuck across iteration N and N−1 with non-empty issues, from N ≥ 2 → exit 3.
- `max_iters=5` reached → exit 4.

## Why `seminars/manim/` breaks the numbered-seminar convention

Elsewhere in this repo, seminars live under `seminars/NN_topic/`. This directory deliberately drops the `NN_` prefix for three reasons:

```text
1. The shared library (shared/neural.py) must be importable from scenes
   belonging to different seminars (09_rnn_transformer, future 10_*, 11_*).
   Pinning it under seminars/09_.../manim/ would force copy-paste across
   seminars or fragile relative imports.

2. The venv, Makefile, and requirements.txt are a single ManimCE toolchain.
   Duplicating them per numbered seminar is wasteful and error-prone.

3. The pilot explicitly tests ManimCE in isolation from the 3b1b stack at
   videos/prefill_decode/ and media/videos/manim_optimizers/. Giving it a
   top-level seminars/manim/ slot reinforces the isolation boundary.
```

If the pilot succeeds, new seminars simply drop their scenes under `seminars/manim/scenes/<topic>/` and reuse the same toolchain.
