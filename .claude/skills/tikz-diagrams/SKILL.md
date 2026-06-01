---
name: tikz-diagrams
description: Use this skill whenever the user wants to create, edit, or compile TikZ diagrams or illustrations. This includes neural network architectures, ML pipeline diagrams, flowcharts, block diagrams, mathematical visualizations, or any request mentioning TikZ, PGF, pgfplots, or LaTeX diagrams. Trigger on phrases like "draw a network", "architecture diagram", "TikZ", "LaTeX figure", "neural net diagram", "model architecture", "pipeline diagram", or any visual/diagrammatic request that would benefit from vector graphics. Even if the user just says "draw me an MLP" or "visualize this architecture", use this skill.
---

# TikZ Diagram Skill

Create publication-quality TikZ diagrams with a focus on neural network and ML architecture visualizations.

**Environment (local macOS / MacTeX or TeX Live):** the compile script auto-detects the LaTeX engine (`pdflatex` by default; falls back to `xelatex`/`lualatex`) and the PDF→PNG converter (`pdftoppm`/`pdftocairo` → Ghostscript `gs` → ImageMagick → `sips`). The full TikZ library set ships with MacTeX (pgfplots, circuitikz, automata, decorations, fit, calc, positioning, arrows.meta, shapes, patterns). **The PDF is the primary deliverable; the PNG is a best-effort preview** — if no converter is available the script still produces the PDF and just warns.

## Workflow

Every TikZ request follows this pipeline:

1. **Write the `.tex` source** in the working directory (e.g. the repo or a temp dir) with a descriptive filename
2. **Compile** using the bundled script: `bash <skill-dir>/scripts/compile_tikz.sh <path>/<name>.tex`
3. **Preview and verify** — read/open the resulting PDF or PNG (e.g. `open <name>.pdf` on macOS, or read the `.png` with the Read tool). Specifically check for:
   - Group box labels (`Encoder`, `Decoder`, etc.) overlapping with arrows, annotations, or adjacent nodes
   - Skip/bypass connections overlapping with nodes or group boxes
   - Annotation text colliding with arrows or other labels
   - Background fill colors being too faint to distinguish from white
   - Diagram being too wide or too tall to read comfortably
   - Arrows entering/leaving the wrong anchor point on a node
   If any of these issues exist, fix the `.tex` and recompile before delivering.
4. **Deliver** — the final `.pdf` (and `.png`/`.tex`) sit next to the source; point the user at those paths.

The compile script handles LaTeX compilation, PNG conversion (default 300 DPI, best-effort), and cleanup of auxiliary files.

## LaTeX Document Structure

Always use the `standalone` document class with a border for clean cropping:

```latex
\documentclass[border=10pt]{standalone}
\usepackage{amsmath}  % ALWAYS include — needed for \boldsymbol, \underbrace, \text{}, etc.
\usepackage{tikz}
% Add additional packages as needed:
% \usepackage{pgfplots}       % for plots
% \usepackage{circuitikz}     % for circuits
\usetikzlibrary{...}          % load only what you need
\begin{document}
\begin{tikzpicture}[...]
  % diagram content
\end{tikzpicture}
\end{document}
```

**Critical**: Always include `\usepackage{amsmath}` in the preamble. Many math commands (`\boldsymbol`, `\underbrace`, `\text{}`, `\operatorname`) require it and will cause "Undefined control sequence" errors without it.

## Neural Network & ML Architecture Patterns

These are the bread and butter of this skill. The key principles for great NN diagrams:

### Color Palette

Use a consistent, professional palette. These work well for papers and slides:

```latex
% Define a cohesive color scheme at the top of every diagram
\definecolor{inputcol}{HTML}{4FC3F7}    % light blue — inputs
\definecolor{hiddencol}{HTML}{AB47BC}   % purple — hidden/processing layers
\definecolor{outputcol}{HTML}{66BB6A}   % green — outputs
\definecolor{attncol}{HTML}{FFA726}     % orange — attention/special
\definecolor{skipcol}{HTML}{EF5350}     % red — skip connections
\definecolor{bgcol}{HTML}{F5F5F5}       % light gray — background boxes
```

These colors are chosen for colorblind-friendliness and good contrast on both white and dark backgrounds.

### Layer Nodes

For neural network layers, use rounded rectangles with minimum dimensions so labels read clearly:

```latex
\tikzset{
  layer/.style={
    draw, rounded corners=3pt, minimum width=2.2cm, minimum height=0.9cm,
    font=\sffamily\small, align=center, line width=0.6pt
  },
  op/.style={
    draw, circle, minimum size=0.7cm, font=\sffamily\small,
    line width=0.6pt, inner sep=1pt
  },
  annot/.style={
    font=\sffamily\scriptsize, text=gray!70!black
  }
}
```

### Connection Arrows

Use `arrows.meta` for clean, modern arrowheads:

```latex
\usetikzlibrary{arrows.meta}
\tikzset{
  conn/.style={-{Stealth[length=5pt, width=4pt]}, line width=0.6pt},
  skip/.style={conn, skipcol, dashed},
  data/.style={conn, gray!60!black}
}
```

### Common Architecture Blocks

**Feedforward / MLP stack** — Vertical stack of layers connected by arrows:
```latex
\node[layer, fill=inputcol!30]  (in)  at (0,0)   {Input};
\node[layer, fill=hiddencol!30] (h1)  at (0,1.5)  {Linear + ReLU};
\node[layer, fill=hiddencol!30] (h2)  at (0,3.0)  {Linear + ReLU};
\node[layer, fill=outputcol!30] (out) at (0,4.5)  {Output};
\draw[conn] (in) -- (h1);
\draw[conn] (h1) -- (h2);
\draw[conn] (h2) -- (out);
```

**Parallel branches** — Side-by-side paths (e.g., dual encoders, siamese nets):
```latex
\node[layer, fill=inputcol!30]  (la) at (-2, 0) {Encoder A};
\node[layer, fill=inputcol!30]  (lb) at ( 2, 0) {Encoder B};
\node[layer, fill=hiddencol!30] (merge) at (0, 1.5) {Concat / Merge};
\draw[conn] (la) -- (merge);
\draw[conn] (lb) -- (merge);
```

**Grouped sub-blocks** — Use the `fit` library with a background box. See the "Group Box Labeling" section below for how to add labels safely:
```latex
\usetikzlibrary{fit, backgrounds}
\begin{scope}[on background layer]
  \node[fit=(h1)(h2), fill=blue!8, rounded corners=6pt,
        inner sep=8pt] (groupbox) {};
\end{scope}
% Place label AFTER the background scope — see Group Box Labeling section
\node[annot, text=blue!50!black, rotate=90, anchor=south] at (groupbox.west) {Encoder};
```

**Skip / residual connections** — These MUST be routed OUTSIDE any enclosing group box. If a skip connection overlaps with interior nodes, the diagram looks messy and is hard to read. The technique: draw the `fit` group node first (give it a name like `groupbox`), then route skip paths relative to its outer edge with clearance:
```latex
% WRONG — hardcoded offset lands inside the box and overlaps blocks:
%   \draw[skip] (h1.east) -- ++(0.8,0) |- (h3.east);

% CORRECT — anchor to the group box edge + clearance so the path
% stays entirely outside all interior nodes:
\coordinate (skipR) at ($(groupbox.east)+(0.5,0)$);
\draw[skip] (h1.east) -| (skipR |- h1.east)
            -- (skipR |- h3.east) -- (h3.east);
```
When there are multiple skip connections side by side, stagger them at increasing offsets (e.g. +0.5, +0.9) so the lines don't overlap each other either.

**Repeat / ×N notation** — Show that a block repeats:
```latex
\node[annot, right=3pt of encoder-box.north east] {$\times N$};
```

### Group Box Labeling (Critical)

The `label={}` option on `fit` nodes is a common source of overlaps. The label position is calculated relative to the box boundary, but it does not account for nearby arrows, annotations, or adjacent boxes — so it frequently collides with other elements.

**NEVER use `label={}` on `fit` nodes.** Instead, place labels manually after the background scope:

```latex
% WRONG — label overlaps with arrows/annotations above the box:
%   \node[fit=(n1)(n2), label={[annot]above left:Encoder}] {};

% CORRECT — rotated side labels, guaranteed not to collide:
\begin{scope}[on background layer]
  \node[fit=(n1)(n2), fill=blue!8, rounded corners=6pt,
        inner sep=8pt] (enc_box) {};
  \node[fit=(n3)(n4), fill=teal!10, rounded corners=6pt,
        inner sep=8pt] (dec_box) {};
\end{scope}
\node[annot, text=blue!50!black, rotate=90, anchor=south] at (enc_box.west) {Encoder};
\node[annot, text=teal!70!black, rotate=90, anchor=south] at (dec_box.west) {Decoder};
```

**Why rotated side labels work best:**
- They occupy the left margin, which is almost always free of arrows (arrows typically flow vertically between nodes)
- They don't compete with annotations that sit to the right of nodes (like "deterministic", "stochastic")
- They don't collide with dashed loss arrows above the top node
- They scale well even when boxes are stacked close together

**If horizontal labels are preferred**, place them with explicit anchor control and verify no collision:
```latex
\node[annot, text=blue!50!black, anchor=north east]
  at (enc_box.north west) [xshift=-4pt] {Encoder};
```

### Group Box Rules

1. **Never share nodes between groups.** If a node (like latent `z`) sits between encoder and decoder, do NOT include it in both `fit` groups — this causes overlapping background boxes. Leave the shared node outside both groups, or include it in only one.

2. **Keep group regions non-overlapping.** If two `fit` boxes overlap, their fills will blend and create visual mud. Add enough vertical spacing (at least 1.5cm gap) between the last node of one group and the first node of the next.

3. **Use distinct, visible fill colors per group.** See the Background Colors section below.

### Background Colors

Fill colors for `fit` group boxes must be clearly distinguishable from white. Very faint fills like `fill=green!4` or `fill=blue!3` are effectively invisible and make the grouping meaningless.

**Minimum fill intensity: `!8` for any color.** Recommended values:

```latex
% Good — clearly visible group backgrounds
fill=blue!8       % encoder groups
fill=teal!10      % decoder groups
fill=orange!8     % attention groups
fill=red!6        % loss/error groups (red is strong, so !6 is enough)
fill=purple!8     % latent space groups

% BAD — indistinguishable from white
% fill=green!4
% fill=blue!3
% fill=gray!4
```

**Avoid pure green for backgrounds.** Light green (`green!N`) tends to look washed out and close to white even at `!8`–`!10`. Use `teal` instead — it is clearly visible at low intensities and provides good contrast with blue encoder boxes.

### Transformer-Specific Patterns

Transformers are the most commonly requested architecture. Key building blocks:

- **Multi-Head Attention**: A box containing Q/K/V projections, attention matrix, and output projection
- **Feed-Forward Network**: Two linear layers with activation in between
- **Layer Norm**: Thin horizontal bars before/after sub-blocks
- **Positional Encoding**: Added to the input embedding as a `+` circle node

Arrange transformer blocks vertically (bottom-to-top data flow) and use the `fit` library to group each encoder/decoder block with a dashed border and a ×N label. Residual / skip connections in transformers must be routed outside the group box boundary — never through the interior blocks (see skip connection pattern above).

### Diagram Size & Layout

These rules prevent the two most common visual issues: overlapping elements and unreadably wide diagrams.

**Width budget**: Keep diagrams between 8–16cm wide for single-column paper figures. If a horizontal pipeline would exceed ~6 nodes in a row, it will be too wide to read comfortably. Use a **zigzag (serpentine) layout** instead: flow left-to-right for one row, then drop down and continue right-to-left on the next row. Connect rows with a vertical elbow using `-|` or `|-` path operators.

```latex
% Row 1: left to right
\node[conv] (c1) at (0,0) {...};
\node[conv, right=of c1] (c2) {...};
\node[pool, right=of c2] (p1) {...};

% Turn-around: drop down, flow right to left
\coordinate (turn) at ($(p1.east)+(0.3,0)$);
\node[conv, below=1.8cm of p1, anchor=east, xshift=0.5cm] (c4) {...};
\draw[conn] (p1.east) -- (turn) -- (turn |- c4.east) -- (c4.east);
```

**Vertical spacing**: Use at least 1.4–1.8cm between vertically stacked layer nodes. For areas with annotations or side elements (like reparameterization trick nodes, epsilon inputs), increase to 1.8–2.2cm to prevent crowding.

**Overlap prevention checklist** — After writing the `.tex`, mentally verify before compiling:
1. Do any skip/bypass connections cross through node interiors? Route them outside the enclosing group box.
2. Do annotation labels collide with arrows or nodes? Shift them with explicit anchors (`above=3pt`, `xshift`).
3. Do parallel paths run at the same x or y coordinate? Stagger them by at least 0.4cm.
4. Do `fit` box labels overlap with arrows or adjacent content? Use rotated side labels instead of `label={}`.
5. Are background fills visible? Check that all fills use at least `!8` intensity.
6. Do adjacent `fit` boxes share any nodes? Separate them to avoid overlapping backgrounds.

**General readability**:
- Use `\sffamily` (sans-serif) for all labels — it reads better in diagrams than serif
- Maintain consistent spacing: 1.4–1.8cm between vertically stacked layers
- Add dimension annotations with `annot` style only when useful context
- For very deep networks, use "$\cdots$" or a break notation rather than drawing every single layer

## Compiling & Delivering

After writing the `.tex` file:

```bash
# Get the directory where this skill lives
SKILL_DIR="<the path where SKILL.md is>"

bash "$SKILL_DIR/scripts/compile_tikz.sh" <path>/<name>.tex
# optional: override DPI -> bash "$SKILL_DIR/scripts/compile_tikz.sh" <path>/<name>.tex 200
# optional: force engine -> TIKZ_ENGINE=xelatex bash "$SKILL_DIR/scripts/compile_tikz.sh" <path>/<name>.tex
```

The `.pdf` (and best-effort `.png`) land next to the source. Inspect the result — `open <path>/<name>.pdf` to view it, or read the `.png` with the Read tool. If something looks off, fix the `.tex` and recompile. Once satisfied, point the user at the output paths (the `.pdf` is the deliverable; `.png` and `.tex` sit alongside it).

## Troubleshooting

If compilation fails, read the last 30 lines of the `.log` file — the error is almost always there. Common issues:
- **Undefined control sequence for `\boldsymbol`, `\underbrace`, `\text{}`**: Add `\usepackage{amsmath}` to the preamble. This is the #1 most common compilation error.
- **Missing library**: Add the missing `\usetikzlibrary{...}` directive
- **Undefined control sequence (other)**: Usually a typo or missing package
- **Dimension too large**: Simplify a path or reduce coordinate values
- **Overfull hbox**: Widen a node or use `text width` to allow line breaks
- **No PNG produced (only PDF)**: No converter found on PATH. The PDF is still valid and is the primary deliverable — PNG is optional.

## Adapting to Other Diagram Types

While NN architectures are the primary focus, the same patterns work for:
- **Flowcharts**: Use `layer` nodes with directional arrows, diamond nodes for decisions
- **System diagrams**: Use `fit` groups for services, `conn` arrows for data flow
- **Math figures**: Switch to pgfplots for function plots, keep TikZ for geometric constructions
- **Timelines**: Horizontal layout with `chains` library

Apply the same color palette, arrow style, and font conventions to keep everything consistent.
