"""Embedding-lookup scene for seminar 09 (V04).

Shows the full pipeline:  word -> vocabulary index -> row in E -> x_t

Layout (horizontal):
- Embedding matrix E is at the top-center.
- Below it, the three timesteps are arranged LEFT to RIGHT:
  each column shows  word  ->  id  ->  x_t  vertically,
  and all three columns sit side by side horizontally.

Arrow colors:
- RED:  "forward" arrows  (id -> E row highlight, word -> id)
- BLUE: "return" arrows   (E row -> x_t vector)

For the center column (t=2) the arrows curve to avoid crossing text.
"""
from __future__ import annotations

from manim import (
    BLUE,
    Create,
    CurvedArrow,
    DOWN,
    FadeIn,
    FadeOut,
    MathTex,
    Rectangle,
    RED,
    Scene,
    Square,
    Tex,
    TransformFromCopy,
    VGroup,
    Write,
    Arrow,
)

from shared.neural import TensorColumn

# Semantic arrow colors
COLOR_FORWARD = RED       # word -> id, id -> E row
COLOR_RETURN = "#4488FF"  # E row -> x_t (blue)


class EmbeddingLookup(Scene):
    """word -> vocab index -> E row -> embedding vector x_t."""

    # ---- Vocabulary (shuffled so lookup indices are non-sequential) ----
    VOCAB = ["dog", "end", "cat", "the", "ran", "sat"]
    EMB_DIM = 4  # d columns in E
    CELL = 0.30  # matrix cell size
    PULLED_CELL = 0.27  # TensorColumn cell size

    # ---- Colors ----
    HIGHLIGHT = "#F5C518"
    INDEX_COLOR = RED

    # ---- Layout ----
    Y_TITLE = 3.55

    # Embedding matrix E (top center)
    MATRIX_X = 0.0
    MATRIX_Y = 1.80

    # Horizontal positions for the 3 timesteps (left, center, right)
    STEP_X = [-4.50, 0.0, 4.50]

    # Vertical positions within each step column
    Y_WORD = -0.70    # input word
    Y_INDEX = -1.45   # vocabulary index
    Y_EMB = -2.80     # embedding vector x_t

    # Indices into VOCAB for the 3 timesteps: "the"=3, "cat"=2, "sat"=5 (idx 1="end")
    TOKEN_IDX = [3, 2, 5]

    def construct(self) -> None:
        # ================ Title ================
        eq_top = MathTex(
            r"x_t = E[\text{token}_t]",
            r"\quad",
            r"E \in \mathbb{R}^{V \times d}",
        ).scale(0.58)
        eq_bot = MathTex(
            r"\text{token}_t \;\xrightarrow{\text{vocab}}\; \text{id}_t"
            r"\;\xrightarrow{E[\text{id}_t]}\; x_t \in \mathbb{R}^d",
        ).scale(0.50)
        title = (
            VGroup(eq_top, eq_bot)
            .arrange(DOWN, buff=0.12)
            .move_to([0.0, self.Y_TITLE, 0.0])
        )
        self.play(Write(title))
        self.wait(0.4)

        # ================ Embedding matrix E (top center) ================
        V = len(self.VOCAB)
        d = self.EMB_DIM
        cell = self.CELL

        grid_w = d * cell
        grid_h = V * cell
        gx0 = self.MATRIX_X - grid_w / 2.0
        gy0 = self.MATRIX_Y + grid_h / 2.0

        # Build grid cells row by row
        cells: list[list[Square]] = []
        cells_flat: list[Square] = []
        for r in range(V):
            row_cells: list[Square] = []
            for c in range(d):
                sq = Square(
                    side_length=cell, color=BLUE, stroke_width=2
                ).set_fill(BLUE, opacity=0.18)
                cx = gx0 + c * cell + cell / 2.0
                cy = gy0 - r * cell - cell / 2.0
                sq.move_to([cx, cy, 0.0])
                row_cells.append(sq)
                cells_flat.append(sq)
            cells.append(row_cells)
        grid = VGroup(*cells_flat)

        # Row labels: index + vocab word to the left of each row
        row_labels: list[VGroup] = []
        for r, tok in enumerate(self.VOCAB):
            row_y = gy0 - r * cell - cell / 2.0
            idx_lbl = (
                MathTex(f"{r}:")
                .scale(0.56)
                .set_color(self.INDEX_COLOR)
                .move_to([gx0 - 1.00, row_y, 0.0])
            )
            tok_lbl = (
                Tex(rf"\textit{{{tok}}}")
                .scale(0.65)
                .move_to([gx0 - 0.50, row_y, 0.0])
            )
            pair = VGroup(idx_lbl, tok_lbl)
            row_labels.append(pair)
        row_labels_group = VGroup(*row_labels)

        # Column headers (1..d)
        col_headers: list[MathTex] = []
        for c in range(d):
            hd = MathTex(str(c + 1)).scale(0.56)
            hd.move_to([gx0 + c * cell + cell / 2.0, gy0 + 0.20, 0.0])
            col_headers.append(hd)
        col_headers_group = VGroup(*col_headers)

        # "E" caption to the left
        e_caption = (
            MathTex("E")
            .scale(0.80)
            .move_to([gx0 - 1.45, self.MATRIX_Y, 0.0])
        )

        # Dimensions under the matrix
        dims_label = (
            MathTex(r"V{=}6,\; d{=}4")
            .scale(0.58)
            .move_to([self.MATRIX_X, gy0 - V * cell - 0.22, 0.0])
        )

        self.play(
            FadeIn(e_caption),
            FadeIn(grid),
            FadeIn(row_labels_group),
            FadeIn(col_headers_group),
            FadeIn(dims_label),
            run_time=1.0,
        )
        self.wait(0.4)

        # ================ Step labels (t=1, t=2, t=3) ================
        for step in range(3):
            t_lbl = (
                MathTex(f"t={step + 1}")
                .scale(0.62)
                .move_to([self.STEP_X[step], self.Y_WORD + 0.55, 0.0])
            )
            self.play(FadeIn(t_lbl), run_time=0.25)

        # ================ Per-step lookup animation ================
        prev_row_highlight: Rectangle | None = None
        prev_return_arrow = None

        for step, vocab_idx in enumerate(self.TOKEN_IDX):
            t = step + 1
            token = self.VOCAB[vocab_idx]
            sx = self.STEP_X[step]
            is_center = (step == 1)  # center column needs curved arrows

            # --- 1. Show the input word ---
            word_tex = (
                Tex(rf"\textit{{``{token}''}}")
                .scale(0.70)
                .move_to([sx, self.Y_WORD, 0.0])
            )
            self.play(FadeIn(word_tex), run_time=0.4)

            # --- 2. Show the vocabulary index with arrow word -> index ---
            idx_tex = (
                MathTex(rf"\text{{id}}={vocab_idx}")
                .scale(0.55)
                .set_color(self.INDEX_COLOR)
                .move_to([sx, self.Y_INDEX, 0.0])
            )
            word_to_idx = Arrow(
                start=word_tex.get_bottom(),
                end=idx_tex.get_top(),
                buff=0.10,
                stroke_width=2.5,
                color=COLOR_FORWARD,
                tip_length=0.12,
            )
            self.play(Create(word_to_idx), FadeIn(idx_tex), run_time=0.5)
            self.wait(0.2)

            # --- 3. Highlight the corresponding row in E ---
            row_y = gy0 - vocab_idx * cell - cell / 2.0
            row_box = Rectangle(
                width=grid_w + 0.08,
                height=cell + 0.06,
                color=self.HIGHLIGHT,
                stroke_width=3,
            ).move_to([self.MATRIX_X, row_y, 0.0])

            # Fade out previous row highlight and return arrow
            anims = [Create(row_box)]
            if prev_row_highlight is not None:
                anims.append(FadeOut(prev_row_highlight))
            if prev_return_arrow is not None:
                anims.append(FadeOut(prev_return_arrow))
            self.play(*anims, run_time=0.5)
            if prev_row_highlight is not None:
                self.remove(prev_row_highlight)
            if prev_return_arrow is not None:
                self.remove(prev_return_arrow)

            # --- 4. Arrow from index up to E row (forward: red) ---
            # All arrows to the matrix row are curved to avoid crossing the matrix labels.
            if step == 0:
                # Left column: curve upward-right, bypassing row labels on left of matrix
                idx_to_row = CurvedArrow(
                    start_point=idx_tex.get_right() + [0, 0.1, 0],
                    end_point=row_box.get_left() + [-0.05, 0, 0],
                    angle=0.9,
                    stroke_width=2.5,
                    color=COLOR_FORWARD,
                    tip_length=0.12,
                )
            elif is_center:
                # Center column: curve LEFT to avoid crossing the matrix labels
                idx_to_row = CurvedArrow(
                    start_point=idx_tex.get_left() + [0, 0.1, 0],
                    end_point=row_box.get_left() + [-0.05, 0, 0],
                    angle=-1.2,
                    stroke_width=2.5,
                    color=COLOR_FORWARD,
                    tip_length=0.12,
                )
            else:
                # Right column: curve upward-left, bypassing row labels on right of matrix
                idx_to_row = CurvedArrow(
                    start_point=idx_tex.get_left() + [0, 0.1, 0],
                    end_point=row_box.get_right() + [0.05, 0, 0],
                    angle=-0.9,
                    stroke_width=2.5,
                    color=COLOR_FORWARD,
                    tip_length=0.12,
                )
            self.play(Create(idx_to_row), run_time=0.4)

            # --- 5. Pull out the row into x_t vector ---
            # Ghost cells over the highlighted row
            row_ghost_cells: list[Square] = []
            for c in range(d):
                src = cells[vocab_idx][c]
                ghost = Square(
                    side_length=cell,
                    color=self.HIGHLIGHT,
                    stroke_width=2,
                ).set_fill(self.HIGHLIGHT, opacity=0.55)
                ghost.move_to(src.get_center())
                row_ghost_cells.append(ghost)
            row_ghost = VGroup(*row_ghost_cells)

            # Target: TensorColumn x_t below the index
            # No built-in label — place label separately to avoid centering jitter.
            x_t = TensorColumn(
                dim=d,
                cell_size=self.PULLED_CELL,
                color=BLUE,
                fill_opacity=0.35,
            ).move_to([sx, self.Y_EMB, 0.0])
            x_t_label = (
                MathTex(f"x_{t}")
                .scale(0.70)
                .next_to(x_t, direction=[1.0, 0.0, 0.0], buff=0.12)
            )

            # Build target ghost cells matching x_t's CELLS positions exactly
            target_cells: list[Square] = []
            for c in range(d):
                tc = Square(
                    side_length=self.PULLED_CELL,
                    color=BLUE,
                    stroke_width=2,
                ).set_fill(BLUE, opacity=0.35)
                tc.move_to(x_t.cells[c].get_center())
                target_cells.append(tc)
            target_group = VGroup(*target_cells)

            # Arrow from E row to x_t (return: blue)
            # All arrows from the matrix row are curved to avoid crossing the matrix labels.
            if step == 0:
                # Left column: curve from left edge of row down to x_t
                row_to_x = CurvedArrow(
                    start_point=row_box.get_left() + [-0.05, 0, 0],
                    end_point=x_t.get_top() + [0.05, 0, 0],
                    angle=-0.9,
                    stroke_width=2.5,
                    color=COLOR_RETURN,
                    tip_length=0.13,
                )
            elif is_center:
                # Center column: curve RIGHT to avoid crossing the matrix labels
                row_to_x = CurvedArrow(
                    start_point=row_box.get_right() + [0.05, 0, 0],
                    end_point=[sx + 0.3, self.Y_EMB + 0.45, 0],
                    angle=-1.2,
                    stroke_width=2.5,
                    color=COLOR_RETURN,
                    tip_length=0.13,
                )
            else:
                # Right column: curve from right edge of row down to x_t
                row_to_x = CurvedArrow(
                    start_point=row_box.get_right() + [0.05, 0, 0],
                    end_point=x_t.get_top() + [-0.05, 0, 0],
                    angle=0.9,
                    stroke_width=2.5,
                    color=COLOR_RETURN,
                    tip_length=0.13,
                )

            self.play(FadeIn(row_ghost), run_time=0.3)
            self.play(
                TransformFromCopy(row_ghost, target_group),
                Create(row_to_x),
                run_time=0.7,
            )
            self.play(
                FadeOut(target_group),
                FadeIn(x_t),
                FadeIn(x_t_label),
                FadeOut(row_ghost),
                FadeOut(idx_to_row),
                run_time=0.4,
            )
            self.remove(target_group, row_ghost, idx_to_row)

            prev_row_highlight = row_box
            prev_return_arrow = row_to_x

            self.wait(0.3)

        # ---- Clean up last highlight ----
        tail = []
        if prev_row_highlight is not None:
            tail.append(FadeOut(prev_row_highlight))
        if prev_return_arrow is not None:
            tail.append(FadeOut(prev_return_arrow))
        if tail:
            self.play(*tail, run_time=0.4)
        if prev_row_highlight is not None:
            self.remove(prev_row_highlight)
        if prev_return_arrow is not None:
            self.remove(prev_return_arrow)

        # ================ Final caption ================
        caption = (
            MathTex(
                r"\text{Shared lookup table } E"
                r"\text{ maps every token to a dense vector}"
            )
            .scale(0.48)
            .to_edge(DOWN, buff=0.15)
        )
        self.play(FadeIn(caption))
        self.wait(1.2)
