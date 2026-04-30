"""Embedding-lookup scene for seminar 09 (V01 of the curriculum catalog).

Анимирует, как дискретный токен превращается в непрерывный векторный
embedding через look-up по строке матрицы ``E ∈ R^{V×d}``:

1. Сверху — формула ``x_t = E[\\text{token}_t]`` и формальное определение
   через one-hot-вектор.
2. Слева — embedding-матрица ``E`` как сетка ``V × d`` со строковыми
   подписями токенов словаря.
3. Справа сверху — лента входных токенов ``"the" / "cat" / "sat"``.
4. Для каждого шага ``t = 1..3`` подсвечивается текущий токен в ленте,
   подсвечивается соответствующая строка ``E`` и анимируется «вытягивание»
   этой строки в столбец-тензор ``x_t`` (TensorColumn) справа. Каждый
   ``x_t`` рисуется на своей вертикальной полосе справа, чтобы стрелки
   look-up'а не пересекали соседние столбцы. К концу видны все три
   ``x_t`` сразу — становится ясно, что ``E`` одна и та же на всех шагах
   (shared lookup table).

Сцена использует общие примитивы ``shared.neural`` (TensorColumn,
arrow_between) и стилистически совместима с ``RNNForward``: те же ячейки
TensorColumn (``cell_size=0.27``), тот же подход «токены сверху →
тензорный поток ниже».
"""
from __future__ import annotations

from manim import (
    BLUE,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    LEFT,
    MathTex,
    Rectangle,
    RIGHT,
    Scene,
    Square,
    Tex,
    TransformFromCopy,
    UP,
    VGroup,
    Write,
)

from shared.neural import TensorColumn, arrow_between


class EmbeddingLookup(Scene):
    """V01: токен → строка в ``E`` → embedding-столбец ``x_t``."""

    # ---- Layout constants — tuned for 720p (-qm) frame ±(7.11, 4.0) ----
    VOCAB = ["the", "cat", "sat", "dog", "ran", "."]
    EMB_DIM = 4  # d — number of columns in E
    CELL = 0.32  # one cell of the matrix (square)
    PULLED_CELL = 0.27  # TensorColumn cell size on the right side

    # Embedding matrix anchor (left half of frame).
    MATRIX_CENTER_X = -4.55
    MATRIX_CENTER_Y = -0.40

    # Title row.
    Y_TITLE = 3.45

    # Token strip row: a single line of input tokens at the top-right area.
    Y_TOKEN_STRIP = 2.45
    TOKEN_STRIP_CENTER_X = 3.40
    TOKEN_STRIP_DX = 1.50  # horizontal spacing between strip tokens

    # Each pulled column x_t gets its own vertical row on the right side
    # so the lookup-arrow from E never crosses a sibling x_t column.
    # Y-rows for x_1, x_2, x_3 (top → bottom).
    PULLED_X = 4.40
    Y_PULLED = [1.10, -0.20, -1.50]

    # Indices into VOCAB for the 3 timesteps shown ("the", "cat", "sat").
    TOKEN_IDX = [0, 1, 2]

    # Highlight colour for active row / token / pulled column ghost.
    HIGHLIGHT = "#F5C518"

    def construct(self) -> None:
        # ---------------- Title ----------------
        eq_top = MathTex(
            r"x_t = E[\text{token}_t]",
            r"\quad",
            r"E \in \mathbb{R}^{V \times d}",
        ).scale(0.62)
        eq_bot = MathTex(
            r"x_t = E^{\top}\, \mathbb{1}_{\mathrm{id}(t)}",
            r"\quad",
            r"V = 6,\; d = 4",
        ).scale(0.55)
        title = (
            VGroup(eq_top, eq_bot)
            .arrange(DOWN, buff=0.14)
            .move_to([0.0, self.Y_TITLE, 0.0])
        )
        self.play(Write(title))
        self.wait(0.2)

        # ---------------- Embedding matrix E (left) ----------------
        V = len(self.VOCAB)
        d = self.EMB_DIM
        cell = self.CELL

        grid_w = d * cell
        grid_h = V * cell
        gx0 = self.MATRIX_CENTER_X - grid_w / 2.0
        gy0 = self.MATRIX_CENTER_Y + grid_h / 2.0

        # Build cells row-by-row (top → bottom).
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

        # Column index header (1..d) immediately above the grid.
        col_headers: list[MathTex] = []
        for c in range(d):
            hd = MathTex(str(c + 1)).scale(0.5)
            hd.move_to([gx0 + c * cell + cell / 2.0, gy0 + 0.22, 0.0])
            col_headers.append(hd)
        col_headers_group = VGroup(*col_headers)

        # Caption "E" above the column-headers, off to the side so the big
        # italic E doesn't visually collide with the digit row.
        e_caption = (
            MathTex("E")
            .scale(0.85)
            .move_to([gx0 - 0.55, gy0 + 0.30, 0.0])
        )

        # Row labels — vocab tokens to the left of each row. Scale 0.55 to
        # clear the 0.18 height floor for the lowercase italic text.
        row_labels: list[Tex] = []
        for r, tok in enumerate(self.VOCAB):
            lbl = (
                Tex(rf"\textit{{``{tok}''}}")
                .scale(0.55)
                .move_to([gx0 - 0.55, gy0 - r * cell - cell / 2.0, 0.0])
            )
            row_labels.append(lbl)
        row_labels_group = VGroup(*row_labels)

        self.play(
            FadeIn(e_caption),
            FadeIn(grid),
            FadeIn(row_labels_group),
            FadeIn(col_headers_group),
            run_time=0.9,
        )
        self.wait(0.3)

        # ---------------- Token strip (top-right) ----------------
        strip_tokens = [self.VOCAB[i] for i in self.TOKEN_IDX]
        strip_x = [
            self.TOKEN_STRIP_CENTER_X - self.TOKEN_STRIP_DX,
            self.TOKEN_STRIP_CENTER_X,
            self.TOKEN_STRIP_CENTER_X + self.TOKEN_STRIP_DX,
        ]
        strip_mobs: list[Tex] = []
        for k, tok in enumerate(strip_tokens):
            t = (
                Tex(rf"\textit{{``{tok}''}}")
                .scale(0.65)
                .move_to([strip_x[k], self.Y_TOKEN_STRIP, 0.0])
            )
            strip_mobs.append(t)
        strip_group = VGroup(*strip_mobs)

        # A small caption identifying the strip as the input sequence.
        strip_caption = (
            MathTex(r"\text{input tokens}")
            .scale(0.55)
            .next_to(strip_group, LEFT, buff=0.30)
        )

        self.play(FadeIn(strip_caption), FadeIn(strip_group), run_time=0.6)
        self.wait(0.2)

        # ---------------- Per-step lookup animation ----------------
        # Each step t draws its x_t at PULLED_X / Y_PULLED[t-1]. Earlier
        # x_t columns persist so by step 3 the viewer sees x_1, x_2, x_3
        # stacked vertically on the right.
        prev_token_box: Rectangle | None = None
        prev_row_highlight: Rectangle | None = None
        prev_arrow = None

        for step, vocab_idx in enumerate(self.TOKEN_IDX):
            t = step + 1
            x_pos_x = self.PULLED_X
            x_pos_y = self.Y_PULLED[step]

            # --- 1. Highlight current token in the strip ---
            tok_mob = strip_mobs[step]
            tok_box = Rectangle(
                width=tok_mob.width + 0.22,
                height=tok_mob.height + 0.20,
                color=self.HIGHLIGHT,
                stroke_width=3,
            ).move_to(tok_mob.get_center())

            # --- 2. Highlight the corresponding row in E ---
            row_y = gy0 - vocab_idx * cell - cell / 2.0
            row_box = Rectangle(
                width=grid_w + 0.08,
                height=cell + 0.06,
                color=self.HIGHLIGHT,
                stroke_width=3,
            ).move_to([self.MATRIX_CENTER_X, row_y, 0.0])

            anims = [Create(tok_box), Create(row_box)]
            # Drop the previous step's highlights (and its row→x arrow) so
            # only the active step glows. Use self.remove for the stale
            # arrow so the lint also sees it gone.
            if prev_token_box is not None:
                anims.append(FadeOut(prev_token_box))
            if prev_row_highlight is not None:
                anims.append(FadeOut(prev_row_highlight))
            if prev_arrow is not None:
                anims.append(FadeOut(prev_arrow))
            self.play(*anims, run_time=0.6)
            # Belt-and-suspenders: explicitly remove from the scene so the
            # static linter (which patches play() and won't run FadeOut)
            # also sees them gone.
            if prev_token_box is not None:
                self.remove(prev_token_box)
            if prev_row_highlight is not None:
                self.remove(prev_row_highlight)
            if prev_arrow is not None:
                self.remove(prev_arrow)

            # --- 3. Build the pulled column tensor x_t at the right side ---
            x_t = TensorColumn(
                dim=d,
                cell_size=self.PULLED_CELL,
                color=BLUE,
                fill_opacity=0.35,
                label=f"x_{t}",
                label_scale=0.6,
            ).move_to([x_pos_x, x_pos_y, 0.0])

            # Ghost row overlaying the highlighted row in E — visualizes the
            # "row pulled out and rotated 90° into a column" intuition via
            # TransformFromCopy.
            row_copy_cells: list[Square] = []
            for c in range(d):
                src = cells[vocab_idx][c]
                ghost = Square(
                    side_length=cell,
                    color=self.HIGHLIGHT,
                    stroke_width=2,
                ).set_fill(self.HIGHLIGHT, opacity=0.55)
                ghost.move_to(src.get_center())
                row_copy_cells.append(ghost)
            row_ghost = VGroup(*row_copy_cells)

            # Build a target-shaped VGroup of cells positioned where x_t's
            # cells will sit (column orientation), used as the destination
            # for TransformFromCopy.
            target_cells: list[Square] = []
            for c in range(d):
                tc = Square(
                    side_length=self.PULLED_CELL,
                    color=BLUE,
                    stroke_width=2,
                ).set_fill(BLUE, opacity=0.35)
                top_y = x_pos_y + (d - 1) / 2.0 * self.PULLED_CELL
                tc.move_to([x_pos_x, top_y - c * self.PULLED_CELL, 0.0])
                target_cells.append(tc)
            target_group = VGroup(*target_cells)

            # Connecting arrow: from the right edge of the highlighted row
            # to the left edge of the pulled tensor column. Because each
            # x_t lives at its own y, the arrow stays within an empty
            # horizontal corridor — no clip through other columns.
            row_to_x = arrow_between(
                row_box, x_t, buff=0.18, tip_length=0.16, color=self.HIGHLIGHT
            )

            self.play(FadeIn(row_ghost), Create(row_to_x), run_time=0.4)
            self.play(
                TransformFromCopy(row_ghost, target_group),
                run_time=0.7,
            )
            # Replace the temporary target_group with the proper TensorColumn
            # (which carries the label x_t).
            self.play(
                FadeOut(target_group),
                FadeIn(x_t),
                FadeOut(row_ghost),
                run_time=0.4,
            )
            self.remove(target_group, row_ghost)

            prev_token_box = tok_box
            prev_row_highlight = row_box
            prev_arrow = row_to_x

        # Fade out the last lingering highlights so the final frame is clean.
        tail_anims = []
        if prev_token_box is not None:
            tail_anims.append(FadeOut(prev_token_box))
        if prev_row_highlight is not None:
            tail_anims.append(FadeOut(prev_row_highlight))
        if prev_arrow is not None:
            tail_anims.append(FadeOut(prev_arrow))
        if tail_anims:
            self.play(*tail_anims, run_time=0.4)
        if prev_token_box is not None:
            self.remove(prev_token_box)
        if prev_row_highlight is not None:
            self.remove(prev_row_highlight)
        if prev_arrow is not None:
            self.remove(prev_arrow)

        # ---------------- Final caption ----------------
        caption = (
            MathTex(
                r"\text{shared lookup: } E \text{ is unchanged across all } t"
            )
            .scale(0.55)
            .to_edge(DOWN, buff=0.25)
        )
        self.play(FadeIn(caption))
        self.wait(1.0)
