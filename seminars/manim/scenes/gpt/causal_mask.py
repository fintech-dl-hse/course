"""GPT causal (triangular) attention mask scene for seminar 09 (V12).

Сцена показывает, как causal-маска зануляет верхний треугольник матрицы
внимания: позиция ``t`` смотрит только на позиции ``≤ t``. Слева — двунаправ-
ленное внимание (BERT-style): все ячейки активны. Справа — пошаговое
построение causal-варианта (GPT-style):

1. Полная матрица raw-логитов ``Q K^\\top / \\sqrt{d_k}``.
2. Поверх верхнего треугольника появляются метки ``-\\infty`` (красная маска).
3. Row-wise softmax: верхний треугольник «исчезает» (opacity → 0), нижний
   треугольник + диагональ становятся ярко-жёлтыми и суммируются по строкам
   в 1.
4. Финал: чистый нижне-треугольный паттерн.

Под правой матрицей — лента из 4 токенов и стрелки от позиции ``t = 3``
(``"on"``) к позициям ``0..3`` — иллюстрация «attend only to ≤ t».

Финальная подпись: ``causal mask = autoregressive generation precondition``.

Цветовая конвенция совместима с V06 (``scaled_dot_product.py``):
- ``BLUE`` / ``YELLOW`` — активные attention-веса (raw → softmax).
- ``GREY_B`` / ``RED`` — замаскированные ячейки и метки ``-\\infty``.
- ``TEAL`` — выделение текущей позиции токена в ленте.

Сцена использует общие примитивы ``shared.neural`` (``LabeledBox`` для
заголовков панелей, ``arrow_between`` — нет, потому что для стрелок «токен →
позиция» нужна жёсткая горизонтальная привязка; используется локальный
``_horizontal_arrow``, как в ``rnn_forward.py:45-58``). Матрицы строятся
напрямую через ``Square`` + ``VGroup``.
"""
from __future__ import annotations

from typing import Any

from manim import (
    Arrow,
    BLUE,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    GREY_B,
    LEFT,
    Line,
    MathTex,
    RED,
    RIGHT,
    Scene,
    Square,
    TEAL,
    Tex,
    UP,
    VGroup,
    VMobject,
    WHITE,
    YELLOW,
    Write,
)


def _horizontal_arrow(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    """Стрелка, всегда прилипающая к правому/левому краю объектов.

    Локальная копия (см. ``rnn_forward.py``). Нужна, чтобы при близких по
    вертикали объектах (token row под matrix) стрелки не «прилипали» сверху/
    снизу и не пересекали соседние ячейки.
    """
    defaults: dict[str, Any] = {"buff": 0.08, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


class CausalMask(Scene):
    """V12: GPT causal mask vs BERT bidirectional attention."""

    # ---- Layout constants — 720p (-qm) frame ±(7.11, 4.0) ----
    SEQ_LEN = 4                  # 4×4 attention matrix
    CELL = 0.45                  # attention-matrix cell side

    # Title row.
    Y_TITLE = 3.55

    # Sub-titles above each panel.
    Y_SUBTITLE = 2.85

    # Matrix row (vertical center of both matrices).
    Y_MATRIX = 0.80

    # Token strip + arrows (RIGHT panel only).
    Y_TOKENS = -2.55
    Y_ARROW_FAN = -1.85          # mid-height of arrow fan from "on" to ≤ t

    # Caption row.
    Y_CAPTION = -3.65

    # Panel x-centers.
    X_LEFT_PANEL = -3.85
    X_RIGHT_PANEL = 3.10         # pull right panel slightly inward so it stays
                                 # clear of the right frame edge after labels.

    # 4 tokens used in the right panel demonstration.
    TOKENS = ["the", "cat", "sat", "on"]

    # Toy raw scores Q K^T / sqrt(d_k). Used only to give the *full* (left)
    # matrix some non-uniform shading; values are pre-softmax-ish.
    RAW_SCORES = [
        [2.1, 0.9, 0.4, 0.6],
        [0.7, 2.3, 1.1, 0.5],
        [0.5, 0.8, 2.4, 0.9],
        [0.4, 0.6, 1.0, 2.5],
    ]
    # Row-wise softmax of RAW_SCORES, full attention (BERT-style).
    BIDIR_WEIGHTS = [
        [0.59, 0.18, 0.11, 0.13],
        [0.13, 0.65, 0.19, 0.10],
        [0.10, 0.13, 0.66, 0.15],
        [0.09, 0.11, 0.16, 0.64],
    ]
    # Row-wise softmax restricted to lower triangle + diagonal (GPT-style).
    # Each row i: positions j > i get 0; positions j ≤ i renormalize.
    CAUSAL_WEIGHTS = [
        [1.00, 0.00, 0.00, 0.00],
        [0.18, 0.82, 0.00, 0.00],
        [0.13, 0.16, 0.71, 0.00],
        [0.09, 0.11, 0.16, 0.64],
    ]

    # ----------------------------- helpers -----------------------------
    def _matrix_cell_position(self, panel_x: float, i: int, j: int) -> list[float]:
        """Return [x, y, 0] center for cell (row i, col j) of a SEQ_LEN matrix."""
        # Matrix centered horizontally at panel_x and vertically at Y_MATRIX.
        x = panel_x + (j - (self.SEQ_LEN - 1) / 2.0) * self.CELL
        y = self.Y_MATRIX + ((self.SEQ_LEN - 1) / 2.0 - i) * self.CELL
        return [x, y, 0.0]

    def _build_matrix(
        self,
        panel_x: float,
        weights: list[list[float]],
        color: str,
        *,
        min_op: float = 0.10,
        max_op: float = 0.85,
    ) -> list[list[Square]]:
        """Build a SEQ_LEN × SEQ_LEN grid of Squares with fill ∝ weights."""
        rows: list[list[Square]] = []
        for i in range(self.SEQ_LEN):
            row_cells: list[Square] = []
            for j in range(self.SEQ_LEN):
                w = weights[i][j]
                # Map w∈[0,1] → opacity. w == 0 stays nearly invisible.
                op = min_op + (max_op - min_op) * float(w)
                cell = Square(
                    side_length=self.CELL,
                    color=color,
                    stroke_width=1.5,
                ).set_fill(color, opacity=op)
                cell.move_to(self._matrix_cell_position(panel_x, i, j))
                row_cells.append(cell)
            rows.append(row_cells)
        return rows

    def _row_col_labels(
        self, panel_x: float, panel_label: str = "q", col_label: str = "k",
    ) -> tuple[list[MathTex], list[MathTex]]:
        """Build q_i / k_j index labels around a matrix at `panel_x`."""
        row_lbls: list[MathTex] = []
        col_lbls: list[MathTex] = []
        # Row labels to the LEFT of column 0.
        for i in range(self.SEQ_LEN):
            pos = self._matrix_cell_position(panel_x, i, 0)
            lbl = (
                MathTex(rf"{panel_label}_{i}")
                .scale(0.5)
                .move_to([pos[0] - self.CELL / 2.0 - 0.22, pos[1], 0.0])
            )
            row_lbls.append(lbl)
        # Column labels BELOW row SEQ_LEN-1.
        for j in range(self.SEQ_LEN):
            pos = self._matrix_cell_position(panel_x, self.SEQ_LEN - 1, j)
            lbl = (
                MathTex(rf"{col_label}_{j}")
                .scale(0.5)
                .move_to([pos[0], pos[1] - self.CELL / 2.0 - 0.24, 0.0])
            )
            col_lbls.append(lbl)
        return row_lbls, col_lbls

    # ----------------------------- main -----------------------------
    def construct(self) -> None:
        # ===================== Phase 0: title =====================
        title = (
            MathTex(
                r"\text{Causal mask: position } t \text{ attends only to "
                r"positions } \leq t"
            )
            .scale(0.55)
            .move_to([0.0, self.Y_TITLE, 0.0])
        )
        self.play(Write(title), run_time=0.6)
        self.wait(0.2)

        # ===================== Phase 1: panel sub-titles =====================
        left_subtitle = (
            MathTex(r"\text{Bidirectional (BERT)}")
            .scale(0.5)
            .move_to([self.X_LEFT_PANEL, self.Y_SUBTITLE, 0.0])
        )
        right_subtitle = (
            MathTex(r"\text{Causal (GPT)}")
            .scale(0.5)
            .move_to([self.X_RIGHT_PANEL, self.Y_SUBTITLE, 0.0])
        )
        self.play(
            FadeIn(left_subtitle), FadeIn(right_subtitle),
            run_time=0.4,
        )
        self.wait(0.15)

        # ===================== Phase 2: LEFT panel — bidirectional matrix =====================
        # Bidirectional: full softmax weights — every cell is active.
        left_cells = self._build_matrix(
            self.X_LEFT_PANEL, self.BIDIR_WEIGHTS, BLUE,
            min_op=0.18, max_op=0.85,
        )
        left_row_lbls, left_col_lbls = self._row_col_labels(
            self.X_LEFT_PANEL, panel_label="q", col_label="k",
        )

        all_left = [c for row in left_cells for c in row]
        self.play(
            *[FadeIn(c) for c in all_left],
            *[FadeIn(lbl) for lbl in left_row_lbls],
            *[FadeIn(lbl) for lbl in left_col_lbls],
            run_time=0.6,
        )
        self.wait(0.3)

        # ===================== Phase 3: RIGHT panel — raw logits matrix =====================
        # Start with the same full matrix on the right (raw pre-softmax look).
        # We'll then mask the upper triangle and re-softmax.
        right_cells = self._build_matrix(
            self.X_RIGHT_PANEL, self.BIDIR_WEIGHTS, BLUE,
            min_op=0.18, max_op=0.85,
        )
        right_row_lbls, right_col_lbls = self._row_col_labels(
            self.X_RIGHT_PANEL, panel_label="q", col_label="k",
        )

        all_right = [c for row in right_cells for c in row]
        self.play(
            *[FadeIn(c) for c in all_right],
            *[FadeIn(lbl) for lbl in right_row_lbls],
            *[FadeIn(lbl) for lbl in right_col_lbls],
            run_time=0.5,
        )
        self.wait(0.2)

        # ===================== Phase 4: overlay -inf mask on upper triangle =====================
        # For each cell with j > i, recolor to GREY_B and overlay a small "-∞"
        # MathTex centered on the cell. The -∞ symbols sit at scale 0.40 so
        # rendered height ≈ 0.20 (inside readability floor of 0.18).
        mask_anims = []
        inf_labels: list[MathTex] = []
        for i in range(self.SEQ_LEN):
            for j in range(self.SEQ_LEN):
                if j <= i:
                    continue
                cell = right_cells[i][j]
                mask_anims.append(
                    cell.animate.set_fill(GREY_B, opacity=0.55).set_stroke(RED)
                )
                inf = (
                    MathTex(r"-\infty")
                    .scale(0.40)
                    .move_to(cell.get_center())
                    .set_color(RED)
                )
                inf_labels.append(inf)
        self.play(*mask_anims, run_time=0.5)
        self.play(*[FadeIn(lbl) for lbl in inf_labels], run_time=0.4)
        self.wait(0.4)

        # ===================== Phase 5: row-wise softmax → causal weights =====================
        # Apply softmax row by row. Masked (-∞) cells drop to ~0 opacity (and
        # their -∞ label fades out — softmax has now "consumed" them). Active
        # cells in the lower triangle + diagonal go to YELLOW with opacity
        # proportional to CAUSAL_WEIGHTS[i][j] / row_max.
        # Replace the small Q K^T look with a softmax indicator to the right
        # of the right panel? Skip — keep title alone to avoid clutter.
        for i in range(self.SEQ_LEN):
            row_max = max(self.CAUSAL_WEIGHTS[i])
            anims = []
            for j in range(self.SEQ_LEN):
                cell = right_cells[i][j]
                w = self.CAUSAL_WEIGHTS[i][j]
                if w == 0.0:
                    # Masked cell: collapse to near-invisible.
                    anims.append(
                        cell.animate.set_fill(GREY_B, opacity=0.05)
                        .set_stroke(GREY_B, opacity=0.25)
                    )
                else:
                    op = 0.20 + 0.65 * w / row_max
                    anims.append(
                        cell.animate.set_fill(YELLOW, opacity=op)
                        .set_stroke(YELLOW)
                    )
            # Also fade out -∞ labels in this row that belonged to masked cells.
            row_inf_fades = []
            # inf_labels are stored in flat order over (i, j) with j > i.
            # Compute the flat index for cells in row i where j > i.
            base = 0
            for ii in range(i):
                base += (self.SEQ_LEN - 1 - ii)
            n_in_row = self.SEQ_LEN - 1 - i
            for k in range(n_in_row):
                row_inf_fades.append(FadeOut(inf_labels[base + k]))
            self.play(*anims, *row_inf_fades, run_time=0.45)
        self.wait(0.3)

        # ===================== Phase 6: token strip + arrows from t=3 =====================
        # Place 4 tokens directly below the RIGHT panel matrix, evenly under
        # the matrix columns. Highlight position 3 ("on") in TEAL — this is
        # the "current" position. Draw arrows from "on" back to each of
        # tokens 0..3 (≤ t).
        token_mobs: list[Tex] = []
        # Token strip uses *wider* spacing than the matrix columns (cell ≈ 0.45,
        # italic word ≈ 0.6+ units wide → would overlap). Center the strip on
        # the right panel.
        tok_strip_spacing = 0.85
        tok_x_base = (
            self.X_RIGHT_PANEL
            - (self.SEQ_LEN - 1) * tok_strip_spacing / 2.0
        )
        col_xs = [tok_x_base + j * tok_strip_spacing for j in range(self.SEQ_LEN)]
        for j, tok in enumerate(self.TOKENS):
            color = TEAL if j == self.SEQ_LEN - 1 else WHITE
            t = (
                Tex(rf"\textit{{``{tok}''}}")
                .scale(0.55)
                .move_to([col_xs[j], self.Y_TOKENS, 0.0])
                .set_color(color)
            )
            token_mobs.append(t)
        self.play(*[FadeIn(t) for t in token_mobs], run_time=0.4)

        # Arrows from "on" (token 3) back to tokens 0..3 (incl. self-loop as
        # a small upward bump). To keep the lint happy we route each arrow as
        # a curved-ish two-segment path: out from "on" → up to Y_ARROW_FAN →
        # over to target token's x → down. We avoid plain Arrows that would
        # otherwise pierce intermediate token boxes if drawn straight.
        attend_segments: list[VMobject] = []
        src = token_mobs[self.SEQ_LEN - 1]
        for j in range(self.SEQ_LEN):
            tgt = token_mobs[j]
            sx = float(src.get_top()[0])
            sy = float(src.get_top()[1]) + 0.02
            tx = float(tgt.get_top()[0])
            ty = float(tgt.get_top()[1]) + 0.02
            rail_y = self.Y_ARROW_FAN
            # Three-segment L-shape: up from src → over to tx → down to tgt.
            seg_up = Line(
                start=[sx, sy, 0.0],
                end=[sx, rail_y, 0.0],
                color=TEAL,
                stroke_width=2,
            )
            seg_over = Line(
                start=[sx, rail_y, 0.0],
                end=[tx, rail_y, 0.0],
                color=TEAL,
                stroke_width=2,
            )
            seg_down = Arrow(
                start=[tx, rail_y, 0.0],
                end=[tx, ty + 0.02, 0.0],
                buff=0.0,
                stroke_width=2,
                tip_length=0.10,
                color=TEAL,
            )
            # For the self-loop (j == 3) the up/down segments overlap. Make
            # the up segment shorter and skip the over-segment so the arrow
            # is a tiny bump above "on".
            if j == self.SEQ_LEN - 1:
                bump_y = sy + 0.30
                seg_up = Line(
                    start=[sx - 0.05, sy, 0.0],
                    end=[sx - 0.05, bump_y, 0.0],
                    color=TEAL,
                    stroke_width=2,
                )
                seg_over = Line(
                    start=[sx - 0.05, bump_y, 0.0],
                    end=[sx + 0.05, bump_y, 0.0],
                    color=TEAL,
                    stroke_width=2,
                )
                seg_down = Arrow(
                    start=[sx + 0.05, bump_y, 0.0],
                    end=[sx + 0.05, sy + 0.02, 0.0],
                    buff=0.0,
                    stroke_width=2,
                    tip_length=0.08,
                    color=TEAL,
                )
            attend_segments.extend([seg_up, seg_over, seg_down])
        self.play(*[Create(s) for s in attend_segments], run_time=0.8)
        self.wait(0.4)

        # ===================== Phase 7: caption =====================
        caption = (
            MathTex(
                r"\text{causal mask = autoregressive generation precondition}"
            )
            .scale(0.50)
            .move_to([0.0, self.Y_CAPTION, 0.0])
        )
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(1.0)
