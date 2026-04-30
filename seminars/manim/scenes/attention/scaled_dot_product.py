"""Scaled dot-product attention scene for seminar 09 (V06 of the curriculum catalog).

Современная, упрощённая, параллелизуемая форма content-based attention из
V05 (BahdanauAttention). Раскладывает механизм на 4 шага:

1. Сверху — формулы:
   ``Q = X W_Q,\\ K = X W_K,\\ V = X W_V``
   ``\\mathrm{Attn}(Q, K, V) = \\mathrm{softmax}(Q K^\\top / \\sqrt{d_k}) V``.
2. Лента из 3 input-токенов (``the / cat / sat``) и под ней ряд эмбеддингов
   ``X`` — 3 TensorColumn (dim=3) одного цвета.
3. Три проекции одновременно: через цветные коробки ``W_Q / W_K / W_V``
   эмбеддинги превращаются в три параллельных ряда тензоров — ``Q`` (cyan),
   ``K`` (orange), ``V`` (green).
4. Attention-матрица 3×3 в центре кадра:
   - сначала появляются raw-скоры ``Q K^\\top / \\sqrt{d_k}``;
   - row-softmax: каждая строка независимо нормализуется, opacity ячеек
     отражает вес.
5. Выход: для строки 1 берётся softmax-строка и свёрнута с V → ``Z_1``.
   Затем quickly показываются ``Z_2`` и ``Z_3``.
6. Финальная подпись: ``parallelizable: all queries computed at once``.

Использует общие примитивы ``shared.neural`` (TensorColumn, LabeledBox,
arrow_between) и стилистически совместима с ``BahdanauAttention`` и
``RNNForward``: те же ячейки TensorColumn, та же палитра
(BLUE = X, TEAL = Q, ORANGE = K, GREEN = V, PURPLE = Z output).

Цветовая конвенция: K — orange (как ``W_{ih}``: input-side), V — green
(как ``W_{hh}``: hidden-side), Q — cyan/teal (новая роль: query).
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
    GREEN,
    LEFT,
    MathTex,
    ORANGE,
    PURPLE,
    Rectangle,
    RIGHT,
    Scene,
    Square,
    TEAL,
    Tex,
    TransformFromCopy,
    UP,
    VGroup,
    VMobject,
    WHITE,
    YELLOW,
    Write,
)

from shared.neural import LabeledBox, TensorColumn, arrow_between


def _horizontal_arrow(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    """Стрелка, всегда прилипающая к правому/левому краю объектов.

    Локальная копия (см. примечание в брифе V06: общий хелпер
    откладывается на отдельный рефактор).
    """
    defaults: dict[str, Any] = {"buff": 0.08, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


class ScaledDotProductAttention(Scene):
    """V06: X → (Q, K, V) → softmax(Q K^T / sqrt(d_k)) → Z = A V."""

    # ---- Layout constants — 720p (-qm) frame ±(7.11, 4.0) ----
    TENSOR_DIM = 3            # short tensors so 3 rows + matrix all fit
    X_CELL = 0.24             # input embedding cell size
    QKV_CELL = 0.22           # Q/K/V cell size (tiny — 3 rows in upper area)
    OUT_CELL = 0.24           # output Z cell size

    # Token positions (3 tokens, centered)
    TOKENS = ["the", "cat", "sat"]
    X_TOK_BASE = -1.50        # leftmost token x
    X_TOK_SPACING = 1.50

    # Row anchors (top→bottom).
    Y_TITLE = 3.35
    Y_TOKEN = 2.55
    Y_X = 1.95                 # X embeddings row
    Y_PROJ = 0.95              # W_Q / W_K / W_V boxes
    Y_QKV = -0.10              # Q / K / V tensor rows (compact, 3-up)

    # Phase-2 (after fade) anchors for the attention-matrix view.
    Y_MATRIX = 0.55            # center of A grid
    MATRIX_CELL = 0.42         # attention-matrix cell side
    X_MATRIX_CENTER = 0.0      # centered horizontally

    # V row (kept on screen for the Z = A V step) and Z output row.
    Y_V_BOTTOM = -1.55         # V row in phase 2
    Y_Z_BOTTOM = -2.85         # Z output row
    X_V_BASE = -3.30           # leftmost V column in phase 2
    X_V_SPACING = 1.10
    X_Z_BASE = 1.55            # leftmost Z column (right side, well-clear of V)
    X_Z_SPACING = 1.10

    # Toy attention scores Q K^T / sqrt(d_k). One dominant cell per row,
    # but row 1 has cross-attention (token 1 attends to token 3 strongest)
    # to make the mechanism visually non-trivial.
    RAW_SCORES = [
        [0.5, 0.8, 2.4],
        [0.3, 2.6, 0.7],
        [0.4, 0.6, 2.5],
    ]
    # Precomputed row-softmax (sum to 1 per row).
    ATTN_WEIGHTS = [
        [0.10, 0.14, 0.76],
        [0.07, 0.85, 0.08],
        [0.10, 0.12, 0.78],
    ]

    def construct(self) -> None:
        # ============== Phase 0: title equations ==============
        eq_proj = MathTex(
            r"Q = X W_Q,\quad K = X W_K,\quad V = X W_V",
        ).scale(0.55)
        eq_attn = MathTex(
            r"\mathrm{Attn}(Q, K, V) = \mathrm{softmax}",
            r"\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V",
        ).scale(0.55)
        title = (
            VGroup(eq_proj, eq_attn)
            .arrange(DOWN, buff=0.14)
            .move_to([0.0, self.Y_TITLE, 0.0])
        )
        self.play(Write(title))
        self.wait(0.2)

        # ============== Phase 1: token strip + X embeddings ==============
        token_mobs: list[Tex] = []
        x_columns: list[TensorColumn] = []
        for j, tok in enumerate(self.TOKENS):
            x_pos = self.X_TOK_BASE + j * self.X_TOK_SPACING
            tok_mob = (
                Tex(rf"\textit{{``{tok}''}}")
                .scale(0.55)
                .move_to([x_pos, self.Y_TOKEN, 0.0])
            )
            token_mobs.append(tok_mob)
            x_col = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.X_CELL,
                color=BLUE,
                fill_opacity=0.30,
            ).move_to([x_pos, self.Y_X, 0.0])
            x_columns.append(x_col)

        x_label = (
            MathTex(r"X")
            .scale(0.55)
            .next_to(x_columns[-1], RIGHT, buff=0.20)
        )
        self.play(*[FadeIn(t) for t in token_mobs], run_time=0.4)
        self.play(*[FadeIn(c) for c in x_columns], FadeIn(x_label), run_time=0.5)
        self.wait(0.2)

        # ============== Phase 2: project X → Q, K, V ==============
        # Three projection boxes side-by-side along the same y row, but each
        # box is offset horizontally from its corresponding x column by a
        # fixed delta so that the box for X·W_Q is to the LEFT of the box for
        # X·W_K, etc. — too crowded. Instead: stack the 3 boxes vertically?
        # No — we have only ~1.0 unit of vertical room between Y_X and Y_QKV.
        #
        # Final layout: keep the Q/K/V rows side-by-side as 3 horizontal
        # GROUPS at the same y, separated by ~3.5 units. Each group has its
        # own (W_proj box, 3 tensor columns). The X embeddings fork into all
        # three groups via arrows.
        #
        # Group centers:
        x_group_q = -4.40
        x_group_k = 0.00
        x_group_v = 4.40
        group_spread = 0.60     # internal column spacing within a Q/K/V group

        def _build_qkv_group(
            x_center: float, color: str, label_str: str,
        ) -> tuple[LabeledBox, list[TensorColumn], MathTex]:
            box = LabeledBox(
                label=f"W_{{{label_str}}}",
                width=0.85,
                height=0.42,
                label_scale=0.50,
                color=color,
            ).move_to([x_center, self.Y_PROJ, 0.0])
            cols: list[TensorColumn] = []
            for j in range(3):
                cx = x_center + (j - 1) * group_spread
                col = TensorColumn(
                    dim=self.TENSOR_DIM,
                    cell_size=self.QKV_CELL,
                    color=color,
                    fill_opacity=0.40,
                ).move_to([cx, self.Y_QKV, 0.0])
                cols.append(col)
            # Group label (Q / K / V) sits to the RIGHT of the rightmost col.
            grp_lbl = (
                MathTex(label_str)
                .scale(0.55)
                .next_to(cols[-1], RIGHT, buff=0.18)
            )
            return box, cols, grp_lbl

        w_q_box, q_cols, q_label = _build_qkv_group(x_group_q, TEAL, "Q")
        w_k_box, k_cols, k_label = _build_qkv_group(x_group_k, ORANGE, "K")
        w_v_box, v_cols, v_label = _build_qkv_group(x_group_v, GREEN, "V")

        # Arrows from X mid-column to each W_* box. To avoid clipping the
        # X tensor columns themselves, we anchor each arrow at the BOTTOM
        # of the centroid of the X row. Use a small dummy reference: the
        # central x column. Three forks, one to each W_* box.
        x_centroid = x_columns[1]
        proj_arrows = [
            arrow_between(
                x_centroid, w_q_box, buff=0.10, tip_length=0.13,
                stroke_width=2.5, color=TEAL,
            ),
            arrow_between(
                x_centroid, w_k_box, buff=0.10, tip_length=0.13,
                stroke_width=2.5, color=ORANGE,
            ),
            arrow_between(
                x_centroid, w_v_box, buff=0.10, tip_length=0.13,
                stroke_width=2.5, color=GREEN,
            ),
        ]
        # Arrows from each W_* to its tensor row centroid (middle column).
        out_arrows = [
            arrow_between(
                w_q_box, q_cols[1], buff=0.08, tip_length=0.12,
                stroke_width=2.5, color=TEAL,
            ),
            arrow_between(
                w_k_box, k_cols[1], buff=0.08, tip_length=0.12,
                stroke_width=2.5, color=ORANGE,
            ),
            arrow_between(
                w_v_box, v_cols[1], buff=0.08, tip_length=0.12,
                stroke_width=2.5, color=GREEN,
            ),
        ]

        self.play(
            FadeIn(w_q_box), FadeIn(w_k_box), FadeIn(w_v_box),
            *[Create(a) for a in proj_arrows],
            run_time=0.7,
        )
        self.play(
            *[FadeIn(c) for c in q_cols + k_cols + v_cols],
            FadeIn(q_label), FadeIn(k_label), FadeIn(v_label),
            *[Create(a) for a in out_arrows],
            run_time=0.9,
        )
        # Hold the Q/K/V layout long enough that midpoint frame sampling
        # has a good chance of catching it (the projections are pedagogically
        # the second-most-important step after the matrix itself).
        self.wait(1.1)

        # ============== Phase 3: clear, build attention matrix ==============
        # Fade out X row, projection boxes, projection arrows. We KEEP Q/K
        # row mini-labels (Q/K) but compact them out. We also KEEP V row but
        # need to MOVE it down to Y_V_BOTTOM so it's clear of the matrix.
        cleanup_phase2 = [
            FadeOut(token_mobs[0]), FadeOut(token_mobs[1]), FadeOut(token_mobs[2]),
            *[FadeOut(c) for c in x_columns],
            FadeOut(x_label),
            FadeOut(w_q_box), FadeOut(w_k_box), FadeOut(w_v_box),
            *[FadeOut(a) for a in proj_arrows],
            *[FadeOut(a) for a in out_arrows],
            *[FadeOut(c) for c in q_cols],
            *[FadeOut(c) for c in k_cols],
            FadeOut(q_label), FadeOut(k_label),
        ]
        self.play(*cleanup_phase2, run_time=0.5)
        # Manually remove from scene tree (FadeOut alone leaves them).
        for m in (
            *token_mobs, *x_columns, x_label,
            w_q_box, w_k_box, w_v_box,
            *proj_arrows, *out_arrows,
            *q_cols, *k_cols,
            q_label, k_label,
        ):
            self.remove(m)

        # Move V row into the bottom-V position, new spacing.
        v_targets = []
        for j, vc in enumerate(v_cols):
            target_x = self.X_V_BASE + j * self.X_V_SPACING
            v_targets.append(
                vc.animate.move_to([target_x, self.Y_V_BOTTOM, 0.0])
            )
        # V label (the big "V" symbol) repositions next to rightmost V col.
        new_v_label_pos = [
            self.X_V_BASE + 2 * self.X_V_SPACING + 0.45,
            self.Y_V_BOTTOM,
            0.0,
        ]
        v_label_anim = v_label.animate.move_to(new_v_label_pos)
        self.play(*v_targets, v_label_anim, run_time=0.6)

        # Build the attention matrix grid (3×3). Cells initially have low
        # opacity reflecting raw scores (after / sqrt(d_k)); we'll then
        # row-softmax them.
        matrix_cells: list[list[Square]] = []
        # Compute raw-score opacity scale relative to global max.
        raw_max = max(max(row) for row in self.RAW_SCORES)
        for i in range(3):
            row_cells: list[Square] = []
            for j in range(3):
                cell_x = self.X_MATRIX_CENTER + (j - 1) * self.MATRIX_CELL
                cell_y = self.Y_MATRIX + (1 - i) * self.MATRIX_CELL
                opacity = 0.10 + 0.55 * self.RAW_SCORES[i][j] / raw_max
                cell = Square(
                    side_length=self.MATRIX_CELL,
                    color=GREEN,           # raw-score phase tinted V-color
                    stroke_width=1.5,
                ).set_fill(GREEN, opacity=opacity)
                cell.move_to([cell_x, cell_y, 0.0])
                row_cells.append(cell)
            matrix_cells.append(row_cells)

        # Matrix label and bracket reading: "Q K^T / sqrt(d_k)" above.
        raw_label = (
            MathTex(r"\frac{Q K^\top}{\sqrt{d_k}}")
            .scale(0.55)
            .move_to([self.X_MATRIX_CENTER, self.Y_MATRIX + 1.55, 0.0])
        )
        # Row indicators i=1..3 to the LEFT of the matrix.
        row_lbls: list[MathTex] = []
        for i in range(3):
            lbl = (
                MathTex(rf"q_{i + 1}")
                .scale(0.45)
                .move_to(
                    [
                        self.X_MATRIX_CENTER - 1.5 * self.MATRIX_CELL - 0.30,
                        self.Y_MATRIX + (1 - i) * self.MATRIX_CELL,
                        0.0,
                    ]
                )
            )
            row_lbls.append(lbl)
        # Column indicators k=1..3 below the matrix.
        col_lbls: list[MathTex] = []
        for j in range(3):
            lbl = (
                MathTex(rf"k_{j + 1}")
                .scale(0.45)
                .move_to(
                    [
                        self.X_MATRIX_CENTER + (j - 1) * self.MATRIX_CELL,
                        self.Y_MATRIX - 1.5 * self.MATRIX_CELL - 0.25,
                        0.0,
                    ]
                )
            )
            col_lbls.append(lbl)

        all_matrix_mobs = [c for row in matrix_cells for c in row]
        self.play(FadeIn(raw_label), run_time=0.3)
        self.play(
            *[FadeIn(c) for c in all_matrix_mobs],
            *[FadeIn(lbl) for lbl in row_lbls],
            *[FadeIn(lbl) for lbl in col_lbls],
            run_time=0.6,
        )
        self.wait(0.4)

        # ============== Phase 4: row-wise softmax ==============
        # Re-color each row's cells to ATTN_WEIGHTS[i][j] opacity, change
        # color from GREEN (raw) → YELLOW (attention weights). Animate row
        # by row to make the row-wise nature visible.
        softmax_label = (
            MathTex(r"A = \mathrm{softmax}_\text{row}(Q K^\top / \sqrt{d_k})")
            .scale(0.50)
            .move_to([self.X_MATRIX_CENTER, self.Y_MATRIX + 1.55, 0.0])
        )
        # Replace the raw_label with the softmax label cleanly: fade out
        # first, then fade in. Overlapping the two during cross-fade caused
        # a visible "double formula" frame at midpoint sampling.
        self.play(FadeOut(raw_label), run_time=0.25)
        self.remove(raw_label)
        self.play(FadeIn(softmax_label), run_time=0.25)

        for i in range(3):
            row_max = max(self.ATTN_WEIGHTS[i])
            anims = []
            for j, cell in enumerate(matrix_cells[i]):
                w = self.ATTN_WEIGHTS[i][j]
                op = 0.12 + 0.78 * w / row_max
                anims.append(cell.animate.set_fill(YELLOW, opacity=op).set_stroke(YELLOW))
            self.play(*anims, run_time=0.45)
        self.wait(0.3)

        # Highlight the most-attended cell of row 1 (i=0): cross-attention
        # example — q_1 attends most to k_3.
        dom_j = max(range(3), key=lambda j: self.ATTN_WEIGHTS[0][j])
        dom_cell = matrix_cells[0][dom_j]
        dom_box = Rectangle(
            width=dom_cell.width + 0.10,
            height=dom_cell.height + 0.10,
            color=PURPLE,
            stroke_width=3,
        ).move_to(dom_cell.get_center())
        self.play(Create(dom_box), run_time=0.3)
        self.wait(0.3)

        # ============== Phase 5: Z = A V — row 1 in detail ==============
        # Show that row 1 of A (3 weights) dot-products with V's 3 columns
        # to produce Z_1. We do this by:
        #   - placing 3 ghost copies of v_cols, each tinted PURPLE with
        #     opacity ∝ ATTN_WEIGHTS[0][j];
        #   - flying them into the Z_1 slot;
        #   - revealing Z_1 (a TensorColumn at X_Z_BASE).
        z_columns: list[TensorColumn] = []
        for i in range(3):
            zx = self.X_Z_BASE + i * self.X_Z_SPACING
            z = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.OUT_CELL,
                color=PURPLE,
                fill_opacity=0.20,
            ).move_to([zx, self.Y_Z_BOTTOM, 0.0])
            z_columns.append(z)
        z_label = (
            MathTex(r"Z = A V")
            .scale(0.55)
            .next_to(z_columns[-1], RIGHT, buff=0.20)
        )

        # Build "Z_1" target first (only z_columns[0]).
        self.play(FadeIn(z_columns[0]), run_time=0.3)

        # Pour V columns into Z_1 with row-1 weights.
        ghost_groups: list[TensorColumn] = []
        pour_anims = []
        row1_max = max(self.ATTN_WEIGHTS[0])
        for j, vc in enumerate(v_cols):
            w = self.ATTN_WEIGHTS[0][j]
            ghost = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.QKV_CELL,
                color=PURPLE,
                fill_opacity=0.10 + 0.65 * w / row1_max,
            ).move_to(vc.get_center())
            ghost_groups.append(ghost)
            target = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.OUT_CELL,
                color=PURPLE,
                fill_opacity=0.10 + 0.65 * w / row1_max,
            ).move_to(z_columns[0].get_center())
            pour_anims.append(TransformFromCopy(ghost, target))

        self.play(*[FadeIn(g) for g in ghost_groups], run_time=0.3)
        self.play(*pour_anims, run_time=0.8)
        self.play(*[FadeOut(g) for g in ghost_groups], run_time=0.2)
        for g in ghost_groups:
            self.remove(g)

        # Brighten Z_1 to indicate it's now filled.
        bright = [c.animate.set_fill(PURPLE, opacity=0.55) for c in z_columns[0].cells]
        self.play(*bright, run_time=0.3)
        self.wait(0.2)

        # ============== Phase 6: Z_2 and Z_3 in batch ==============
        self.play(FadeIn(z_columns[1]), FadeIn(z_columns[2]), FadeIn(z_label), run_time=0.4)
        # Quickly fill them with their final fills (different dominance).
        batch_bright = []
        for i in (1, 2):
            for c in z_columns[i].cells:
                batch_bright.append(c.animate.set_fill(PURPLE, opacity=0.55))
        self.play(*batch_bright, run_time=0.5)
        self.wait(0.4)

        # ============== Phase 7: final caption ==============
        caption = (
            MathTex(
                r"\text{parallelizable: all queries computed at once via matmul}"
            )
            .scale(0.50)
            .to_edge(DOWN, buff=0.18)
        )
        self.play(FadeIn(caption))
        self.wait(0.8)
