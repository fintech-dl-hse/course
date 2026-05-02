"""Multi-head attention scene for seminar 09 (V07 of the curriculum catalog).

Расширение scaled-dot-product attention (V06) на ``h`` параллельных голов.
Идея: вместо одной attention-матрицы Q, K, V проецируются и разбиваются
на ``h`` подпространств, каждая голова считает собственный softmax(Q K^T)
и собственный выход; результаты конкатенируются и линейно проецируются
``W_O``.

Сценарий по фазам:

1. Сверху — формулы:
   ``\\mathrm{head}_i = \\mathrm{Attn}(Q_i, K_i, V_i)``
   ``\\mathrm{MHA}(Q,K,V) = \\mathrm{Concat}(\\mathrm{head}_1,\\ldots,\\mathrm{head}_h) W_O``.
2. Компактные Q (cyan), K (orange), V (green) в виде 3 колонок (как в V06).
3. Split: каждая колонка визуально расслаивается на 4 горизонтальных
   полосы, окрашенных в палитру голов (red / yellow / green / purple).
4. Параллельные attention: 4 мини-матрицы 3×3, каждая со своим
   распределением весов — разные паттерны внимания.
5. 4 выходных слаба голов появляются под матрицами.
6. Concat: 4 слаба склеиваются вертикально в один тензор на токен.
7. ``W_O``: финальный LabeledBox (TEAL) проецирует склеенный тензор в
   итоговый MHA-выход.
8. Финальная подпись: ``multiple heads = multiple subspaces of attention``.

Палитра: Q = TEAL (cyan), K = ORANGE, V = GREEN — наследуется из V06.
Головы: head1 = RED, head2 = YELLOW, head3 = GREEN, head4 = PURPLE.
Финальная W_O — TEAL.

Использует общие примитивы ``shared.neural`` (TensorColumn, LabeledBox,
arrow_between). Локальный ``_horizontal_arrow`` хелпер скопирован из
V06 — лифт в shared/ запланирован отдельным рефактором.
"""
from __future__ import annotations

from typing import Any

from manim import (
    Arrow,
    DOWN,
    FadeIn,
    FadeOut,
    GREEN,
    LEFT,
    MathTex,
    ORANGE,
    PURPLE,
    RED,
    RIGHT,
    Rectangle,
    Scene,
    Square,
    TEAL,
    Tex,
    Transform,
    UP,
    VGroup,
    VMobject,
    WHITE,
    YELLOW,
    Write,
)

from shared.neural import LabeledBox, TensorColumn, arrow_between


# Color palette for the 4 heads — red, yellow, green, purple.
HEAD_COLORS: tuple[str, str, str, str] = (RED, YELLOW, GREEN, PURPLE)
HEAD_NAMES: tuple[str, str, str, str] = ("1", "2", "3", "4")


def _horizontal_arrow(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    """Стрелка, всегда прилипающая к правому/левому краю объектов.

    Локальная копия (см. ``scaled_dot_product.py`` V06).
    """
    defaults: dict[str, Any] = {"buff": 0.08, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


class MultiHeadAttention(Scene):
    """V07: Q, K, V → split into h heads → parallel Attn → Concat → W_O."""

    # ---- Layout constants — 720p (-qm) frame ±(7.11, 4.0) ----
    NUM_HEADS = 4
    NUM_TOKENS = 3

    # Phase 1 (compact Q/K/V before split)
    QKV_CELL_PRE = 0.22           # 4 cells per token column — slightly larger for readability
    QKV_DIM_PRE = NUM_HEADS       # one cell per head — important for split metaphor
    Y_QKV_PRE = 1.40              # lower to give more vertical breathing room
    X_Q_PRE_BASE = -4.80          # leftmost Q column x — brought in from edge
    X_K_PRE_BASE = -1.20          # leftmost K column x
    X_V_PRE_BASE = 2.40           # leftmost V column x
    PRE_TOK_SPACING = 0.50        # within-group token spacing

    # Phase 2 — split: same x positions, but cells get color-coded by head row.

    # Title row — moved down slightly so equations don't dominate the frame
    Y_TITLE = 3.35

    # Phase 3 — 4 mini attention matrices in the middle row.
    Y_MATRIX = 0.10
    MATRIX_CELL = 0.26            # larger cells for better readability
    MATRIX_GROUP_DX = 3.50        # spacing between adjacent matrix centers
    MATRIX_X_BASE = -1.5 * 3.50  # leftmost matrix center (-5.25, -1.75, +1.75, +5.25)

    # Phase 4 — head output slabs (1 per head per token), Y row near bottom.
    Y_HEAD_OUT = -1.40
    HEAD_OUT_CELL = 0.22          # match matrix cell size
    HEAD_OUT_DIM = 1              # 1 cell per slab (slab IS the head's output for that token)
    # We display 4 head-output slabs per matrix group, side-by-side under
    # each matrix; each slab is itself a tiny TensorColumn of 1 cell.
    # NUM_TOKENS slabs per head per matrix group.

    # Phase 5 — concat: a single tall TensorColumn per token, Y row.
    Y_CONCAT = -1.40              # same row will be re-used after fade
    CONCAT_CELL = 0.22
    CONCAT_DIM = NUM_HEADS        # 4-tall stack
    CONCAT_TOK_SPACING = 0.85
    CONCAT_X_BASE = -1.0          # leftmost concat token column x

    # Phase 6 — W_O and final output
    Y_WO = -2.65
    Y_FINAL = -3.50               # final MHA output row
    FINAL_CELL = 0.22
    FINAL_DIM = NUM_HEADS         # keep same height as concat for visual continuity
    FINAL_TOK_SPACING = 0.85
    FINAL_X_BASE = -1.0

    # Toy attention patterns for 4 heads — distinct dominant cells.
    # Each head: 3x3 row-softmax, but with a unique "favorite" pattern.
    # head1 (red):    diagonal — local attention.
    # head2 (yellow): top-right triangle — looks ahead.
    # head3 (green):  off-diagonal swap (1<->3).
    # head4 (purple): uniform — broad context.
    HEAD_WEIGHTS: tuple[tuple[tuple[float, ...], ...], ...] = (
        # head 1: diagonal
        (
            (0.78, 0.12, 0.10),
            (0.10, 0.80, 0.10),
            (0.12, 0.10, 0.78),
        ),
        # head 2: looks ahead (each row attends to k_j with j>=i)
        (
            (0.20, 0.30, 0.50),
            (0.10, 0.30, 0.60),
            (0.05, 0.20, 0.75),
        ),
        # head 3: swap (q_1 -> k_3, q_3 -> k_1, q_2 -> k_2)
        (
            (0.10, 0.15, 0.75),
            (0.10, 0.80, 0.10),
            (0.75, 0.15, 0.10),
        ),
        # head 4: near-uniform
        (
            (0.32, 0.36, 0.32),
            (0.34, 0.34, 0.32),
            (0.32, 0.36, 0.32),
        ),
    )

    def construct(self) -> None:
        # ============== Phase 0: title ==============
        eq_head = MathTex(
            r"\mathrm{head}_i = \mathrm{Attn}(Q_i,\, K_i,\, V_i)",
        ).scale(0.55)
        eq_mha = MathTex(
            r"\mathrm{MHA}(Q,K,V) = \mathrm{Concat}",
            r"(\mathrm{head}_1, \ldots, \mathrm{head}_h)\, W_O",
        ).scale(0.55)
        title = (
            VGroup(eq_head, eq_mha)
            .arrange(DOWN, buff=0.14)
            .move_to([0.0, self.Y_TITLE, 0.0])
        )
        self.play(Write(title))
        self.wait(0.2)

        # ============== Phase 1: compact Q, K, V (pre-split) ==============
        # Each of Q/K/V: 3 token-columns, each 4 cells tall, single base color.
        def _build_pre_group(
            x_base: float, color: str, label_str: str,
        ) -> tuple[list[TensorColumn], MathTex]:
            cols: list[TensorColumn] = []
            for j in range(self.NUM_TOKENS):
                cx = x_base + j * self.PRE_TOK_SPACING
                col = TensorColumn(
                    dim=self.QKV_DIM_PRE,
                    cell_size=self.QKV_CELL_PRE,
                    color=color,
                    fill_opacity=0.40,
                ).move_to([cx, self.Y_QKV_PRE, 0.0])
                cols.append(col)
            lbl = (
                MathTex(label_str)
                .scale(0.55)
                .next_to(cols[-1], RIGHT, buff=0.18)
            )
            return cols, lbl

        q_cols, q_label = _build_pre_group(self.X_Q_PRE_BASE, TEAL, "Q")
        k_cols, k_label = _build_pre_group(self.X_K_PRE_BASE, ORANGE, "K")
        v_cols, v_label = _build_pre_group(self.X_V_PRE_BASE, GREEN, "V")

        self.play(
            *[FadeIn(c) for c in q_cols + k_cols + v_cols],
            FadeIn(q_label), FadeIn(k_label), FadeIn(v_label),
            run_time=0.6,
        )
        self.wait(0.4)

        # ============== Phase 2: split into h heads (per-cell recolor) ==============
        # Each Q/K/V column has NUM_HEADS=4 cells, top→bottom — recolor each
        # cell to its head color. This is the "split" metaphor: row k of the
        # column is now visibly the slice that flows into head k.
        split_anims = []
        for col in q_cols + k_cols + v_cols:
            for k, cell in enumerate(col.cells):
                hc = HEAD_COLORS[k]
                split_anims.append(
                    cell.animate.set_fill(hc, opacity=0.55).set_stroke(hc)
                )
        # Tiny head-key legend on the LEFT margin so the audience can decode
        # the new colors without guessing.
        legend_items: list[VGroup] = []
        legend_x = -6.20          # safe margin (frame goes to ±7.11)
        legend_y_top = 1.40       # aligned with Y_QKV_PRE
        legend_step = -0.36
        for k in range(self.NUM_HEADS):
            sq = Square(
                side_length=0.22, color=HEAD_COLORS[k], stroke_width=2,
            ).set_fill(HEAD_COLORS[k], opacity=0.55)
            txt = MathTex(rf"\mathrm{{head}}_{k + 1}").scale(0.50)
            grp = VGroup(sq, txt).arrange(RIGHT, buff=0.10)
            grp.move_to([legend_x, legend_y_top + k * legend_step, 0.0])
            # Shift left edge of group to legend_x so the swatch aligns left.
            grp.shift([legend_x - grp.get_left()[0], 0, 0])
            legend_items.append(grp)

        self.play(*split_anims, run_time=0.7)
        self.play(*[FadeIn(g) for g in legend_items], run_time=0.4)
        self.wait(0.6)

        # ============== Phase 3: clear pre-split, build 4 mini matrices ==============
        # Also fade legend here so it doesn't overlap matrix head titles on-screen.
        cleanup1 = [
            *[FadeOut(c) for c in q_cols + k_cols + v_cols],
            FadeOut(q_label), FadeOut(k_label), FadeOut(v_label),
            *[FadeOut(g) for g in legend_items],
        ]
        self.play(*cleanup1, run_time=0.4)
        for m in (*q_cols, *k_cols, *v_cols, q_label, k_label, v_label, *legend_items):
            self.remove(m)

        # 4 mini attention matrices side-by-side at Y_MATRIX.
        # Each matrix is 3x3 cells of MATRIX_CELL side.
        matrix_groups: list[list[list[Square]]] = []
        head_titles: list[MathTex] = []
        for h_idx in range(self.NUM_HEADS):
            cx = self.MATRIX_X_BASE + h_idx * self.MATRIX_GROUP_DX
            color = HEAD_COLORS[h_idx]
            cells: list[list[Square]] = []
            for i in range(self.NUM_TOKENS):
                row_cells: list[Square] = []
                for j in range(self.NUM_TOKENS):
                    cell_x = cx + (j - 1) * self.MATRIX_CELL
                    cell_y = self.Y_MATRIX + (1 - i) * self.MATRIX_CELL
                    w = self.HEAD_WEIGHTS[h_idx][i][j]
                    row_max = max(self.HEAD_WEIGHTS[h_idx][i])
                    op = 0.15 + 0.75 * w / row_max
                    cell = Square(
                        side_length=self.MATRIX_CELL,
                        color=color,
                        stroke_width=1.2,
                    ).set_fill(color, opacity=op)
                    cell.move_to([cell_x, cell_y, 0.0])
                    row_cells.append(cell)
                cells.append(row_cells)
            matrix_groups.append(cells)

            # Head title above each matrix.
            ht = (
                MathTex(rf"\mathrm{{head}}_{h_idx + 1}")
                .scale(0.60)
                .move_to([cx, self.Y_MATRIX + 1.5 * self.MATRIX_CELL + 0.38, 0.0])
            )
            head_titles.append(ht)

        # Single shared formula above the row of matrices.
        attn_label = (
            MathTex(r"A_i = \mathrm{softmax}(Q_i K_i^\top / \sqrt{d_k})")
            .scale(0.55)
            .move_to([0.0, self.Y_MATRIX + 1.5 * self.MATRIX_CELL + 1.00, 0.0])
        )

        all_cells = [c for grp in matrix_groups for row in grp for c in row]
        self.play(FadeIn(attn_label), run_time=0.3)
        self.play(
            *[FadeIn(c) for c in all_cells],
            *[FadeIn(t) for t in head_titles],
            run_time=0.7,
        )
        self.wait(0.7)

        # ============== Phase 4: each matrix produces a head-output slab ==============
        # Below each matrix, draw a small TensorColumn (one per token, 1 cell
        # tall, head-colored).
        Y_HEAD_OUT = -0.95
        head_slabs: list[list[TensorColumn]] = []
        for h_idx in range(self.NUM_HEADS):
            cx = self.MATRIX_X_BASE + h_idx * self.MATRIX_GROUP_DX
            color = HEAD_COLORS[h_idx]
            slabs: list[TensorColumn] = []
            for j in range(self.NUM_TOKENS):
                slab_x = cx + (j - 1) * self.MATRIX_CELL
                slab = TensorColumn(
                    dim=1,
                    cell_size=self.MATRIX_CELL,  # match matrix cells
                    color=color,
                    fill_opacity=0.55,
                ).move_to([slab_x, Y_HEAD_OUT, 0.0])
                slabs.append(slab)
            head_slabs.append(slabs)

        # Arrows from each matrix down to its slab row (centered).
        head_out_arrows: list[Arrow] = []
        for h_idx in range(self.NUM_HEADS):
            cx = self.MATRIX_X_BASE + h_idx * self.MATRIX_GROUP_DX
            mid_cell = matrix_groups[h_idx][2][1]   # bottom-middle cell of matrix
            mid_slab = head_slabs[h_idx][1]
            color = HEAD_COLORS[h_idx]
            arr = arrow_between(
                mid_cell, mid_slab,
                buff=0.06, tip_length=0.12, stroke_width=2.2, color=color,
            )
            head_out_arrows.append(arr)

        # Slab labels: Z_1, Z_2, Z_3, Z_4 next to rightmost slab of each group?
        # Too many labels — skip and rely on color-coding + head titles.
        self.play(
            *[FadeIn(s) for grp in head_slabs for s in grp],
            *[FadeIn(a) for a in head_out_arrows],
            run_time=0.6,
        )
        self.wait(0.6)

        # ============== Phase 5: clean up matrices, concat heads ==============
        # Fade matrices + arrows + head titles + attn_label, KEEP head_slabs
        # but transform them into a stacked Concat representation.
        cleanup2 = [
            *[FadeOut(c) for c in all_cells],
            *[FadeOut(t) for t in head_titles],
            *[FadeOut(a) for a in head_out_arrows],
            FadeOut(attn_label),
        ]
        self.play(*cleanup2, run_time=0.4)
        for m in (*all_cells, *head_titles, *head_out_arrows, attn_label):
            self.remove(m)

        # Now MOVE the head slabs into a Concat layout: per-token vertical
        # stacks of 4 head-colored cells. Place 3 token columns centered.
        # Target geometry: per token j, stack of 4 cells (head 1 top → head 4 bottom).
        Y_CONCAT_TOP = 1.20         # top of the stack
        CONCAT_CELL = 0.32          # bigger so the Concat is visually prominent
        CONCAT_TOK_SPACING = 1.40
        CONCAT_X_BASE = -CONCAT_TOK_SPACING  # 3 columns: -1.4, 0.0, +1.4

        slab_move_anims = []
        for h_idx in range(self.NUM_HEADS):
            for j in range(self.NUM_TOKENS):
                target_x = CONCAT_X_BASE + j * CONCAT_TOK_SPACING
                target_y = Y_CONCAT_TOP - h_idx * CONCAT_CELL
                slab = head_slabs[h_idx][j]
                # Build a target square at the new location with new size.
                target_square = Square(
                    side_length=CONCAT_CELL,
                    color=HEAD_COLORS[h_idx],
                    stroke_width=2,
                ).set_fill(HEAD_COLORS[h_idx], opacity=0.55)
                target_square.move_to([target_x, target_y, 0.0])
                slab_move_anims.append(Transform(slab, target_square))

        concat_label = (
            MathTex(r"\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_4)")
            .scale(0.55)
            .move_to([0.0, Y_CONCAT_TOP + 4 * CONCAT_CELL * 0.5 + 0.55, 0.0])
        )
        self.play(*slab_move_anims, FadeIn(concat_label), run_time=0.9)
        self.wait(0.5)

        # ============== Phase 6: W_O and final output ==============
        # W_O is a single LabeledBox (TEAL) below the concat columns. Output
        # is a row of 3 final tokens (TEAL TensorColumns of dim 1).
        Y_WO = Y_CONCAT_TOP - self.NUM_HEADS * CONCAT_CELL - 0.55   # below concat
        Y_FINAL = Y_WO - 0.85
        wo_box = LabeledBox(
            label="W_O",
            width=2.4,
            height=0.55,
            label_scale=0.55,
            color=TEAL,
            fill_opacity=0.18,
        ).move_to([0.0, Y_WO, 0.0])

        final_cols: list[TensorColumn] = []
        for j in range(self.NUM_TOKENS):
            fx = CONCAT_X_BASE + j * CONCAT_TOK_SPACING
            fc = TensorColumn(
                dim=1,
                cell_size=CONCAT_CELL,
                color=TEAL,
                fill_opacity=0.55,
            ).move_to([fx, Y_FINAL, 0.0])
            final_cols.append(fc)

        final_label = (
            MathTex(r"\mathrm{MHA}(Q,K,V)")
            .scale(0.50)
            .next_to(final_cols[-1], RIGHT, buff=0.20)
        )

        # Arrows: each concat token column → wo_box → corresponding final token.
        # We anchor incoming arrows at the bottom of the bottom-most slab cell.
        concat_to_wo: list[Arrow] = []
        wo_to_final: list[Arrow] = []
        for j in range(self.NUM_TOKENS):
            # bottommost slab cell at column j is head_slabs[3][j] (after Transform).
            bottom_slab = head_slabs[self.NUM_HEADS - 1][j]
            arr_in = arrow_between(
                bottom_slab, wo_box,
                buff=0.10, tip_length=0.12, stroke_width=2.2, color=WHITE,
            )
            concat_to_wo.append(arr_in)

            final_col = final_cols[j]
            arr_out = arrow_between(
                wo_box, final_col,
                buff=0.10, tip_length=0.12, stroke_width=2.2, color=TEAL,
            )
            wo_to_final.append(arr_out)

        self.play(FadeIn(wo_box), *[FadeIn(a) for a in concat_to_wo], run_time=0.5)
        self.play(
            *[FadeIn(c) for c in final_cols],
            *[FadeIn(a) for a in wo_to_final],
            FadeIn(final_label),
            run_time=0.6,
        )
        self.wait(0.7)

        # ============== Phase 7: caption ==============
        caption = (
            MathTex(
                r"\text{multiple heads = multiple subspaces of attention}"
            )
            .scale(0.50)
            .to_edge(DOWN, buff=0.18)
        )
        self.play(FadeIn(caption))
        self.wait(0.8)
