"""Scaled dot-product attention scene.

Shows the mechanism in 4 steps:
1. Input tokens → X embeddings
2. Project X → Q, K, V via learned W matrices
3. Attention matrix: softmax(QK^T / sqrt(d_k))
4. Output: Z = A · V

Convention: bottom-to-top computation flow (inputs at bottom, outputs at top).
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
    SurroundingRectangle,
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


class ScaledDotProductAttention(Scene):
    """X → (Q, K, V) → softmax(Q K^T / sqrt(d_k)) V → Z."""

    # ---- Layout constants ----
    TENSOR_DIM = 3
    X_CELL = 0.24
    QKV_CELL = 0.22
    OUT_CELL = 0.24

    # Token positions (3 tokens, centered)
    TOKENS = ["the", "cat", "sat"]
    X_TOK_BASE = -1.50
    X_TOK_SPACING = 1.50

    # ---- Phase 1: bottom-to-top flow ----
    Y_TITLE = 3.35            # reference formulas (always at top)
    Y_TOKEN = -2.80           # input tokens (BOTTOM — inputs)
    Y_X = -2.10               # X embeddings (above tokens)
    Y_PROJ = -1.00             # W_Q / W_K / W_V boxes
    Y_QKV = 0.20              # Q / K / V tensor rows (TOP of phase 1)

    # ---- Phase 2: attention matrix view (bottom-to-top) ----
    Y_V_BOTTOM = -1.80        # V row (input to multiplication — bottom)
    Y_MATRIX = 0.10           # center of A grid (middle)
    MATRIX_CELL = 0.42
    X_MATRIX_CENTER = -0.80  # shift matrix left to leave room for Z on right

    Y_Z_TOP = 0.10            # Z output row — same height as matrix, placed to the right
    X_V_BASE = -3.30
    X_V_SPACING = 1.10
    X_Z_BASE = 2.30
    X_Z_SPACING = 0.80

    # Toy attention scores Q K^T / sqrt(d_k).
    RAW_SCORES = [
        [0.5, 0.8, 2.4],
        [0.3, 2.6, 0.7],
        [0.4, 0.6, 2.5],
    ]
    # Correct row-softmax (sum to 1 per row).
    ATTN_WEIGHTS = [
        [0.10, 0.14, 0.76],   # row 0: attends to token 3
        [0.08, 0.80, 0.12],   # row 1: attends to token 2 (was wrong: 0.07/0.85/0.08)
        [0.10, 0.12, 0.78],   # row 2: attends to token 3
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

        # ============== Phase 1: tokens (bottom) → X → projections → Q,K,V (top) ==============
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
            .next_to(x_columns[-1], UP, buff=0.15)
        )
        self.play(*[FadeIn(t) for t in token_mobs], run_time=0.4)
        self.play(*[FadeIn(c) for c in x_columns], FadeIn(x_label), run_time=0.5)
        self.wait(0.2)

        # ============== Phase 2: project X → Q, K, V ==============
        x_group_q = -4.40
        x_group_k = 0.00
        x_group_v = 4.40
        group_spread = 0.60

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
            grp_lbl = (
                MathTex(label_str)
                .scale(0.55)
                .next_to(cols[-1], RIGHT, buff=0.18)
            )
            return box, cols, grp_lbl

        w_q_box, q_cols, q_label = _build_qkv_group(x_group_q, TEAL, "Q")
        w_k_box, k_cols, k_label = _build_qkv_group(x_group_k, ORANGE, "K")
        w_v_box, v_cols, v_label = _build_qkv_group(x_group_v, GREEN, "V")

        # Arrows from X columns upward to each W_* box (bottom-to-top)
        # Use separate X columns to avoid arrows crossing through W_K
        proj_arrows = [
            arrow_between(
                x_columns[0], w_q_box, buff=0.10, tip_length=0.13,
                stroke_width=2.5, color=TEAL,
            ),
            arrow_between(
                x_columns[1], w_k_box, buff=0.10, tip_length=0.13,
                stroke_width=2.5, color=ORANGE,
            ),
            arrow_between(
                x_columns[2], w_v_box, buff=0.10, tip_length=0.13,
                stroke_width=2.5, color=GREEN,
            ),
        ]
        # Arrows from each W_* upward to its tensor row
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
        self.wait(1.1)

        # ============== Phase 3: Q·K^T → attention matrix ==============
        # Clean up tokens, X, W boxes, arrows — keep Q, K, V + labels
        cleanup_phase2 = [
            FadeOut(token_mobs[0]), FadeOut(token_mobs[1]), FadeOut(token_mobs[2]),
            *[FadeOut(c) for c in x_columns],
            FadeOut(x_label),
            FadeOut(w_q_box), FadeOut(w_k_box), FadeOut(w_v_box),
            *[FadeOut(a) for a in proj_arrows],
            *[FadeOut(a) for a in out_arrows],
        ]
        self.play(*cleanup_phase2, run_time=0.5)
        for m in (
            *token_mobs, *x_columns, x_label,
            w_q_box, w_k_box, w_v_box,
            *proj_arrows, *out_arrows,
        ):
            self.remove(m)

        # -- Target positions for Q·K^T layout --
        # Q columns → left of matrix (aligned with rows)
        q_dst_x = self.X_MATRIX_CENTER - 1.5 * self.MATRIX_CELL - 0.55
        q_move_anims: list[Any] = []
        for i, qc in enumerate(q_cols):
            ty = self.Y_MATRIX + (1 - i) * self.MATRIX_CELL
            q_move_anims.append(qc.animate.move_to([q_dst_x, ty, 0.0]))
        q_move_anims.append(q_label.animate.move_to([
            q_dst_x, self.Y_MATRIX + 1.5 * self.MATRIX_CELL + 0.25, 0.0,
        ]))

        # K columns → above matrix columns (= K^T)
        k_dst_y = self.Y_MATRIX + 1.5 * self.MATRIX_CELL + 0.55
        k_move_anims: list[Any] = []
        for j, kc in enumerate(k_cols):
            tx = self.X_MATRIX_CENTER + (j - 1) * self.MATRIX_CELL
            k_move_anims.append(kc.animate.move_to([tx, k_dst_y, 0.0]))
        # Replace "K" label with "K^T"
        self.play(FadeOut(k_label), run_time=0.15)
        self.remove(k_label)
        kt_label = (
            MathTex(r"K^\top")
            .scale(0.55)
            .move_to([
                self.X_MATRIX_CENTER + 1.5 * self.MATRIX_CELL + 0.40,
                k_dst_y, 0.0,
            ])
        )

        # V → bottom-left
        v_targets: list[Any] = []
        for j, vc in enumerate(v_cols):
            target_x = self.X_V_BASE + j * self.X_V_SPACING
            v_targets.append(
                vc.animate.move_to([target_x, self.Y_V_BOTTOM, 0.0])
            )
        new_v_label_pos = [
            self.X_V_BASE + 2 * self.X_V_SPACING + 0.50,
            self.Y_V_BOTTOM,
            0.0,
        ]
        v_label_anim = v_label.animate.move_to(new_v_label_pos)

        # Animate all repositioning together
        self.play(
            *q_move_anims, *k_move_anims, *v_targets, v_label_anim,
            run_time=0.8,
        )
        self.play(FadeIn(kt_label), run_time=0.25)

        # Build 3×3 attention matrix cells (not yet visible)
        matrix_cells: list[list[Square]] = []
        raw_max = max(max(row) for row in self.RAW_SCORES)
        for i in range(3):
            row_cells: list[Square] = []
            for j in range(3):
                cell_x = self.X_MATRIX_CENTER + (j - 1) * self.MATRIX_CELL
                cell_y = self.Y_MATRIX + (1 - i) * self.MATRIX_CELL
                opacity = 0.10 + 0.55 * self.RAW_SCORES[i][j] / raw_max
                cell = Square(
                    side_length=self.MATRIX_CELL,
                    color=TEAL,
                    stroke_width=1.5,
                ).set_fill(TEAL, opacity=opacity)
                cell.move_to([cell_x, cell_y, 0.0])
                row_cells.append(cell)
            matrix_cells.append(row_cells)

        # QK^T / sqrt(d_k) label above K^T columns
        raw_label = (
            MathTex(r"\frac{Q K^\top}{\sqrt{d_k}}")
            .scale(0.55)
            .move_to([self.X_MATRIX_CENTER, k_dst_y + 0.65, 0.0])
        )
        self.play(FadeIn(raw_label), run_time=0.3)

        # -- Row 0: step-by-step dot products q_1 · k_j → cell(0,j) --
        q0_rect = SurroundingRectangle(
            q_cols[0], color=YELLOW, buff=0.06, stroke_width=2.5,
        )
        self.play(Create(q0_rect), run_time=0.25)
        for j in range(3):
            kj_rect = SurroundingRectangle(
                k_cols[j], color=YELLOW, buff=0.06, stroke_width=2.5,
            )
            self.play(
                Create(kj_rect), FadeIn(matrix_cells[0][j]), run_time=0.35,
            )
            self.play(FadeOut(kj_rect), run_time=0.15)
        self.play(FadeOut(q0_rect), run_time=0.15)

        # -- Rows 1–2: batch fill --
        row12_anims = [
            FadeIn(matrix_cells[i][j]) for i in (1, 2) for j in range(3)
        ]
        self.play(*row12_anims, run_time=0.5)
        self.wait(0.3)

        # Fade out Q, K tensor columns; add text labels instead
        self.play(
            *[FadeOut(qc) for qc in q_cols],
            FadeOut(q_label),
            *[FadeOut(kc) for kc in k_cols],
            FadeOut(kt_label),
            run_time=0.4,
        )
        for m in (*q_cols, q_label, *k_cols, kt_label):
            self.remove(m)

        # Move raw_label down to standard position above matrix
        self.play(
            raw_label.animate.move_to([
                self.X_MATRIX_CENTER, self.Y_MATRIX + 1.20, 0.0,
            ]),
            run_time=0.3,
        )

        # Row labels q_1..q_3 to the left
        row_lbls: list[MathTex] = []
        for i in range(3):
            lbl = (
                MathTex(rf"q_{i + 1}")
                .scale(0.58)
                .move_to([
                    self.X_MATRIX_CENTER - 1.5 * self.MATRIX_CELL - 0.30,
                    self.Y_MATRIX + (1 - i) * self.MATRIX_CELL,
                    0.0,
                ])
            )
            row_lbls.append(lbl)
        # Column labels k_1..k_3 below
        col_lbls: list[MathTex] = []
        for j in range(3):
            lbl = (
                MathTex(rf"k_{j + 1}")
                .scale(0.45)
                .move_to([
                    self.X_MATRIX_CENTER + (j - 1) * self.MATRIX_CELL,
                    self.Y_MATRIX - 1.5 * self.MATRIX_CELL - 0.25,
                    0.0,
                ])
            )
            col_lbls.append(lbl)
        self.play(
            *[FadeIn(lbl) for lbl in row_lbls],
            *[FadeIn(lbl) for lbl in col_lbls],
            run_time=0.3,
        )
        self.wait(0.4)

        # ============== Phase 4: row-wise softmax ==============
        softmax_label = (
            MathTex(r"A = \mathrm{softmax}_\text{row}(Q K^\top / \sqrt{d_k})")
            .scale(0.50)
            .move_to([self.X_MATRIX_CENTER, self.Y_MATRIX + 1.20, 0.0])
        )
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

        # Highlight dominant cell of row 1: q_1 attends to k_3
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

        # ============== Phase 5: Z = A V — output at TOP ==============
        z_columns: list[TensorColumn] = []
        for i in range(3):
            zx = self.X_Z_BASE + i * self.X_Z_SPACING
            z = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.OUT_CELL,
                color=PURPLE,
                fill_opacity=0.20,
            ).move_to([zx, self.Y_Z_TOP, 0.0])
            z_columns.append(z)
        z_label = (
            MathTex(r"Z = A V")
            .scale(0.55)
            .next_to(z_columns[-1], RIGHT, buff=0.20)
        )

        # Arrow from attention matrix to Z_1 (horizontal, A·V → Z)
        matrix_group = VGroup(*[c for row in matrix_cells for c in row])
        av_arrow = arrow_between(
            matrix_group, z_columns[0],
            buff=0.12, tip_length=0.13,
            stroke_width=2.5, color=PURPLE,
        )

        # Build Z_1 first — show the A→Z arrow at same time
        self.play(FadeIn(z_columns[0]), Create(av_arrow), run_time=0.4)

        # Pour V columns into Z_1 with row-1 weights
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

        # Brighten Z_1
        bright = [c.animate.set_fill(PURPLE, opacity=0.55) for c in z_columns[0].cells]
        self.play(*bright, run_time=0.3)
        self.wait(0.2)

        # ============== Phase 6: Z_2 and Z_3 ==============
        self.play(FadeIn(z_columns[1]), FadeIn(z_columns[2]), FadeIn(z_label), run_time=0.4)
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
