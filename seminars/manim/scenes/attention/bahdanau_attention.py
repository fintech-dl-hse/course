"""Bahdanau additive-attention scene for seminar 09 (V05 of the curriculum catalog).

Показывает, как attention снимает bottleneck из V04: декодер на шаге t
вместо одного фиксированного ``h_T`` читает ВСЕ скрытые состояния
энкодера ``h_1..h_T`` через content-based weighted sum.

1. Сверху — формулы Bahdanau:
   ``e_{t,j} = v^\\top \\tanh(W_a s_t + U_a h_j)``,
   ``\\alpha_{t,j} = \\mathrm{softmax}_j(e_{t,j})``,
   ``c_t = \\sum_j \\alpha_{t,j}\\, h_j``.
2. Сверху-середина — лента из 5 source-токенов и под ней ряд
   скрытых тензоров ``h_1..h_5`` (TensorColumn, dim=4, cell=0.25,
   BLUE) — рисуются разом без recurrence: тут речь не про энкодер.
3. Слева внизу — query-состояние декодера ``s_t`` (TensorColumn, GREEN).
4. Между ``s_t`` и каждым ``h_j`` рисуются стрелки выравнивания и
   над каждым ``h_j`` появляется маленький скаляр ``e_{t,j}``.
5. Softmax-шаг: значения ``e_{t,j}`` превращаются в ``\\alpha_{t,j}``
   (отображаются как ряд маленьких ячеек с opacity, пропорциональным
   весу; одна-две ячейки доминируют).
6. Контекст-вектор ``c_t`` справа — каждый ``h_j`` «вливается» в него
   с весом ``\\alpha_{t,j}`` (TransformFromCopy с opacity ∝ вес).
7. Намёк на следующий шаг декодера: ``s_{t+1}`` появляется правее и
   рисуется альтернативный паттерн весов (другая «строка»).
8. Финальная подпись: ``attention: decoder freely reads any source
   position``.

Сцена использует общие примитивы ``shared.neural`` (TensorColumn,
LabeledBox, arrow_between) и стилистически совместима с V04
``Seq2SeqBottleneck`` и V02 ``RNNForward``: те же тензорные ячейки
(``cell_size=0.25..0.27``), та же палитра (BLUE = encoder hidden,
ORANGE = attention weights, GREEN = decoder, PURPLE = context).
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
    """Стрелка, всегда прилипающая к правому/левому краю объектов."""
    defaults: dict[str, Any] = {"buff": 0.08, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


class BahdanauAttention(Scene):
    """V05: alignment scores → softmax → context vector ``c_t``."""

    # ---- Layout constants — tuned for 720p (-qm) frame ±(7.11, 4.0) ----
    SRC_CELL = 0.25       # encoder hidden tensor cell size (5 columns is dense)
    CTX_CELL = 0.27       # context vector cell size (slightly larger, accent)
    DEC_CELL = 0.27       # decoder state cell size
    TENSOR_DIM = 4

    # Vertical anchors.
    Y_TITLE = 3.30        # title equations
    Y_SRC_TOKEN = 1.95    # source tokens row
    Y_ENC = 1.20          # encoder hidden row h_1..h_5
    Y_SCORE = 0.45        # alignment score e_{t,j} above each h_j
    Y_WEIGHT = -0.30      # softmax weight cells α_{t,j}
    Y_DEC = -1.85         # decoder state s_t (lower-left)
    Y_CAPTION = -3.65

    # Encoder horizontal layout: 5 columns, centered in upper region.
    SRC_TOKENS = ["the", "quick", "brown", "fox", "jumps"]
    X_ENC_BASE = -3.95
    X_ENC_SPACING = 1.20

    # Decoder query position (lower-left).
    X_DEC = -5.10
    # Context vector position (mid-right).
    X_CTX = 4.40
    Y_CTX = -1.10
    # Hint of next decoder step.
    X_DEC_NEXT = -3.40

    # Toy alignment scores: scalar e_{t,j} values (just for display).
    SCORE_VALUES = [0.4, 1.6, 2.8, 0.9, 0.2]
    # Resulting softmax weights (precomputed, sum to 1.0). One dominant.
    WEIGHT_VALUES = [0.05, 0.16, 0.55, 0.18, 0.06]
    # Hypothetical second-step weights for the "next step" hint.
    WEIGHT_VALUES_NEXT = [0.10, 0.05, 0.07, 0.62, 0.16]

    def construct(self) -> None:
        # ---------------- Title ----------------
        eq_score = MathTex(
            r"e_{t,j} = v^\top \tanh(W_a\, s_t + U_a\, h_j)",
        ).scale(0.55)
        eq_softmax = MathTex(
            r"\alpha_{t,j} = \mathrm{softmax}_j(e_{t,j})",
            r"\quad",
            r"c_t = \sum_j \alpha_{t,j}\, h_j",
        ).scale(0.55)
        title = (
            VGroup(eq_score, eq_softmax)
            .arrange(DOWN, buff=0.14)
            .move_to([0.0, self.Y_TITLE, 0.0])
        )
        self.play(Write(title))
        self.wait(0.2)

        # ---------------- Encoder hidden row h_1..h_5 ----------------
        src_tokens: list[Tex] = []
        h_columns: list[TensorColumn] = []
        h_labels: list[MathTex] = []
        for j, tok in enumerate(self.SRC_TOKENS):
            x_pos = self.X_ENC_BASE + j * self.X_ENC_SPACING
            tok_mob = (
                Tex(rf"\textit{{``{tok}''}}")
                .scale(0.55)
                .move_to([x_pos, self.Y_SRC_TOKEN, 0.0])
            )
            h_t = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.SRC_CELL,
                color=BLUE,
                fill_opacity=0.30,
            ).move_to([x_pos, self.Y_ENC, 0.0])
            h_lbl = (
                MathTex(rf"h_{j + 1}")
                .scale(0.50)
                .next_to(h_t, DOWN, buff=0.08)
            )
            src_tokens.append(tok_mob)
            h_columns.append(h_t)
            h_labels.append(h_lbl)

        enc_group = VGroup(*src_tokens, *h_columns, *h_labels)
        self.play(FadeIn(enc_group), run_time=0.7)
        self.wait(0.2)

        # ---------------- Decoder query state s_t ----------------
        s_t = TensorColumn(
            dim=self.TENSOR_DIM,
            cell_size=self.DEC_CELL,
            color=GREEN,
            fill_opacity=0.40,
        ).move_to([self.X_DEC, self.Y_DEC, 0.0])
        s_lbl = (
            MathTex(r"s_t")
            .scale(0.55)
            .next_to(s_t, DOWN, buff=0.10)
        )
        s_caption = (
            MathTex(r"\text{decoder query}")
            .scale(0.45)
            .next_to(s_t, UP, buff=0.10)
        )
        self.play(FadeIn(s_t), FadeIn(s_lbl), FadeIn(s_caption), run_time=0.5)
        self.wait(0.2)

        # ---------------- Alignment-score arrows + scalars e_{t,j} ----------------
        # Arrows from s_t up to each h_j, plus a small scalar above h_j.
        # arrow_between picks top/bottom for vertical-dominant pairs, which is
        # fine here: source x-positions are spread across the frame so each
        # arrow has an empty corridor.
        score_arrows: list[Arrow] = []
        score_labels: list[MathTex] = []
        for j, h_j in enumerate(h_columns):
            arr = arrow_between(
                s_t,
                h_j,
                buff=0.10,
                tip_length=0.10,
                stroke_width=2,
                color=ORANGE,
            )
            score_arrows.append(arr)
            e_lbl = (
                MathTex(rf"e_{{t,{j + 1}}}={self.SCORE_VALUES[j]:.1f}")
                .scale(0.40)
                .move_to([h_j.get_center()[0], self.Y_SCORE, 0.0])
            )
            score_labels.append(e_lbl)

        self.play(
            *[Create(a) for a in score_arrows],
            run_time=0.7,
        )
        self.play(
            *[FadeIn(lbl) for lbl in score_labels],
            run_time=0.5,
        )
        self.wait(0.3)

        # ---------------- Softmax: scores → weights row ----------------
        # Each e_{t,j} morphs into a cell α_{t,j} whose fill opacity is
        # proportional to the (precomputed) softmax weight.
        weight_cells: list[Square] = []
        weight_labels: list[MathTex] = []
        for j, h_j in enumerate(h_columns):
            w = self.WEIGHT_VALUES[j]
            # Opacity scales between 0.12 (small) and 0.92 (dominant).
            opacity = 0.12 + 0.80 * w / max(self.WEIGHT_VALUES)
            cell = Square(
                side_length=0.36,
                color=ORANGE,
                stroke_width=2,
            ).set_fill(ORANGE, opacity=opacity)
            cell.move_to([h_j.get_center()[0], self.Y_WEIGHT, 0.0])
            weight_cells.append(cell)
            w_lbl = (
                MathTex(rf"{w:.2f}")
                .scale(0.38)
                .next_to(cell, DOWN, buff=0.06)
            )
            weight_labels.append(w_lbl)

        # Softmax caption to the LEFT of the row, in an empty corridor.
        softmax_caption = (
            MathTex(r"\alpha_{t,\cdot}=\mathrm{softmax}(e_{t,\cdot})")
            .scale(0.45)
            .move_to([-5.45, self.Y_WEIGHT, 0.0])
        )

        self.play(FadeIn(softmax_caption), run_time=0.3)
        self.play(
            *[FadeIn(c) for c in weight_cells],
            *[FadeIn(lbl) for lbl in weight_labels],
            *[FadeOut(lbl) for lbl in score_labels],
            run_time=0.7,
        )
        # Belt-and-suspenders: remove the faded score labels from scene tree.
        for lbl in score_labels:
            self.remove(lbl)
        self.wait(0.3)

        # Highlight the dominant weight cell (j=2 → "brown").
        dominant_idx = max(
            range(len(self.WEIGHT_VALUES)),
            key=lambda k: self.WEIGHT_VALUES[k],
        )
        dom_cell = weight_cells[dominant_idx]
        dom_box = Rectangle(
            width=dom_cell.width + 0.10,
            height=dom_cell.height + 0.10,
            color=YELLOW,
            stroke_width=3,
        ).move_to(dom_cell.get_center())
        self.play(Create(dom_box), run_time=0.3)
        self.wait(0.2)

        # ---------------- Context vector c_t ----------------
        # The context vector lives on the right; each h_j flashes and pours
        # in proportionally to alpha_{t,j}.
        c_t = TensorColumn(
            dim=self.TENSOR_DIM,
            cell_size=self.CTX_CELL,
            color=PURPLE,
            fill_opacity=0.20,
        ).move_to([self.X_CTX, self.Y_CTX, 0.0])
        c_lbl = (
            MathTex(r"c_t")
            .scale(0.55)
            .next_to(c_t, DOWN, buff=0.10)
        )
        c_caption = (
            MathTex(r"\text{context}")
            .scale(0.45)
            .next_to(c_t, UP, buff=0.10)
        )
        self.play(FadeIn(c_t), FadeIn(c_lbl), FadeIn(c_caption), run_time=0.4)

        # Each h_j contributes a faded copy that flies to c_t. Use
        # TransformFromCopy on a transient ghost so the original h_j stays
        # in place. Heavier weights = higher opacity copy.
        pour_anims = []
        ghost_groups: list[VGroup] = []
        for j, h_j in enumerate(h_columns):
            w = self.WEIGHT_VALUES[j]
            # Build a ghost column at h_j's position, opacity ∝ weight.
            ghost = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.SRC_CELL,
                color=PURPLE,
                fill_opacity=0.10 + 0.70 * w / max(self.WEIGHT_VALUES),
            ).move_to(h_j.get_center())
            ghost_groups.append(ghost)
            # Target shape: a copy at c_t's position with c_t cell size.
            target = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CTX_CELL,
                color=PURPLE,
                fill_opacity=0.10 + 0.70 * w / max(self.WEIGHT_VALUES),
            ).move_to(c_t.get_center())
            pour_anims.append(TransformFromCopy(ghost, target))

        self.play(*[FadeIn(g) for g in ghost_groups], run_time=0.3)
        self.play(*pour_anims, run_time=0.9)
        # Drop the ghost source columns; the targets blended into c_t.
        self.play(*[FadeOut(g) for g in ghost_groups], run_time=0.2)
        for g in ghost_groups:
            self.remove(g)

        # Brighten c_t to its "filled" state — sum of weighted contributions.
        bright_anims = []
        for cell in c_t.cells:
            bright_anims.append(cell.animate.set_fill(PURPLE, opacity=0.55))
        self.play(*bright_anims, run_time=0.4)
        self.wait(0.4)

        # ---------------- Hint of next decoder step ----------------
        # A second decoder query s_{t+1} appears slightly to the right; the
        # weight row redistributes to a different dominant column ("fox").
        # Drop the current alignment so the redistribution reads cleanly.
        cleanup = []
        cleanup.extend([FadeOut(a) for a in score_arrows])
        cleanup.append(FadeOut(dom_box))
        self.play(*cleanup, run_time=0.4)
        for a in score_arrows:
            self.remove(a)
        self.remove(dom_box)

        s_next = TensorColumn(
            dim=self.TENSOR_DIM,
            cell_size=self.DEC_CELL,
            color=GREEN,
            fill_opacity=0.40,
        ).move_to([self.X_DEC_NEXT, self.Y_DEC, 0.0])
        s_next_lbl = (
            MathTex(r"s_{t+1}")
            .scale(0.50)
            .next_to(s_next, DOWN, buff=0.10)
        )
        self.play(FadeIn(s_next), FadeIn(s_next_lbl), run_time=0.4)

        # Redistribute the weight row.
        dominant_idx_next = max(
            range(len(self.WEIGHT_VALUES_NEXT)),
            key=lambda k: self.WEIGHT_VALUES_NEXT[k],
        )
        recolor_anims = []
        for j, cell in enumerate(weight_cells):
            w = self.WEIGHT_VALUES_NEXT[j]
            opacity = 0.12 + 0.80 * w / max(self.WEIGHT_VALUES_NEXT)
            recolor_anims.append(cell.animate.set_fill(ORANGE, opacity=opacity))
            recolor_anims.append(
                weight_labels[j].animate.become(
                    MathTex(rf"{w:.2f}")
                    .scale(0.38)
                    .move_to(weight_labels[j].get_center())
                )
            )
        self.play(*recolor_anims, run_time=0.7)

        # New dominant cell highlight.
        dom_cell_next = weight_cells[dominant_idx_next]
        dom_box_next = Rectangle(
            width=dom_cell_next.width + 0.10,
            height=dom_cell_next.height + 0.10,
            color=YELLOW,
            stroke_width=3,
        ).move_to(dom_cell_next.get_center())
        self.play(Create(dom_box_next), run_time=0.3)
        self.wait(0.4)

        # ---------------- Final caption ----------------
        caption = (
            MathTex(
                r"\text{attention: decoder freely reads any source position}"
            )
            .scale(0.50)
            .move_to([0.0, self.Y_CAPTION, 0.0])
        )
        self.play(FadeIn(caption))
        self.wait(0.8)
