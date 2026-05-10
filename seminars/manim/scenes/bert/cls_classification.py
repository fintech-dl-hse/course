"""BERT [CLS] classification head scene (V11).

Демонстрирует, как токен ``[CLS]`` агрегирует информацию о всём предложении
и используется как «вектор предложения» для классификации.

Сценарий по фазам:

1. Заголовок: ``Classification: [CLS] -> MLP -> P(class)``.
2. Строка из 5 токенов с предпосылкой ``[CLS]``: ``["[CLS]", "the", "movie",
   "was", "great"]``. Токен ``[CLS]`` выделен золотой обводкой.
3. Каждый токен превращается в эмбеддинг ``x_i`` (TensorColumn dim=4) с
   аннотацией ``+PE_i`` рядом.
4. Эмбеддинги идут в стек из 3 блоков энкодера (``Encoder Block x N``).
5. Контекстные скрытые состояния ``h_[CLS], h_1, h_2, h_3, h_4``;
   ``h_[CLS]`` ярко выделен, остальные приглушены.
6. ``h_[CLS]`` извлекается и проходит через MLP (Linear -> tanh -> Linear).
7. После softmax — bar-chart из 3 классов (positive / neutral / negative),
   argmax-бар подсвечен.
8. Финальная подпись: pooling at [CLS] gives a single sentence embedding for
   classification.

Используются примитивы ``shared.neural`` (TensorColumn, LabeledBox,
arrow_between). Локальный хелпер ``_horizontal_arrow`` мирорит
``rnn_forward.py`` — нужен, когда требуется принудительно горизонтальное
крепление стрелки.
"""
from __future__ import annotations

from typing import Any

from manim import (
    Arrow,
    BLUE,
    Create,
    DOWN,
    FadeIn,
    GREEN,
    GREY_B,
    LEFT,
    Line,
    MathTex,
    ORANGE,
    PURPLE,
    RIGHT,
    Rectangle,
    Scene,
    TEAL,
    Tex,
    UP,
    VGroup,
    VMobject,
    WHITE,
    Write,
)

from shared.neural import LabeledBox, TensorColumn, arrow_between


CLS_GOLD = "#F5C518"


def _horizontal_arrow(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    """Стрелка, всегда прилипающая к правому/левому краю объектов.

    Локальная копия (см. ``rnn_forward.py:45-58``).
    """
    defaults: dict[str, Any] = {"buff": 0.08, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


def _vertical_arrow(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    """Стрелка, всегда прилипающая к верхнему/нижнему краю объектов."""
    defaults: dict[str, Any] = {"buff": 0.06, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[1] >= b.get_center()[1]:
        start, end = a.get_bottom(), b.get_top()
    else:
        start, end = a.get_top(), b.get_bottom()
    return Arrow(start=start, end=end, **defaults)


class BERTClsClassification(Scene):
    """V11: BERT [CLS] sentence-level classification head."""

    # ---- Layout constants — 720p (-qm) frame ±(7.11, 4.0) ----
    TENSOR_DIM = 4
    CELL = 0.22
    NUM_TOKENS = 5  # including [CLS]

    # Vertical anchors.
    Y_TITLE = 3.55
    Y_TOKENS = 2.70           # token strings
    Y_EMBED = 1.55            # x_i tensor columns
    Y_PE = 0.95               # +PE_i annotation row (centered between embed and stack)
    Y_STACK = 0.10            # encoder block stack center
    Y_HIDDEN = -1.05          # h_i tensor columns
    Y_MLP = -2.20             # MLP boxes

    # Token x-positions: spread across [-5.4, +5.4].
    X_LEFT = -5.40
    X_RIGHT = 5.40

    # Encoder stack (single visible "× N" stack on the right side of the diagram).
    X_STACK = 0.0
    STACK_W = 1.85
    STACK_H = 0.55
    STACK_DY = 0.13           # offset between three layered boxes

    # MLP boxes — placed on the right of the hidden row, below h_[CLS] but
    # routed via a vertical drop then horizontal sweep.
    X_MLP_LIN1 = -3.40
    X_MLP_TANH = -1.55
    X_MLP_LIN2 = 0.30
    MLP_W = 1.35
    MLP_H = 0.55

    # Bar chart geometry. Bars are placed to the right of the MLP+softmax
    # chain on the same row.
    BAR_W = 0.42
    BAR_GAP = 0.30
    BAR_X_CENTER = 5.50
    BAR_Y_BASE = -2.50
    BAR_HEIGHTS = (0.55, 0.22, 0.36)  # positive / neutral / negative
    BAR_LABELS = ("pos", "neu", "neg")

    def _token_x(self, i: int) -> float:
        if self.NUM_TOKENS == 1:
            return 0.0
        return self.X_LEFT + (self.X_RIGHT - self.X_LEFT) * i / (self.NUM_TOKENS - 1)

    def construct(self) -> None:
        # ===================== Phase 0: title =====================
        title = MathTex(
            r"\text{Classification: } [\mathrm{CLS}] \rightarrow \mathrm{MLP} "
            r"\rightarrow P(\mathrm{class})"
        ).scale(0.60).move_to([0.0, self.Y_TITLE, 0.0])
        self.play(Write(title), run_time=0.6)
        self.wait(0.2)

        # ===================== Phase 1: tokens =====================
        tokens = ["[CLS]", "the", "movie", "was", "great"]
        token_mobs: list[VGroup] = []
        for i, tok in enumerate(tokens):
            x = self._token_x(i)
            if i == 0:
                # [CLS] gets a gold border; using MathTex for monospaced look.
                lbl = MathTex(r"[\mathrm{CLS}]").scale(0.55)
                lbl.set_color(CLS_GOLD)
                border = Rectangle(
                    width=lbl.width + 0.18,
                    height=lbl.height + 0.16,
                    color=CLS_GOLD,
                    stroke_width=2.5,
                ).set_fill(CLS_GOLD, opacity=0.10)
                grp = VGroup(border, lbl)
                grp.move_to([x, self.Y_TOKENS, 0.0])
            else:
                lbl = (
                    Tex(rf"\textit{{``{tok}''}}")
                    .scale(0.55)
                    .move_to([x, self.Y_TOKENS, 0.0])
                )
                grp = VGroup(lbl)
            token_mobs.append(grp)

        self.play(*[FadeIn(g) for g in token_mobs], run_time=0.5)
        self.wait(0.2)

        # ===================== Phase 2: embeddings + PE annotations =====================
        embed_cols: list[TensorColumn] = []
        pe_labels: list[MathTex] = []
        embed_arrows: list[Arrow] = []

        for i in range(self.NUM_TOKENS):
            x = self._token_x(i)
            emb_color = CLS_GOLD if i == 0 else BLUE
            emb_op = 0.55 if i == 0 else 0.35
            col = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL,
                color=emb_color,
                fill_opacity=emb_op,
            ).move_to([x, self.Y_EMBED, 0.0])
            embed_cols.append(col)

            # Token -> embedding arrow (vertical, short).
            arr = _vertical_arrow(
                token_mobs[i], col, buff=0.08, tip_length=0.10, stroke_width=2.0
            )
            embed_arrows.append(arr)

            pe = (
                MathTex(rf"+\,PE_{i}")
                .scale(0.55)
                .next_to(col, RIGHT, buff=0.08)
            )
            pe_labels.append(pe)

        self.play(
            *[FadeIn(c) for c in embed_cols],
            *[Create(a) for a in embed_arrows],
            run_time=0.6,
        )
        self.play(*[FadeIn(p) for p in pe_labels], run_time=0.4)
        self.wait(0.2)

        # ===================== Phase 3: encoder block stack =====================
        # Visualize a stack of 3 boxes with "Encoder Block x N" label at center.
        # Place them with slight diagonal offset so the "stack" reads.
        # Use Rectangle (not LabeledBox) for the back/mid layers so they don't
        # carry stray tiny MathTex submobjects, and set explicit z-indices so
        # the front (labeled) box always renders on top.
        from manim import RoundedRectangle  # local import — keeps top-level tidy
        back_rect = RoundedRectangle(
            corner_radius=0.12,
            width=self.STACK_W, height=self.STACK_H,
            color=PURPLE, stroke_width=2,
        ).set_fill(PURPLE, opacity=0.10)
        back_rect.move_to([self.X_STACK + 2 * self.STACK_DY, self.Y_STACK + 2 * self.STACK_DY, 0.0])
        back_rect.set_z_index(0)

        mid_rect = RoundedRectangle(
            corner_radius=0.12,
            width=self.STACK_W, height=self.STACK_H,
            color=PURPLE, stroke_width=2,
        ).set_fill(PURPLE, opacity=0.15)
        mid_rect.move_to([self.X_STACK + self.STACK_DY, self.Y_STACK + self.STACK_DY, 0.0])
        mid_rect.set_z_index(1)

        front_box = LabeledBox(
            label=r"\text{Encoder Block} \times N",
            width=self.STACK_W,
            height=self.STACK_H,
            color=PURPLE,
            label_scale=0.52,
            fill_opacity=0.95,
        ).move_to([self.X_STACK, self.Y_STACK, 0.0])
        front_box.set_z_index(2)

        # Single big arrow from the *embedding row* center to the front box top.
        # We model it as a vertical arrow from the center embedding column down.
        # Use the middle embed column as the source anchor for the centered arrow.
        center_emb = embed_cols[self.NUM_TOKENS // 2]
        arr_emb_to_stack = _vertical_arrow(
            center_emb, front_box, buff=0.08, tip_length=0.14,
            stroke_width=2.5, color=PURPLE,
        )

        # Animate from back to front so layering reads naturally.
        self.play(FadeIn(back_rect), run_time=0.20)
        self.play(FadeIn(mid_rect), run_time=0.20)
        self.play(FadeIn(front_box), run_time=0.30)
        self.play(Create(arr_emb_to_stack), run_time=0.3)
        self.wait(0.2)

        # ===================== Phase 4: hidden states =====================
        # Output row: 5 contextual hidden tensors. h_[CLS] is bright gold,
        # others are grey/dim.
        hidden_cols: list[TensorColumn] = []
        hidden_labels: list[MathTex] = []
        for i in range(self.NUM_TOKENS):
            x = self._token_x(i)
            if i == 0:
                color = CLS_GOLD
                op = 0.85
            else:
                color = GREY_B
                op = 0.30
            col = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL,
                color=color,
                fill_opacity=op,
            ).move_to([x, self.Y_HIDDEN, 0.0])
            hidden_cols.append(col)

            if i == 0:
                lbl_text = r"h_{[\mathrm{CLS}]}"
                lbl_scale = 0.58
                lbl_color = CLS_GOLD
            else:
                lbl_text = rf"h_{i}"
                lbl_scale = 0.58
                lbl_color = GREY_B
            lbl = MathTex(lbl_text).scale(lbl_scale)
            lbl.set_color(lbl_color)
            lbl.next_to(col, DOWN, buff=0.10)
            hidden_labels.append(lbl)

        # Single arrow from front box bottom to the center of the hidden row.
        center_hidden = hidden_cols[self.NUM_TOKENS // 2]
        arr_stack_to_hidden = _vertical_arrow(
            front_box, center_hidden, buff=0.08, tip_length=0.14,
            stroke_width=2.5, color=PURPLE,
        )

        self.play(Create(arr_stack_to_hidden), run_time=0.3)
        self.play(
            *[FadeIn(c) for c in hidden_cols],
            *[FadeIn(l) for l in hidden_labels],
            run_time=0.6,
        )
        self.wait(0.3)

        # ===================== Phase 5: pluck h_[CLS] -> MLP =====================
        # MLP: three boxes (Linear_1, tanh, Linear_2). Place them on the
        # MLP row, horizontally arranged.
        mlp_lin1 = LabeledBox(
            label=r"\mathrm{Linear}",
            width=self.MLP_W, height=self.MLP_H,
            color=ORANGE, label_scale=0.55, fill_opacity=0.20,
        ).move_to([self.X_MLP_LIN1, self.Y_MLP, 0.0])
        mlp_tanh = LabeledBox(
            label=r"\tanh",
            width=self.MLP_W, height=self.MLP_H,
            color=ORANGE, label_scale=0.60, fill_opacity=0.20,
        ).move_to([self.X_MLP_TANH, self.Y_MLP, 0.0])
        mlp_lin2 = LabeledBox(
            label=r"\mathrm{Linear}",
            width=self.MLP_W, height=self.MLP_H,
            color=ORANGE, label_scale=0.55, fill_opacity=0.20,
        ).move_to([self.X_MLP_LIN2, self.Y_MLP, 0.0])

        # h_[CLS] is at left side (x = X_LEFT). MLP starts at X_MLP_LIN1 = -2.40.
        # We need a path: h_[CLS] -> down then right to Linear_1.
        # Use an L-shape with a Line + terminal Arrow.
        cls_col = hidden_cols[0]
        cls_label = hidden_labels[0]
        elbow_y = self.Y_MLP
        elbow_x = float(cls_label.get_bottom()[0])
        cls_anchor = [
            float(cls_label.get_bottom()[0]),
            float(cls_label.get_bottom()[1]) - 0.05,
            0.0,
        ]

        l_down = Line(
            start=cls_anchor,
            end=[elbow_x, elbow_y, 0.0],
            color=CLS_GOLD,
            stroke_width=2.5,
        )
        # Final segment: arrow to the left edge of mlp_lin1.
        arr_to_lin1 = Arrow(
            start=[elbow_x, elbow_y, 0.0],
            end=mlp_lin1.get_left() + 0.0,
            buff=0.05,
            tip_length=0.14,
            color=CLS_GOLD,
            stroke_width=2.5,
        )

        # MLP internal arrows.
        arr_lin1_tanh = _horizontal_arrow(
            mlp_lin1, mlp_tanh, buff=0.05, tip_length=0.12,
            color=ORANGE, stroke_width=2.0,
        )
        arr_tanh_lin2 = _horizontal_arrow(
            mlp_tanh, mlp_lin2, buff=0.05, tip_length=0.12,
            color=ORANGE, stroke_width=2.0,
        )

        self.play(
            Create(l_down), Create(arr_to_lin1),
            FadeIn(mlp_lin1),
            run_time=0.5,
        )
        self.play(
            FadeIn(mlp_tanh), Create(arr_lin1_tanh),
            run_time=0.4,
        )
        self.play(
            FadeIn(mlp_lin2), Create(arr_tanh_lin2),
            run_time=0.4,
        )
        self.wait(0.2)

        # ===================== Phase 6: softmax + bar chart =====================
        # Softmax label sits in a small box right of Linear_2.
        softmax_box = LabeledBox(
            label=r"\mathrm{softmax}",
            width=1.30, height=self.MLP_H,
            color=ORANGE, label_scale=0.55, fill_opacity=0.20,
        ).move_to([self.X_MLP_LIN2 + self.MLP_W / 2.0 + 0.10 + 1.30 / 2.0, self.Y_MLP, 0.0])
        # Tiny arrow from mlp_lin2 to softmax_box.
        arr_lin2_softmax = _horizontal_arrow(
            mlp_lin2, softmax_box, buff=0.05, tip_length=0.12,
            color=ORANGE, stroke_width=2.0,
        )

        # Bar chart: 3 vertical Rectangles + class labels below.
        # Bars sit at base y = BAR_Y_BASE; tops vary by BAR_HEIGHTS.
        # Center the bars at BAR_X_CENTER.
        n_bars = len(self.BAR_HEIGHTS)
        base_x = self.BAR_X_CENTER - (n_bars - 1) * (self.BAR_W + self.BAR_GAP) / 2.0
        bars: list[Rectangle] = []
        bar_labels: list[Tex] = []
        argmax_idx = max(range(n_bars), key=lambda k: self.BAR_HEIGHTS[k])
        for k in range(n_bars):
            bx = base_x + k * (self.BAR_W + self.BAR_GAP)
            h = self.BAR_HEIGHTS[k]
            color = CLS_GOLD if k == argmax_idx else GREY_B
            opacity = 0.85 if k == argmax_idx else 0.40
            rect = Rectangle(
                width=self.BAR_W, height=h,
                color=color, stroke_width=2,
            ).set_fill(color, opacity=opacity)
            # bottom edge at BAR_Y_BASE; center is at BAR_Y_BASE + h/2.
            rect.move_to([bx, self.BAR_Y_BASE + h / 2.0, 0.0])
            bars.append(rect)

            lbl = (
                Tex(self.BAR_LABELS[k])
                .scale(0.80)
                .next_to(rect, DOWN, buff=0.06)
            )
            bar_labels.append(lbl)

        # Horizontal arrow from softmax box to the bar chart (left side of
        # leftmost bar).
        leftmost_bar = bars[0]
        arr_softmax_to_bars = _horizontal_arrow(
            softmax_box, leftmost_bar, buff=0.10, tip_length=0.12,
            color=ORANGE, stroke_width=2.0,
        )

        self.play(
            FadeIn(softmax_box), Create(arr_lin2_softmax),
            run_time=0.4,
        )
        self.play(
            Create(arr_softmax_to_bars),
            *[FadeIn(b) for b in bars],
            *[FadeIn(l) for l in bar_labels],
            run_time=0.6,
        )
        self.wait(0.6)
