"""RNN forward-pass language-model scene for seminar 09.

Анимирует прямой проход RNN-языковой модели на трёх токенах
(``the``, ``cat``, ``sat``). На каждом шаге t показывает три уровня смысла:

1. Токен (строка вверху) — один из реальных слов входной последовательности.
2. Тензорный поток (середина) — токен превращается в абстрактный embedding
   ``x_t`` (4 ячейки) и через матрицы ``W_{ih}``/``W_{hh}`` смешивается с
   предыдущим скрытым состоянием в ``h_t``.
3. Предсказание следующего токена (низ) — ``h_t`` проектируется через
   ``W_{ho}`` в softmax-выход ``y_t`` (4 ячейки, одна подсвечена как
   argmax) и в подписанный «следующий токен».

Сцена использует общие примитивы ``shared.neural`` (Neuron, LabeledBox,
TensorColumn, arrow_between) и не пересекается с уже существующей
``RNNUnroll``: это родственная сцена, демонстрирующая ту же топологию,
но в виде потока тензоров языковой модели.
"""
from __future__ import annotations

from typing import Any

from manim import (
    Arrow,
    Create,
    DOWN,
    FadeIn,
    GREEN,
    LEFT,
    MathTex,
    ORANGE,
    RIGHT,
    Scene,
    Tex,
    UP,
    VGroup,
    VMobject,
    WHITE,
    Write,
)

from shared.neural import LabeledBox, TensorColumn, arrow_between


def _horizontal_arrow(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    """Стрелка, всегда прилипающая к правому/левому краю объектов.

    Нужна, когда ``arrow_between`` выбрал бы вертикальное крепление (dy > dx)
    и прошёл бы через посторонний узел. Сценарий тот же, что и в
    ``LSTMGates._horizontal_arrow``.
    """
    defaults: dict[str, Any] = {"buff": 0.1, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


class RNNForward(Scene):
    """RNN языковая модель на 3 шагах: tokens → x_t → h_t → y_t → next-token."""

    # Layout constants — tuned for 720p (-qm) frame ±(7.11, 4.0).
    CELL_SIZE = 0.27
    TENSOR_DIM = 4

    # Vertical anchors (y-coordinates) for each conceptual row.
    Y_TITLE = 3.55
    Y_TOKEN = 2.95  # input-token text
    Y_X = 2.15  # x_t TensorColumn center
    Y_WIH = 1.10  # W_ih label-box
    Y_H = 0.10  # h_t TensorColumn center
    Y_WHO = -0.90  # W_ho label-box
    Y_Y = -1.85  # y_t TensorColumn center
    Y_PRED = -2.95  # predicted next-token text

    # Horizontal layout.
    X_BASE = -5.55  # x of h_0 (anchor for the recurrence)
    X_SPACING = 3.55  # spacing between consecutive timesteps
    BOX_W = 0.85
    BOX_H = 0.46

    def construct(self) -> None:
        # --- Title (compact two-line equation) ---
        eq_line1 = MathTex(
            r"x_t = E[\text{token}_t]",
            r"\quad h_t = \tanh(W_{ih}\, x_t + W_{hh}\, h_{t-1})",
        ).scale(0.55)
        eq_line2 = MathTex(
            r"y_t = \mathrm{softmax}(W_{ho}\, h_t)",
            r"\quad \text{next}_t = \arg\max_v y_t[v]",
        ).scale(0.55)
        equations = (
            VGroup(eq_line1, eq_line2)
            .arrange(DOWN, buff=0.14)
            .move_to([0.0, self.Y_TITLE, 0.0])
        )
        self.play(Write(equations))
        self.wait(0.3)

        # --- h_0: initial hidden state, just a tensor column with a label ---
        h_prev = TensorColumn(
            dim=self.TENSOR_DIM,
            cell_size=self.CELL_SIZE,
            label_scale=0.55,
        ).move_to([self.X_BASE, self.Y_H, 0.0])
        h_prev_lbl = MathTex(r"h_0").scale(0.6).next_to(h_prev, DOWN, buff=0.12)
        self.play(FadeIn(h_prev), FadeIn(h_prev_lbl))
        self.wait(0.2)

        tokens = ["the", "cat", "sat"]
        # The "predicted next token" for the last step is unknown to the model
        # (we don't show ground-truth past the sequence), so it stays "?".
        pred_tokens = ["cat", "sat", "?"]
        argmax_indices = [1, 2, 3]  # which output cell is the argmax per step

        prev_h = h_prev

        for i, token in enumerate(tokens):
            t = i + 1
            x_pos = self.X_BASE + t * self.X_SPACING

            # --- Row 0: input token (Tex string) ---
            tok = (
                Tex(rf"\textit{{``{token}''}}")
                .scale(0.65)
                .move_to([x_pos, self.Y_TOKEN, 0.0])
            )

            # --- Row 1: embedding tensor x_t with label to its right ---
            x_t = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL_SIZE,
                label=f"x_{t}",
                label_scale=0.55,
            ).move_to([x_pos, self.Y_X, 0.0])

            # --- W_ih label-box between x_t and h_t (orange = input→hidden) ---
            w_ih = LabeledBox(
                label="W_{ih}",
                width=self.BOX_W,
                height=self.BOX_H,
                label_scale=0.55,
                color=ORANGE,
            ).move_to([x_pos, self.Y_WIH, 0.0])

            # --- Row 2: hidden tensor h_t ---
            h_t = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL_SIZE,
                label=f"h_{t}",
                label_scale=0.55,
            ).move_to([x_pos, self.Y_H, 0.0])

            # --- W_ho label-box between h_t and y_t ---
            w_ho = LabeledBox(
                label="W_{ho}", width=self.BOX_W, height=self.BOX_H, label_scale=0.55
            ).move_to([x_pos, self.Y_WHO, 0.0])

            # --- Row 4: output softmax tensor y_t (argmax cell highlighted) ---
            y_t = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL_SIZE,
                label=f"y_{t}",
                label_scale=0.55,
                highlight_index=argmax_indices[i] % self.TENSOR_DIM,
            ).move_to([x_pos, self.Y_Y, 0.0])

            # --- Row 5: predicted next-token label ---
            pred = (
                Tex(rf"\textit{{``{pred_tokens[i]}''}}\,'")
                .scale(0.6)
                .move_to([x_pos, self.Y_PRED, 0.0])
            )

            # ---- Animation step 1: token → embedding x_t ----
            tok_to_x = arrow_between(tok, x_t, buff=0.12, tip_length=0.14)
            self.play(FadeIn(tok), run_time=0.3)
            self.play(FadeIn(x_t), Create(tok_to_x), run_time=0.5)

            # ---- Animation step 2: x_t → W_ih → h_t and h_{t-1} → h_t (W_hh) ----
            x_to_wih = arrow_between(x_t, w_ih, buff=0.1, tip_length=0.13)
            wih_to_h = arrow_between(w_ih, h_t, buff=0.1, tip_length=0.13)
            self.play(FadeIn(w_ih), Create(x_to_wih), run_time=0.4)

            # Recurrent W_hh: a small label-box sitting on the horizontal
            # arrow from prev_h to h_t. Place it midway between the two
            # tensor columns, on the same y as h_t. We then build TWO arrows:
            # prev_h → w_hh and w_hh → h_t. Forced horizontal so the long
            # diagonal isn't routed top/bottom.
            w_hh_x = (prev_h.get_right()[0] + h_t.get_left()[0]) / 2.0
            w_hh = LabeledBox(
                label="W_{hh}",
                width=0.7,
                height=self.BOX_H,
                label_scale=0.5,
                color=GREEN,
            ).move_to([w_hh_x, self.Y_H, 0.0])
            a_prev_to_whh = _horizontal_arrow(
                prev_h, w_hh, buff=0.1, tip_length=0.13
            )
            a_whh_to_h = _horizontal_arrow(w_hh, h_t, buff=0.1, tip_length=0.13)

            self.play(
                FadeIn(w_hh),
                Create(a_prev_to_whh),
                Create(a_whh_to_h),
                Create(wih_to_h),
                FadeIn(h_t),
                run_time=0.7,
            )

            # ---- Animation step 3: h_t → W_ho → y_t ----
            h_to_who = arrow_between(h_t, w_ho, buff=0.1, tip_length=0.13)
            who_to_y = arrow_between(w_ho, y_t, buff=0.1, tip_length=0.13)
            self.play(
                FadeIn(w_ho),
                Create(h_to_who),
                Create(who_to_y),
                FadeIn(y_t),
                run_time=0.6,
            )

            # ---- Animation step 4: y_t → predicted next-token (argmax) ----
            y_to_pred = arrow_between(y_t, pred, buff=0.12, tip_length=0.14)
            self.play(FadeIn(pred), Create(y_to_pred), run_time=0.4)

            prev_h = h_t

        self.wait(1.0)

        caption = (
            MathTex(
                r"\text{RNN language model: tokens} \to x_t \to h_t \to y_t "
                r"\to \text{next token}"
            )
            .scale(0.45)
            .to_edge(DOWN, buff=0.08)
        )
        self.play(FadeIn(caption))
        self.wait(0.8)
