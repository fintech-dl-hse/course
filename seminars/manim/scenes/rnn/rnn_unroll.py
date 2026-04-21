"""RNN unrolling scene for seminar 09.

Анимирует разворачивание простой RNN (Элман-ячейка) на три шага по времени:
    h_t = tanh(W_{ih} x_t + W_{hh} h_{t-1})

Сцена импортирует переиспользуемые примитивы из ``shared.neural`` (Neuron,
LabeledBox, arrow_between), проверяя, что библиотека общих компонент работает
на реальном пилоте (принцип 2 консенсус-плана).
"""
from __future__ import annotations

from manim import (
    Create,
    DOWN,
    FadeIn,
    LEFT,
    MathTex,
    RIGHT,
    Scene,
    UP,
    Write,
)

from shared.neural import Neuron, LabeledBox, arrow_between


class RNNUnroll(Scene):
    """Разворачивает RNN на три шага по времени с подсветкой рекуррентной связи."""

    def construct(self) -> None:
        title = MathTex(r"h_t = \tanh(W_{ih} x_t + W_{hh} h_{t-1})").to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        h_prev = Neuron(label="h_0").shift(LEFT * 4.5 + DOWN * 0.5)
        h_prev_label = MathTex(r"h_0").scale(0.6).next_to(h_prev, DOWN, buff=0.15)
        self.play(FadeIn(h_prev), FadeIn(h_prev_label))

        timesteps = [1, 2, 3]
        x_spacing = 2.2
        prev_h = h_prev

        for i, t in enumerate(timesteps):
            x_pos = -4.6 + (i + 1) * x_spacing
            x_t = Neuron(label=f"x_{t}").shift(RIGHT * x_pos + DOWN * 3.0)
            h_t = Neuron(label=f"h_{t}").shift(RIGHT * x_pos + DOWN * 0.5)

            w_ih = LabeledBox(label="W_{ih}", width=0.9, height=0.55).shift(
                RIGHT * x_pos + DOWN * 1.75
            )
            w_hh = LabeledBox(label="W_{hh}", width=0.9, height=0.55).move_to(
                (prev_h.get_center() + h_t.get_center()) / 2
            )

            a1 = arrow_between(x_t, w_ih)
            a2 = arrow_between(w_ih, h_t)
            a3 = arrow_between(prev_h, w_hh)
            a4 = arrow_between(w_hh, h_t)

            self.play(FadeIn(x_t), run_time=0.35)
            self.play(Create(a1), FadeIn(w_ih), Create(a2), run_time=0.7)
            self.play(Create(a3), FadeIn(w_hh), Create(a4), run_time=0.7)
            self.play(FadeIn(h_t), run_time=0.35)

            prev_h = h_t

        self.wait(1.2)

        caption = MathTex(r"\text{RNN unrolled across 3 timesteps}").scale(0.55).to_edge(DOWN, buff=0.15)
        self.play(FadeIn(caption))
        self.wait(1.0)
