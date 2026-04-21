"""LSTM Gates scene for seminar materials.

Анимирует LSTM-ячейку на три шага по времени, показывая:
    f_t = sigma(W_f [h_{t-1}, x_t] + b_f)   -- forget gate
    i_t = sigma(W_i [h_{t-1}, x_t] + b_i)   -- input gate
    g_t = tanh(W_g [h_{t-1}, x_t] + b_g)    -- candidate cell update
    o_t = sigma(W_o [h_{t-1}, x_t] + b_o)   -- output gate
    c_t = f_t * c_{t-1} + i_t * g_t          -- cell state
    h_t = o_t * tanh(c_t)                     -- hidden state

Сцена использует Neuron для gate-активаций и LabeledBox для
весовых/нелинейных блоков. Раскладка горизонтальная по шагам времени,
аналогично scenes/rnn/rnn_unroll.py.
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
    VGroup,
    Write,
)

from shared.keyframes import KeyframeRecorder
from shared.neural import Neuron, LabeledBox, arrow_between


class LSTMGates(KeyframeRecorder, Scene):
    """LSTM cell unrolled across 3 timesteps with gate activations and cell state."""

    def construct(self) -> None:
        # --- Title equations (split across two lines to keep LaTeX simple) ---
        eq_top = MathTex(
            r"f_t = \sigma(W_f x_t + U_f h_{t-1})",
            r"\quad i_t = \sigma(W_i x_t)",
            r"\quad g_t = \tanh(W_g x_t)",
        ).scale(0.45)
        eq_bot = MathTex(
            r"c_t = f_t \odot c_{t-1} + i_t \odot g_t",
            r"\quad h_t = o_t \odot \tanh(c_t)",
        ).scale(0.45)
        equations = VGroup(eq_top, eq_bot).arrange(DOWN, buff=0.12).to_edge(UP, buff=0.15)
        self.play(Write(equations))
        self.wait(0.5)

        # --- Initial states ---
        h_prev = Neuron(label="h_0").shift(LEFT * 5.2 + DOWN * 0.6)
        c_prev = Neuron(label="c_0").shift(LEFT * 5.2 + DOWN * 2.4)
        h_label = MathTex(r"h_0").scale(0.55).next_to(h_prev, DOWN, buff=0.12)
        c_label = MathTex(r"c_0").scale(0.55).next_to(c_prev, DOWN, buff=0.12)
        self.play(FadeIn(h_prev), FadeIn(c_prev), FadeIn(h_label), FadeIn(c_label))

        timesteps = [1, 2, 3]
        x_spacing = 3.0
        prev_h = h_prev
        prev_c = c_prev

        for i, t in enumerate(timesteps):
            x_pos = -5.2 + (i + 1) * x_spacing

            # Input neuron
            x_t = Neuron(label=f"x_{t}").shift(RIGHT * x_pos + DOWN * 4.2)

            # Gate LabeledBoxes (forget, input, candidate, output)
            f_box = LabeledBox(label=r"\sigma_f", width=0.85, height=0.55).shift(
                RIGHT * x_pos + DOWN * 0.6
            )
            i_box = LabeledBox(label=r"\sigma_i", width=0.85, height=0.55).shift(
                RIGHT * x_pos + DOWN * 1.5
            )
            g_box = LabeledBox(label=r"\tanh_g", width=0.85, height=0.55).shift(
                RIGHT * x_pos + DOWN * 2.4
            )
            o_box = LabeledBox(label=r"\sigma_o", width=0.85, height=0.55).shift(
                RIGHT * x_pos + DOWN * 3.3
            )

            # Gate activation neurons
            f_t = Neuron(label=f"f_{t}", radius=0.32).shift(
                RIGHT * (x_pos + 1.1) + DOWN * 0.6
            )
            i_t = Neuron(label=f"i_{t}", radius=0.32).shift(
                RIGHT * (x_pos + 1.1) + DOWN * 1.5
            )
            g_t = Neuron(label=f"g_{t}", radius=0.32).shift(
                RIGHT * (x_pos + 1.1) + DOWN * 2.4
            )
            o_t = Neuron(label=f"o_{t}", radius=0.32).shift(
                RIGHT * (x_pos + 1.1) + DOWN * 3.3
            )

            # Cell state and hidden state outputs
            c_t = Neuron(label=f"c_{t}", radius=0.36).shift(
                RIGHT * (x_pos + 2.2) + DOWN * 1.5
            )
            h_t = Neuron(label=f"h_{t}").shift(
                RIGHT * (x_pos + 2.2) + DOWN * 0.6
            )

            # --- Step 1: show input x_t and gates ---
            self.play(FadeIn(x_t), run_time=0.3)
            self.play(
                FadeIn(f_box), FadeIn(i_box), FadeIn(g_box), FadeIn(o_box),
                run_time=0.4,
            )

            # Arrows: x_t -> each gate box
            ax_f = arrow_between(x_t, f_box)
            ax_i = arrow_between(x_t, i_box)
            ax_g = arrow_between(x_t, g_box)
            ax_o = arrow_between(x_t, o_box)
            self.play(
                Create(ax_f), Create(ax_i), Create(ax_g), Create(ax_o),
                run_time=0.5,
            )

            # Arrows: h_{t-1} -> gate boxes (recurrent connection)
            ah_f = arrow_between(prev_h, f_box)
            ah_i = arrow_between(prev_h, i_box)
            self.play(Create(ah_f), Create(ah_i), run_time=0.4)

            # --- Step 2: gate activation neurons ---
            af_out = arrow_between(f_box, f_t)
            ai_out = arrow_between(i_box, i_t)
            ag_out = arrow_between(g_box, g_t)
            ao_out = arrow_between(o_box, o_t)
            self.play(
                FadeIn(f_t), FadeIn(i_t), FadeIn(g_t), FadeIn(o_t),
                Create(af_out), Create(ai_out), Create(ag_out), Create(ao_out),
                run_time=0.5,
            )

            # --- Step 3: cell state update c_t ---
            # f_t -> c_t (forget gate modulates c_{t-1})
            ac_f = arrow_between(f_t, c_t)
            ac_prev = arrow_between(prev_c, c_t)
            ac_i = arrow_between(i_t, c_t)
            ac_g = arrow_between(g_t, c_t)
            self.play(
                FadeIn(c_t),
                Create(ac_f), Create(ac_prev), Create(ac_i), Create(ac_g),
                run_time=0.6,
            )

            # --- Step 4: hidden state h_t ---
            ah_o = arrow_between(o_t, h_t)
            ah_c = arrow_between(c_t, h_t)
            self.play(FadeIn(h_t), Create(ah_o), Create(ah_c), run_time=0.4)

            prev_h = h_t
            prev_c = c_t

        self.wait(1.5)

        caption = (
            MathTex(r"\text{LSTM unrolled across 3 timesteps}")
            .scale(0.5)
            .to_edge(DOWN, buff=0.15)
        )
        self.play(FadeIn(caption))
        self.wait(1.0)
