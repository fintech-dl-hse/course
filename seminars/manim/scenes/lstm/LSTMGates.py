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

from shared.neural import Neuron, LabeledBox, arrow_between


class LSTMGates(Scene):
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
        equations = VGroup(eq_top, eq_bot).arrange(DOWN, buff=0.14).to_edge(UP, buff=0.55)
        self.play(Write(equations))
        self.wait(0.5)

        # --- Initial states. No extra MathTex sublabels — Neuron already
        # renders h_0 / c_0 inside the circle.
        h_prev = Neuron(label="h_0").shift(LEFT * 6.0 + DOWN * 0.48)
        c_prev = Neuron(label="c_0").shift(LEFT * 6.0 + DOWN * 1.56)
        c_prev.set_z_index(2)
        self.play(FadeIn(h_prev), FadeIn(c_prev))

        timesteps = [1, 2, 3]
        # Layout per cell (relative to x_pos, where x_pos is the gate-box column):
        #   gate boxes         -> x_pos + 0.0 (width 0.85)
        #   gate neurons f/i/g/o -> x_pos + 0.95
        #   c_t / h_t column   -> x_pos + 2.0 (radius 0.4)
        # Cell spans roughly [x_pos-0.425, x_pos+2.4] = 2.825 units wide.
        # x_spacing = 3.6: gap between h_t right edge (x_pos+2.4) and next
        # sigma_f left edge (x_pos+3.6-0.425 = x_pos+3.175) = 0.775 units.
        x_spacing = 3.6
        # Base chosen so h_3 = -6.4 + 3*3.6 + 2.0 = 6.4, right edge 6.8 < 7.11.
        # h_0 at x=-6.4, left edge -6.8, margin 0.31 from -7.11.
        x_base = -6.4
        prev_h = h_prev
        prev_c = c_prev

        # Shared arrow tweak: shorter tip_length so arrowheads don't cover glyphs.
        tip_kw = {"tip_length": 0.17}

        for i, t in enumerate(timesteps):
            x_pos = x_base + (i + 1) * x_spacing

            # Input neuron
            x_t = Neuron(label=f"x_{t}").shift(RIGHT * x_pos + DOWN * 3.2)

            # Gate LabeledBoxes with z_index=2 so labels stay above recurrent arrows.
            f_box = LabeledBox(label=r"\sigma_f", width=0.85, height=0.55).shift(
                RIGHT * x_pos + DOWN * 0.48
            )
            i_box = LabeledBox(label=r"\sigma_i", width=0.85, height=0.55).shift(
                RIGHT * x_pos + DOWN * 1.2
            )
            g_box = LabeledBox(label=r"\tanh_g", width=0.85, height=0.55).shift(
                RIGHT * x_pos + DOWN * 1.92
            )
            o_box = LabeledBox(label=r"\sigma_o", width=0.85, height=0.55).shift(
                RIGHT * x_pos + DOWN * 2.64
            )
            for box in (f_box, i_box, g_box, o_box):
                box.set_z_index(2)

            # Gate activation neurons (smaller radius, close to the box).
            f_t = Neuron(label=f"f_{t}", radius=0.28).shift(
                RIGHT * (x_pos + 0.95) + DOWN * 0.48
            )
            i_t = Neuron(label=f"i_{t}", radius=0.28).shift(
                RIGHT * (x_pos + 0.95) + DOWN * 1.2
            )
            g_t = Neuron(label=f"g_{t}", radius=0.28).shift(
                RIGHT * (x_pos + 0.95) + DOWN * 1.92
            )
            o_t = Neuron(label=f"o_{t}", radius=0.28).shift(
                RIGHT * (x_pos + 0.95) + DOWN * 2.64
            )

            # Cell state and hidden state outputs. c_t at a row between i and g
            # (DOWN*1.56), radius 0.4 matches h_t so c_t doesn't bleed into the
            # neighbouring cell's gate column. z_index=3 keeps the glyph above
            # any converging arrow tip that clips the rim.
            c_t = Neuron(label=f"c_{t}", radius=0.4).shift(
                RIGHT * (x_pos + 2.0) + DOWN * 1.56
            )
            c_t.set_z_index(3)
            h_t = Neuron(label=f"h_{t}").shift(
                RIGHT * (x_pos + 2.0) + DOWN * 0.48
            )
            h_t.set_z_index(3)

            # --- Step 1: show input x_t and gates ---
            self.play(FadeIn(x_t), run_time=0.3)
            self.play(
                FadeIn(f_box), FadeIn(i_box), FadeIn(g_box), FadeIn(o_box),
                run_time=0.4,
            )

            # Arrows: x_t -> each gate box
            ax_f = arrow_between(x_t, f_box, **tip_kw)
            ax_i = arrow_between(x_t, i_box, **tip_kw)
            ax_g = arrow_between(x_t, g_box, **tip_kw)
            ax_o = arrow_between(x_t, o_box, **tip_kw)
            self.play(
                Create(ax_f), Create(ax_i), Create(ax_g), Create(ax_o),
                run_time=0.5,
            )

            # Arrows: h_{t-1} -> gate boxes (recurrent connection)
            ah_f = arrow_between(prev_h, f_box, **tip_kw)
            ah_i = arrow_between(prev_h, i_box, **tip_kw)
            self.play(Create(ah_f), Create(ah_i), run_time=0.4)

            # --- Step 2: gate activation neurons ---
            af_out = arrow_between(f_box, f_t, **tip_kw)
            ai_out = arrow_between(i_box, i_t, **tip_kw)
            ag_out = arrow_between(g_box, g_t, **tip_kw)
            ao_out = arrow_between(o_box, o_t, **tip_kw)
            self.play(
                FadeIn(f_t), FadeIn(i_t), FadeIn(g_t), FadeIn(o_t),
                Create(af_out), Create(ai_out), Create(ag_out), Create(ao_out),
                run_time=0.5,
            )

            # --- Step 3: cell state update c_t ---
            # Large buff + short tips so arrowheads stop well clear of the c_t
            # glyph when four arrows fan in at once. Radius 0.55 + buff 0.32
            # puts tips outside the label region; z_index=3 on c_t keeps the
            # glyph on top even if a tip drifts over the rim.
            converge_kw = {"buff": 0.32, "tip_length": 0.13}
            ac_f = arrow_between(f_t, c_t, **converge_kw)
            ac_prev = arrow_between(prev_c, c_t, **converge_kw)
            ac_i = arrow_between(i_t, c_t, **converge_kw)
            ac_g = arrow_between(g_t, c_t, **converge_kw)
            self.play(
                FadeIn(c_t),
                Create(ac_f), Create(ac_prev), Create(ac_i), Create(ac_g),
                run_time=0.6,
            )

            # --- Step 4: hidden state h_t ---
            ah_o = arrow_between(o_t, h_t, **tip_kw)
            ah_c = arrow_between(c_t, h_t, **tip_kw)
            self.play(FadeIn(h_t), Create(ah_o), Create(ah_c), run_time=0.4)

            prev_h = h_t
            prev_c = c_t

        self.wait(1.5)
