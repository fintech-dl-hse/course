"""LSTM Gates scene for seminar materials.

Анимирует LSTM-ячейку на три шага по времени, показывая:
    f_t = sigma(W_f [h_{t-1}, x_t] + b_f)   -- forget gate
    i_t = sigma(W_i [h_{t-1}, x_t] + b_i)   -- input gate
    g_t = tanh(W_g [h_{t-1}, x_t] + b_g)    -- candidate cell update
    o_t = sigma(W_o [h_{t-1}, x_t] + b_o)   -- output gate
    c_t = f_t * c_{t-1} + i_t * g_t          -- cell state
    h_t = o_t * tanh(c_t)                     -- hidden state

Каждый гейт получает один и тот же объединённый вход [h_{t-1}, x_t];
в диаграмме это отражено явным concat-узлом, который собирает h_{t-1} и x_t
перед раздачей по четырём воротам. Такая топология устраняет пересечения
стрелок, которые неизбежны при прямой раздаче x_t во все gate-боксы.
"""
from __future__ import annotations

from typing import Any

from manim import (
    Arrow,
    Create,
    DOWN,
    FadeIn,
    LEFT,
    MathTex,
    RIGHT,
    Scene,
    UP,
    VGroup,
    VMobject,
    WHITE,
    Write,
)

from shared.neural import Neuron, LabeledBox, arrow_between


def _horizontal_arrow(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    """Стрелка, которая всегда прилипает к правому/левому краю объектов.

    Используется когда arrow_between выбрал бы вертикальное крепление
    (dy > dx) и прошёл бы сквозь промежуточный узел.
    """
    defaults: dict[str, Any] = {"buff": 0.1, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


class LSTMGates(Scene):
    """LSTM cell unrolled across 3 timesteps with gate activations and cell state."""

    def construct(self) -> None:
        # --- Title equations (three lines, scaled for readability at 720p) ---
        eq_line1 = MathTex(
            r"f_t = \sigma(W_f [h_{t-1}, x_t])",
            r"\quad i_t = \sigma(W_i [h_{t-1}, x_t])",
        ).scale(0.55)
        eq_line2 = MathTex(
            r"g_t = \tanh(W_g [h_{t-1}, x_t])",
            r"\quad o_t = \sigma(W_o [h_{t-1}, x_t])",
        ).scale(0.55)
        eq_line3 = MathTex(
            r"c_t = f_t \odot c_{t-1} + i_t \odot g_t",
            r"\quad h_t = o_t \odot \tanh(c_t)",
        ).scale(0.55)
        equations = VGroup(eq_line1, eq_line2, eq_line3).arrange(DOWN, buff=0.14).to_edge(UP, buff=0.3)
        self.play(Write(equations))
        self.wait(0.4)

        # Push the diagram up to eat the empty band between title and h_0.
        y_offset = 0.9

        # --- Initial states ---
        h_prev = Neuron(label="h_0").shift(LEFT * 6.0 + DOWN * (0.5 - y_offset))
        c_prev = Neuron(label="c_0").shift(LEFT * 6.0 + DOWN * (1.7 - y_offset))
        h_prev.set_z_index(3)
        c_prev.set_z_index(3)
        self.play(FadeIn(h_prev), FadeIn(c_prev))

        timesteps = [1, 2, 3]
        # Layout per cell (relative to x_pos = gate-box column x):
        #   x_t / concat node -> x_pos - 0.95  (radius 0.4 / 0.22)
        #   gate boxes         -> x_pos + 0.0  (width 0.85)
        #   gate neurons       -> x_pos + 0.85 (radius 0.24)
        #   c_t / h_t column   -> x_pos + 1.85 (radius 0.4)
        # Cell footprint: [x_pos - 1.35, x_pos + 2.25] = 3.6 units wide.
        x_spacing = 3.7
        x_base = -6.4
        prev_h = h_prev
        prev_c = c_prev

        tip_kw = {"tip_length": 0.16}

        for i, t in enumerate(timesteps):
            x_pos = x_base + (i + 1) * x_spacing

            # Input neuron sits at the bottom-left of the cell, aligned with the
            # concat node above it so the x_t -> concat arrow is purely vertical.
            x_t = Neuron(label=f"x_{t}").shift(
                RIGHT * (x_pos - 0.95) + DOWN * (3.1 - y_offset)
            )

            # Concat node collects [h_{t-1}, x_t] and fans out to the four gates.
            # Placed above the c_t lane (between h row and i row) so the long
            # c_{t-1} -> c_t recurrent arrow passes BELOW the concat rather
            # than through it. Small radius keeps it visually lightweight.
            concat = Neuron(label="", radius=0.22).shift(
                RIGHT * (x_pos - 0.95) + DOWN * (0.95 - y_offset)
            )
            concat.set_z_index(3)

            # Gate boxes — stacked vertically; z_index=2 keeps labels above arrows.
            # Order top→bottom: o, f, i, g — keeps o_t aligned with h_t so the
            # o_t → h_t arrow stays horizontal instead of slicing through c_t.
            o_box = LabeledBox(label=r"\sigma_o", width=0.85, height=0.52).shift(
                RIGHT * x_pos + DOWN * (0.5 - y_offset)
            )
            f_box = LabeledBox(label=r"\sigma_f", width=0.85, height=0.52).shift(
                RIGHT * x_pos + DOWN * (1.2 - y_offset)
            )
            i_box = LabeledBox(label=r"\sigma_i", width=0.85, height=0.52).shift(
                RIGHT * x_pos + DOWN * (1.9 - y_offset)
            )
            g_box = LabeledBox(label=r"\tanh_g", width=0.85, height=0.52).shift(
                RIGHT * x_pos + DOWN * (2.6 - y_offset)
            )
            for box in (f_box, i_box, g_box, o_box):
                box.set_z_index(2)

            # Gate activation neurons — slightly to the right of each gate box.
            o_t = Neuron(label=f"o_{t}", radius=0.24).shift(
                RIGHT * (x_pos + 0.85) + DOWN * (0.5 - y_offset)
            )
            f_t = Neuron(label=f"f_{t}", radius=0.24).shift(
                RIGHT * (x_pos + 0.85) + DOWN * (1.2 - y_offset)
            )
            i_t = Neuron(label=f"i_{t}", radius=0.24).shift(
                RIGHT * (x_pos + 0.85) + DOWN * (1.9 - y_offset)
            )
            g_t = Neuron(label=f"g_{t}", radius=0.24).shift(
                RIGHT * (x_pos + 0.85) + DOWN * (2.6 - y_offset)
            )

            # c_t between f and i rows; h_t aligned with o row so o_t → h_t is
            # a short horizontal hop rather than a diagonal that clips c_t.
            c_t = Neuron(label=f"c_{t}", radius=0.4).shift(
                RIGHT * (x_pos + 1.85) + DOWN * (1.55 - y_offset)
            )
            c_t.set_z_index(3)
            h_t = Neuron(label=f"h_{t}").shift(
                RIGHT * (x_pos + 1.85) + DOWN * (0.5 - y_offset)
            )
            h_t.set_z_index(3)

            # --- Step 1: input x_t and concat node ---
            self.play(FadeIn(x_t), FadeIn(concat), run_time=0.3)
            a_x_to_cat = arrow_between(x_t, concat, **tip_kw)
            a_h_to_cat = _horizontal_arrow(prev_h, concat, **tip_kw)
            self.play(Create(a_x_to_cat), Create(a_h_to_cat), run_time=0.5)

            # --- Step 2: gate boxes appear and receive the shared [h, x] bus ---
            self.play(
                FadeIn(f_box), FadeIn(i_box), FadeIn(g_box), FadeIn(o_box),
                run_time=0.4,
            )
            # Force horizontal attachment so arrows to f/o (dy > dx) don't pick
            # a vertical path through i_box or g_box.
            a_cat_f = _horizontal_arrow(concat, f_box, **tip_kw)
            a_cat_i = _horizontal_arrow(concat, i_box, **tip_kw)
            a_cat_g = _horizontal_arrow(concat, g_box, **tip_kw)
            a_cat_o = _horizontal_arrow(concat, o_box, **tip_kw)
            self.play(
                Create(a_cat_f), Create(a_cat_i), Create(a_cat_g), Create(a_cat_o),
                run_time=0.5,
            )

            # --- Step 3: gate activation neurons ---
            af_out = arrow_between(f_box, f_t, **tip_kw)
            ai_out = arrow_between(i_box, i_t, **tip_kw)
            ag_out = arrow_between(g_box, g_t, **tip_kw)
            ao_out = arrow_between(o_box, o_t, **tip_kw)
            self.play(
                FadeIn(f_t), FadeIn(i_t), FadeIn(g_t), FadeIn(o_t),
                Create(af_out), Create(ai_out), Create(ag_out), Create(ao_out),
                run_time=0.5,
            )

            # --- Step 4: cell state update c_t ---
            # Four arrows fan into c_t (c_{t-1}, f_t, i_t, g_t). Larger buff +
            # short tips keep arrowheads clear of the c_t glyph.
            converge_kw = {"buff": 0.28, "tip_length": 0.13}
            ac_prev = arrow_between(prev_c, c_t, **converge_kw)
            ac_f = arrow_between(f_t, c_t, **converge_kw)
            ac_i = arrow_between(i_t, c_t, **converge_kw)
            ac_g = arrow_between(g_t, c_t, **converge_kw)
            self.play(
                FadeIn(c_t),
                Create(ac_prev), Create(ac_f), Create(ac_i), Create(ac_g),
                run_time=0.6,
            )

            # --- Step 5: hidden state h_t ---
            ah_o = arrow_between(o_t, h_t, **tip_kw)
            ah_c = arrow_between(c_t, h_t, **tip_kw)
            self.play(FadeIn(h_t), Create(ah_o), Create(ah_c), run_time=0.4)

            prev_h = h_t
            prev_c = c_t

        self.wait(1.4)
