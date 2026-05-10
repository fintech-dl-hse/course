"""RNN internals scene — low-level tensor operations inside one RNN cell.

Shows how x_t, h_{t-1} are combined through weight matrices W_ih, W_hh,
summed, passed through tanh to produce h_t, and then h_t is projected
through W_ho + softmax to produce y_t.

Flow is BOTTOM-TO-TOP: input token at bottom, output prediction at top.
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
    LEFT,
    MathTex,
    ORANGE,
    RED,
    RIGHT,
    Scene,
    Tex,
    UP,
    VGroup,
    WHITE,
    Write,
    DashedVMobject,
    RoundedRectangle,
)

from shared.neural import LabeledBox, TensorColumn, arrow_between


# Colors (consistent with high-level scene)
COLOR_HIDDEN = GREEN
COLOR_INPUT = ORANGE
COLOR_OUTPUT = RED
COLOR_WEIGHT = "#AAAAAA"


def _horizontal_arrow(a, b, **kwargs: Any) -> Arrow:
    defaults: dict[str, Any] = {"buff": 0.10, "stroke_width": 2.5, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


class RNNInternals(Scene):
    """Inside one RNN cell: x_t + h_{t-1} -> W_ih, W_hh -> tanh -> h_t -> W_ho -> softmax -> y_t.

    Bottom-to-top flow: input token at bottom, y_t prediction at top.
    Compact layout with formulas in bottom-right.
    """

    CELL_SIZE = 0.20
    DIM = 4

    # Vertical layout — BOTTOM TO TOP, compact
    Y_INPUT_TOKEN = -3.20     # input token (BOTTOM)
    Y_X = -2.40               # x_t embedding
    Y_WIH = -1.60             # W_ih
    Y_SUM = -0.80             # sum node
    Y_TANH = -0.05            # tanh
    Y_H = 0.70                # h_t
    Y_WHO = 1.45              # W_ho
    Y_SOFTMAX = 2.10          # softmax
    Y_Y = 2.80                # y_t (output, TOP)

    # Horizontal — narrower layout
    X_MAIN = 0.00             # main vertical flow centered
    X_H_PREV = -3.50          # h_{t-1} on the left
    X_H_NEXT = 3.00           # h_t copy going right
    X_WHH = -1.80             # W_hh between h_prev and sum

    BOX_W = 0.90
    BOX_H = 0.40

    def construct(self) -> None:
        # ================ Title ================
        title = (
            Tex(r"\textbf{Inside the RNN cell}")
            .scale(0.55)
            .move_to([0, 3.55, 0])
        )
        self.play(Write(title), run_time=0.4)

        # ================ Formulas (bottom-right corner) ================
        eq1 = MathTex(
            r"h_t = \tanh(W_{ih}\, x_t + W_{hh}\, h_{t-1} + b)"
        ).scale(0.42)
        eq2 = MathTex(
            r"y_t = \mathrm{softmax}(W_{ho}\, h_t)"
        ).scale(0.42)
        formulas = (
            VGroup(eq1, eq2)
            .arrange(DOWN, buff=0.08)
            .move_to([4.50, -2.80, 0])
        )
        self.play(Write(formulas), run_time=0.5)
        self.wait(0.2)

        # ================ Input token (BOTTOM) ================
        tok = (
            Tex(r"\textit{``cat''}")
            .scale(0.60)
            .move_to([self.X_MAIN, self.Y_INPUT_TOKEN, 0])
        )
        tok_label = (
            MathTex(r"\text{token}_t")
            .scale(0.50)
            .next_to(tok, RIGHT, buff=0.20)
        )
        self.play(FadeIn(tok), FadeIn(tok_label), run_time=0.4)

        # ================ x_t (embedding) ================
        x_t = TensorColumn(
            dim=self.DIM, cell_size=self.CELL_SIZE,
            color=COLOR_INPUT, fill_opacity=0.35,
        ).move_cells_to([self.X_MAIN, self.Y_X, 0])
        x_t_label = (
            MathTex(r"x_t")
            .scale(0.62)
            .set_color(COLOR_INPUT)
            .next_to(x_t, RIGHT, buff=0.18)
        )

        arr_tok_x = Arrow(
            start=tok.get_top(), end=x_t.get_bottom(),
            buff=0.10, stroke_width=2.5, color=COLOR_INPUT, tip_length=0.11,
        )
        emb_note = (
            MathTex(r"E[\text{id}]")
            .scale(0.50)
            .move_to([self.X_MAIN - 1.00, (self.Y_INPUT_TOKEN + self.Y_X) / 2, 0])
        )
        self.play(Create(arr_tok_x), FadeIn(x_t), FadeIn(x_t_label), FadeIn(emb_note), run_time=0.5)

        # ================ W_ih ================
        w_ih = LabeledBox(
            label="W_{ih}", width=self.BOX_W, height=self.BOX_H,
            label_scale=0.50, color=COLOR_INPUT,
        ).move_to([self.X_MAIN, self.Y_WIH, 0])

        arr_x_wih = Arrow(
            start=x_t.get_top(), end=w_ih.get_bottom(),
            buff=0.10, stroke_width=2.5, color=COLOR_INPUT, tip_length=0.11,
        )
        self.play(FadeIn(w_ih), Create(arr_x_wih), run_time=0.4)

        # ================ h_{t-1} on the left ================
        h_prev = TensorColumn(
            dim=self.DIM, cell_size=self.CELL_SIZE,
            color=COLOR_HIDDEN, fill_opacity=0.3,
        ).move_cells_to([self.X_H_PREV, self.Y_SUM, 0])
        h_prev_label = (
            MathTex(r"h_{t-1}")
            .scale(0.48)
            .set_color(COLOR_HIDDEN)
            .next_to(h_prev, DOWN, buff=0.10)
        )

        self.play(FadeIn(h_prev), FadeIn(h_prev_label), run_time=0.4)

        # ================ W_hh between h_prev and sum ================
        w_hh = LabeledBox(
            label="W_{hh}", width=0.80, height=self.BOX_H,
            label_scale=0.48, color=COLOR_HIDDEN,
        ).move_to([self.X_WHH, self.Y_SUM, 0])

        arr_h_whh = _horizontal_arrow(
            h_prev, w_hh, buff=0.10, tip_length=0.11, color=COLOR_HIDDEN,
        )
        self.play(FadeIn(w_hh), Create(arr_h_whh), run_time=0.4)

        # ================ Sum node (circle with +) ================
        sum_node = LabeledBox(
            label=r"+", width=0.45, height=0.45,
            corner_radius=0.22, label_scale=0.60,
            color=WHITE,
        ).move_to([self.X_MAIN, self.Y_SUM, 0])

        arr_wih_sum = Arrow(
            start=w_ih.get_top(), end=sum_node.get_bottom(),
            buff=0.08, stroke_width=2.5, tip_length=0.11,
        )
        arr_whh_sum = _horizontal_arrow(
            w_hh, sum_node, buff=0.08, tip_length=0.11, color=COLOR_HIDDEN,
        )

        self.play(
            FadeIn(sum_node),
            Create(arr_wih_sum),
            Create(arr_whh_sum),
            run_time=0.5,
        )

        # ================ tanh ================
        tanh_box = LabeledBox(
            label=r"\tanh", width=0.80, height=self.BOX_H,
            label_scale=0.58, color=WHITE,
        ).move_to([self.X_MAIN, self.Y_TANH, 0])

        arr_sum_tanh = Arrow(
            start=sum_node.get_top(), end=tanh_box.get_bottom(),
            buff=0.08, stroke_width=2.5, tip_length=0.11,
        )
        self.play(FadeIn(tanh_box), Create(arr_sum_tanh), run_time=0.4)

        # ================ h_t ================
        h_t = TensorColumn(
            dim=self.DIM, cell_size=self.CELL_SIZE,
            color=COLOR_HIDDEN, fill_opacity=0.35,
        ).move_cells_to([self.X_MAIN, self.Y_H, 0])
        h_t_label = (
            MathTex(r"h_t")
            .scale(0.48)
            .set_color(COLOR_HIDDEN)
            .next_to(h_t, LEFT, buff=0.15)
        )

        arr_tanh_h = Arrow(
            start=tanh_box.get_top(), end=h_t.get_bottom(),
            buff=0.10, stroke_width=2.5, color=COLOR_HIDDEN, tip_length=0.11,
        )
        self.play(FadeIn(h_t), FadeIn(h_t_label), Create(arr_tanh_h), run_time=0.5)

        # ================ h_t copy going right (to next step) ================
        h_t_next = TensorColumn(
            dim=self.DIM, cell_size=self.CELL_SIZE,
            color=COLOR_HIDDEN, fill_opacity=0.2,
        ).move_cells_to([self.X_H_NEXT, self.Y_H, 0])
        h_next_label = (
            MathTex(r"h_t \to \text{next step}")
            .scale(0.48)
            .set_color(COLOR_HIDDEN)
            .next_to(h_t_next, DOWN, buff=0.10)
        )
        arr_h_right = Arrow(
            start=h_t.get_right(), end=h_t_next.get_left(),
            buff=0.15, stroke_width=2.5, color=COLOR_HIDDEN, tip_length=0.12,
        )
        self.play(
            Create(arr_h_right), FadeIn(h_t_next), FadeIn(h_next_label),
            run_time=0.5,
        )

        # ================ W_ho ================
        w_ho = LabeledBox(
            label="W_{ho}", width=self.BOX_W, height=self.BOX_H,
            label_scale=0.50, color=COLOR_OUTPUT,
        ).move_to([self.X_MAIN, self.Y_WHO, 0])
        arr_h_who = Arrow(
            start=h_t.get_top(), end=w_ho.get_bottom(),
            buff=0.10, stroke_width=2.5, color=COLOR_OUTPUT, tip_length=0.11,
        )
        self.play(FadeIn(w_ho), Create(arr_h_who), run_time=0.4)

        # ================ softmax ================
        sm_box = LabeledBox(
            label=r"\text{softmax}", width=1.10, height=self.BOX_H,
            label_scale=0.55, color=COLOR_OUTPUT,
        ).move_to([self.X_MAIN, self.Y_SOFTMAX, 0])
        arr_who_sm = Arrow(
            start=w_ho.get_top(), end=sm_box.get_bottom(),
            buff=0.08, stroke_width=2.5, color=COLOR_OUTPUT, tip_length=0.11,
        )
        self.play(FadeIn(sm_box), Create(arr_who_sm), run_time=0.3)

        # ================ y_t (output, TOP) ================
        y_t = TensorColumn(
            dim=self.DIM, cell_size=self.CELL_SIZE,
            color=COLOR_OUTPUT, fill_opacity=0.35,
            highlight_index=2,
        ).move_cells_to([self.X_MAIN, self.Y_Y, 0])
        y_t_label = (
            MathTex(r"y_t")
            .scale(0.60)
            .set_color(COLOR_OUTPUT)
            .next_to(y_t, RIGHT, buff=0.18)
        )

        arr_sm_y = Arrow(
            start=sm_box.get_top(), end=y_t.get_bottom(),
            buff=0.08, stroke_width=2.5, color=COLOR_OUTPUT, tip_length=0.11,
        )
        pred_label = (
            Tex(r"\textit{``sat''}")
            .scale(0.60)
            .next_to(y_t, RIGHT, buff=0.50)
        )
        argmax_label = (
            MathTex(r"\arg\max")
            .scale(0.58)
            .next_to(pred_label, DOWN, buff=0.05)
        )

        self.play(
            FadeIn(y_t), FadeIn(y_t_label), Create(arr_sm_y),
            FadeIn(pred_label), FadeIn(argmax_label),
            run_time=0.5,
        )

        self.wait(1.5)
