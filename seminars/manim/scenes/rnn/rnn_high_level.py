"""RNN high-level computation graph scene.

Shows the RNN as a repeating block operating on a sequence of tokens.
Focus: overall computation graph, hidden state as a "memory" vector
passed between timesteps, input/output tokens.

Flow is BOTTOM-TO-TOP (inputs at bottom, outputs at top), LEFT-TO-RIGHT.

Each timestep is a single "RNN cell" box:
  token_out (top)
       ^
  [RNN Cell]  -- h_t -->
       ^
  token_in (bottom)
"""
from __future__ import annotations

from manim import (
    BLUE,
    Create,
    DOWN,
    DashedLine,
    FadeIn,
    FadeOut,
    GREEN,
    LEFT,
    MathTex,
    ORANGE,
    RED,
    RIGHT,
    RoundedRectangle,
    Scene,
    Tex,
    UP,
    VGroup,
    WHITE,
    Write,
    Arrow,
    YELLOW,
)

from shared.neural import LabeledBox, TensorColumn


# Colors
COLOR_CELL = BLUE
COLOR_HIDDEN = GREEN
COLOR_INPUT = ORANGE
COLOR_OUTPUT = RED


class RNNHighLevel(Scene):
    """High-level RNN computation graph: token -> RNN cell -> next token.

    Bottom-to-top flow: input tokens at bottom, output tokens at top.
    """

    # Layout — bottom to top
    Y_TITLE = 3.55
    Y_TOKEN_OUT = 2.50        # output token row (TOP)
    Y_CELL = 0.80             # RNN cell row (MIDDLE)
    Y_TOKEN_IN = -0.80        # input token row (BOTTOM)
    Y_HIDDEN = 0.80           # same as cell (horizontal arrows)

    X_POSITIONS = [-4.50, -1.50, 1.50, 4.50]  # h0, t=1, t=2, t=3

    CELL_W = 1.60
    CELL_H = 0.90

    def construct(self) -> None:
        # ================ Title ================
        title = MathTex(
            r"h_t = f(h_{t-1},\; x_t)"
            r"\qquad y_t = g(h_t)"
        ).scale(0.58).move_to([0, self.Y_TITLE, 0])
        self.play(Write(title))
        self.wait(0.3)

        # ================ h_0 — initial hidden state ================
        h0_x = self.X_POSITIONS[0]
        h0 = TensorColumn(
            dim=4, cell_size=0.22, color=COLOR_HIDDEN, fill_opacity=0.3,
        ).move_cells_to([h0_x, self.Y_HIDDEN, 0])
        h0_label = MathTex(r"h_0").scale(0.62).next_to(h0, DOWN, buff=0.15)
        h0_caption = (
            Tex(r"\textit{zeros}")
            .scale(0.82)
            .next_to(h0_label, DOWN, buff=0.08)
        )

        self.play(FadeIn(h0), FadeIn(h0_label), FadeIn(h0_caption))
        self.wait(0.3)

        tokens_in = ["the", "cat", "sat"]
        tokens_out = ["cat", "sat", "?"]
        prev_h_mob = h0

        rnn_cells = []
        h_tensors = []

        for step in range(3):
            t = step + 1
            cx = self.X_POSITIONS[step + 1]

            # --- Input token (BOTTOM) ---
            tok_in = (
                Tex(rf"\textit{{``{tokens_in[step]}''}}")
                .scale(0.65)
                .move_to([cx, self.Y_TOKEN_IN, 0])
            )
            tok_in_label = (
                MathTex(f"x_{t}")
                .scale(0.62)
                .set_color(COLOR_INPUT)
                .next_to(tok_in, LEFT, buff=0.15)
            )

            # --- RNN Cell box (MIDDLE) ---
            cell_box = RoundedRectangle(
                corner_radius=0.15,
                width=self.CELL_W,
                height=self.CELL_H,
                color=COLOR_CELL,
                stroke_width=3,
            ).set_fill(COLOR_CELL, opacity=0.15).move_to([cx, self.Y_CELL, 0])
            cell_label = (
                Tex(r"\textbf{RNN}")
                .scale(0.55)
                .move_to(cell_box.get_center())
            )
            cell_group = VGroup(cell_box, cell_label)

            # --- Output token (TOP) ---
            tok_out = (
                Tex(rf"\textit{{``{tokens_out[step]}''}}")
                .scale(0.60)
                .move_to([cx, self.Y_TOKEN_OUT, 0])
            )
            tok_out_label = (
                MathTex(f"y_{t}")
                .scale(0.62)
                .set_color(COLOR_OUTPUT)
                .next_to(tok_out, LEFT, buff=0.15)
            )

            # --- h_t tensor (right of cell) ---
            h_t = TensorColumn(
                dim=4, cell_size=0.22, color=COLOR_HIDDEN, fill_opacity=0.3,
            ).move_cells_to([cx + self.CELL_W / 2 + 0.60, self.Y_HIDDEN, 0])
            h_t_label = (
                MathTex(f"h_{t}")
                .scale(0.50)
                .set_color(COLOR_HIDDEN)
                .next_to(h_t, DOWN, buff=0.12)
            )

            # --- Arrows (bottom-to-top flow) ---
            # token_in (bottom) -> cell
            arr_in = Arrow(
                start=tok_in.get_top(), end=cell_box.get_bottom(),
                buff=0.10, stroke_width=2.5, color=COLOR_INPUT, tip_length=0.13,
            )

            # cell -> token_out (top)
            arr_out = Arrow(
                start=cell_box.get_top(), end=tok_out.get_bottom(),
                buff=0.10, stroke_width=2.5, color=COLOR_OUTPUT, tip_length=0.13,
            )

            # h_{t-1} -> cell (horizontal)
            arr_h_in = Arrow(
                start=prev_h_mob.get_right(), end=cell_box.get_left(),
                buff=0.10, stroke_width=2.5, color=COLOR_HIDDEN, tip_length=0.13,
            )

            # cell -> h_t (horizontal)
            arr_h_out = Arrow(
                start=cell_box.get_right(), end=h_t.get_left(),
                buff=0.10, stroke_width=2.5, color=COLOR_HIDDEN, tip_length=0.13,
            )

            # --- Animate ---
            # Step 1: input token
            self.play(FadeIn(tok_in), FadeIn(tok_in_label), run_time=0.4)

            # Step 2: h_{t-1} -> cell <- x_t
            self.play(
                FadeIn(cell_group),
                Create(arr_in),
                Create(arr_h_in),
                run_time=0.6,
            )

            # Step 3: cell -> h_t, cell -> y_t
            self.play(
                Create(arr_h_out),
                FadeIn(h_t), FadeIn(h_t_label),
                Create(arr_out),
                FadeIn(tok_out), FadeIn(tok_out_label),
                run_time=0.7,
            )

            self.wait(0.3)

            prev_h_mob = h_t
            rnn_cells.append(cell_group)
            h_tensors.append(h_t)

        # ================ Highlight recurrent chain ================
        self.wait(0.5)
        caption = (
            MathTex(
                r"h_t \text{ carries information from all previous tokens}"
            )
            .scale(0.48)
            .to_edge(DOWN, buff=0.15)
        )
        self.play(FadeIn(caption))
        self.wait(1.5)
