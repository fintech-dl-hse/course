"""RNN bottleneck scene — motivation for attention (NIAH example).

Shows ONE RNN cell processing tokens sequentially. The hidden state h_t
accumulates all context but has fixed dimensionality.

Left side: RNN cell with token input and h_t output.
Right side: growing context bar — shows all processed tokens accumulating.
Visual contrast: context grows unboundedly, but h_t stays fixed size.

Uses NIAH example: "The password is 7392" at start, ~1000 filler, then
"What was the password?" at the end.
"""
from __future__ import annotations

from typing import Any

from manim import (
    Arrow,
    BLUE,
    Create,
    DOWN,
    DashedLine,
    FadeIn,
    FadeOut,
    GREEN,
    GREY,
    LEFT,
    Line,
    MathTex,
    ORANGE,
    RED,
    RIGHT,
    Rectangle,
    RoundedRectangle,
    Scene,
    Tex,
    UP,
    VGroup,
    WHITE,
    YELLOW,
    Write,
    ReplacementTransform,
    Brace,
)

from shared.neural import TensorColumn


COLOR_CELL = BLUE
COLOR_HIDDEN = GREEN
COLOR_INPUT = ORANGE


class Seq2SeqBottleneck(Scene):
    """RNN bottleneck via NIAH: one cell, tokens cycle through, info gets lost."""

    CELL_SIZE = 0.22
    DIM = 5

    # Layout (bottom-to-top) — RNN on the left
    Y_TOKEN = -2.60
    Y_CELL = -0.80
    Y_H = 1.40
    Y_TITLE = 3.40
    Y_STEP_LABEL = -3.40

    X_MAIN = -2.50           # RNN cell shifted left
    CELL_W = 1.60
    CELL_H = 0.90

    # Context panel on the right
    X_CTX = 3.50             # center of context panel
    CTX_W = 2.80             # width of context area
    Y_CTX_TOP = 2.80
    Y_CTX_BOT = -3.00
    CTX_ROW_H = 0.30         # height per token row in context

    NEEDLE_TOKENS = ["The", "password", "is", "7392"]
    FILLER_TEXT = r"\ldots \text{1000 tokens} \ldots"
    QUESTION_TOKENS = ["What", "was", "the", "password", "?"]

    def construct(self) -> None:
        # ================ Title ================
        title = (
            Tex(r"\textbf{RNN Bottleneck}")
            .scale(0.65)
            .move_to([0, self.Y_TITLE, 0])
        )
        self.play(Write(title), run_time=0.4)
        self.wait(0.2)

        # ================ Context panel header (right side) ================
        ctx_header = (
            Tex(r"\textbf{Context seen so far}")
            .scale(0.45)
            .move_to([self.X_CTX, self.Y_CTX_TOP + 0.30, 0])
        )
        # Vertical line separating RNN from context
        sep_line = DashedLine(
            start=[1.0, self.Y_CTX_TOP + 0.50, 0],
            end=[1.0, self.Y_CTX_BOT - 0.50, 0],
            stroke_width=1, color=GREY, dash_length=0.12,
        ).set_opacity(0.4)
        self.play(FadeIn(ctx_header), FadeIn(sep_line), run_time=0.3)

        # ================ RNN Cell (permanent) ================
        cell_box = RoundedRectangle(
            corner_radius=0.12,
            width=self.CELL_W,
            height=self.CELL_H,
            color=COLOR_CELL,
            stroke_width=3,
        ).set_fill(COLOR_CELL, opacity=0.12).move_to([self.X_MAIN, self.Y_CELL, 0])
        cell_label = (
            Tex(r"\textbf{RNN Cell}")
            .scale(0.45)
            .move_to(cell_box.get_center())
        )
        cell_group = VGroup(cell_box, cell_label)
        self.play(FadeIn(cell_group), run_time=0.4)

        # ================ Permanent arrows ================
        arr_in = Arrow(
            start=[self.X_MAIN, self.Y_TOKEN + 0.25, 0],
            end=cell_box.get_bottom(),
            buff=0.08, stroke_width=2.5, color=COLOR_INPUT, tip_length=0.11,
        )
        input_label = (
            MathTex(r"x_t")
            .scale(0.50)
            .set_color(COLOR_INPUT)
            .next_to(arr_in, LEFT, buff=0.08)
        )
        arr_out = Arrow(
            start=cell_box.get_top(),
            end=[self.X_MAIN, self.Y_H - 0.50, 0],
            buff=0.08, stroke_width=2.5, color=COLOR_HIDDEN, tip_length=0.11,
        )
        output_label = (
            MathTex(r"h_t")
            .scale(0.45)
            .set_color(COLOR_HIDDEN)
            .next_to(arr_out, LEFT, buff=0.08)
        )
        recur_label = (
            MathTex(r"h_{t-1}")
            .scale(0.42)
            .set_color(COLOR_HIDDEN)
            .move_to([self.X_MAIN + self.CELL_W / 2 + 0.50, self.Y_CELL, 0])
        )

        self.play(
            Create(arr_in), FadeIn(input_label),
            Create(arr_out), FadeIn(output_label),
            FadeIn(recur_label),
            run_time=0.4,
        )

        # ================ h_t display ================
        h_vec = TensorColumn(
            dim=self.DIM, cell_size=self.CELL_SIZE,
            color=COLOR_HIDDEN, fill_opacity=0.30,
        ).move_cells_to([self.X_MAIN, self.Y_H, 0])
        dim_label = (
            MathTex(r"\dim = 256")
            .scale(0.40)
            .set_color(COLOR_HIDDEN)
            .next_to(h_vec, UP, buff=0.10)
        )
        self.play(FadeIn(h_vec), FadeIn(dim_label), run_time=0.3)

        # Step counter
        step_counter = (
            MathTex(r"t = 0")
            .scale(0.42)
            .move_to([self.X_MAIN, self.Y_STEP_LABEL, 0])
        )
        self.play(FadeIn(step_counter), run_time=0.2)

        # ================ Context tracking ================
        ctx_items: list[VGroup] = []  # accumulated context display items
        ctx_y_cursor = self.Y_CTX_TOP  # next Y position for context item

        def _add_ctx_item(text: str, color=WHITE, is_block: bool = False,
                          block_h: float = 0.0) -> VGroup:
            nonlocal ctx_y_cursor
            if is_block:
                h = block_h
                rect = Rectangle(
                    width=self.CTX_W - 0.40, height=h,
                    color=GREY, stroke_width=1,
                ).set_fill(GREY, opacity=0.10)
                label = MathTex(text).scale(0.38).set_color(color)
                item = VGroup(rect, label)
                ctx_y_cursor -= h / 2
                item.move_to([self.X_CTX, ctx_y_cursor, 0])
                ctx_y_cursor -= h / 2 + 0.05
            else:
                label = (
                    Tex(rf"\textit{{{text}}}")
                    .scale(0.63)
                    .set_color(color)
                )
                ctx_y_cursor -= self.CTX_ROW_H / 2
                label.move_to([self.X_CTX, ctx_y_cursor, 0])
                ctx_y_cursor -= self.CTX_ROW_H / 2 + 0.02
                item = VGroup(label)
            ctx_items.append(item)
            return item

        # ================ Process needle tokens ================
        current_tok_mob = None
        needle_highlight = None
        needle_in_h = None

        for i, tok in enumerate(self.NEEDLE_TOKENS):
            t = i + 1
            is_needle = (tok == "7392")
            tok_color = RED if is_needle else WHITE

            tok_mob = (
                Tex(rf"\textit{{{tok}}}")
                .scale(0.65)
                .set_color(tok_color)
                .move_to([self.X_MAIN, self.Y_TOKEN, 0])
            )
            new_counter = (
                MathTex(rf"t = {t}")
                .scale(0.42)
                .move_to([self.X_MAIN, self.Y_STEP_LABEL, 0])
            )

            # Add to context panel
            ctx_item = _add_ctx_item(tok, color=tok_color)

            anims = [
                FadeIn(tok_mob),
                ReplacementTransform(step_counter, new_counter),
                FadeIn(ctx_item),
            ]
            if current_tok_mob is not None:
                anims.append(FadeOut(current_tok_mob))
            self.play(*anims, run_time=0.4)

            step_counter = new_counter
            current_tok_mob = tok_mob

            # Flash h_t
            flash_anims = []
            for cell in h_vec.cells:
                flash_anims.append(cell.animate.set_fill(
                    RED if is_needle else COLOR_HIDDEN,
                    opacity=0.6 if is_needle else 0.40,
                ))
            self.play(*flash_anims, run_time=0.25)

            if is_needle:
                needle_highlight = (
                    Tex(r"$\leftarrow$ needle!")
                    .scale(0.45)
                    .set_color(RED)
                    .next_to(tok_mob, RIGHT, buff=0.20)
                )
                needle_in_h = (
                    Tex(r"7392 in $h_4$")
                    .scale(0.40)
                    .set_color(RED)
                    .next_to(h_vec, LEFT, buff=0.20)
                )
                self.play(FadeIn(needle_highlight), FadeIn(needle_in_h), run_time=0.3)
                self.wait(0.3)

        self.wait(0.2)

        # ================ Filler: ~1000 tokens ================
        fade_list = [FadeOut(current_tok_mob)]
        if needle_highlight is not None:
            fade_list.append(FadeOut(needle_highlight))
        if needle_in_h is not None:
            fade_list.append(FadeOut(needle_in_h))
        self.play(*fade_list, run_time=0.3)

        filler_mob = (
            MathTex(self.FILLER_TEXT)
            .scale(0.50)
            .set_color(GREY)
            .move_to([self.X_MAIN, self.Y_TOKEN, 0])
        )
        new_counter = (
            MathTex(r"t = 5 \ldots 1004")
            .scale(0.42)
            .move_to([self.X_MAIN, self.Y_STEP_LABEL, 0])
        )
        # Context panel: big filler block
        filler_ctx = _add_ctx_item(
            r"\vdots \quad \text{1000 tokens} \quad \vdots",
            color=GREY, is_block=True, block_h=1.80,
        )

        self.play(
            FadeIn(filler_mob),
            ReplacementTransform(step_counter, new_counter),
            FadeIn(filler_ctx),
            run_time=0.4,
        )
        step_counter = new_counter

        # h_t churning
        for _ in range(3):
            churn = [c.animate.set_fill(GREY, opacity=0.25) for c in h_vec.cells]
            self.play(*churn, run_time=0.20)
            churn2 = [c.animate.set_fill(COLOR_HIDDEN, opacity=0.30) for c in h_vec.cells]
            self.play(*churn2, run_time=0.20)

        self.wait(0.2)

        # ================ Question tokens ================
        self.play(FadeOut(filler_mob), run_time=0.2)

        for i, tok in enumerate(self.QUESTION_TOKENS):
            t = 1005 + i
            tok_mob = (
                Tex(rf"\textit{{{tok}}}")
                .scale(0.65)
                .set_color(YELLOW)
                .move_to([self.X_MAIN, self.Y_TOKEN, 0])
            )
            new_counter = (
                MathTex(rf"t = {t}")
                .scale(0.42)
                .move_to([self.X_MAIN, self.Y_STEP_LABEL, 0])
            )
            ctx_item = _add_ctx_item(tok, color=YELLOW)

            anims = [
                FadeIn(tok_mob),
                ReplacementTransform(step_counter, new_counter),
                FadeIn(ctx_item),
            ]
            if i > 0:
                anims.append(FadeOut(current_tok_mob))
            self.play(*anims, run_time=0.30)
            step_counter = new_counter
            current_tok_mob = tok_mob

        self.wait(0.2)

        # ================ Punchline: contrast context size vs h_t ================
        self.play(FadeOut(current_tok_mob), run_time=0.2)

        question_full = (
            Tex(r'\textit{``What was the password?"}')
            .scale(0.50)
            .set_color(YELLOW)
            .move_to([self.X_MAIN, self.Y_TOKEN, 0])
        )
        self.play(FadeIn(question_full), run_time=0.3)

        # Brace on context panel — "1009 tokens"
        ctx_brace = Brace(
            VGroup(*ctx_items), direction=RIGHT, buff=0.08,
        ).set_color(WHITE).scale(0.9)
        ctx_brace_label = (
            MathTex(r"\text{1009 tokens}")
            .scale(0.40)
            .next_to(ctx_brace, RIGHT, buff=0.08)
        )
        self.play(FadeIn(ctx_brace), FadeIn(ctx_brace_label), run_time=0.4)

        # Highlight h_t and show "ALL of this → one vector"
        punchline_anims = []
        for cell in h_vec.cells:
            punchline_anims.append(cell.animate.set_stroke(YELLOW, width=3))
            punchline_anims.append(cell.animate.set_fill(YELLOW, opacity=0.45))
        self.play(*punchline_anims, run_time=0.5)

        # Arrow from context to h_t with "compressed into"
        compress_arrow = Arrow(
            start=[self.X_CTX - self.CTX_W / 2 - 0.10, self.Y_H, 0],
            end=[self.X_MAIN + 0.60, self.Y_H, 0],
            buff=0.08, stroke_width=3, color=RED, tip_length=0.14,
        )
        compress_label = (
            MathTex(r"\text{all } \to h_t")
            .scale(0.42)
            .set_color(RED)
            .next_to(compress_arrow, UP, buff=0.08)
        )
        self.play(Create(compress_arrow), FadeIn(compress_label), run_time=0.4)

        # Question
        doubt = (
            MathTex(r"\text{is 7392 still in } h_{1009} \text{?}")
            .scale(0.45)
            .set_color(RED)
            .next_to(h_vec, LEFT, buff=0.25)
        )
        self.play(FadeIn(doubt), run_time=0.4)
        self.wait(0.5)

        # ================ Conclusion ================
        conclusion = (
            MathTex(
                r"\Rightarrow \text{ need access to all } h_1 \ldots h_T"
                r"\text{ — attention!}"
            )
            .scale(0.50)
            .move_to([0, -3.65, 0])
        )
        self.play(FadeIn(conclusion), run_time=0.4)
        self.wait(1.5)
