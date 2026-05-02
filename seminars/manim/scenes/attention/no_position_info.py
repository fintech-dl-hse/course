"""Attention has no positional information — permutation equivariance.

Shows that self-attention output for each token is the SAME regardless of
its position in the sequence:
  "dog bites man"  vs  "man bites dog"

The attention MATRICES are different (permuted rows/columns), but the
output representation z_dog, z_man, z_bites is identical in both cases.

Key formula: z_i = sum_j softmax(q_i . k_j / sqrt(d)) * v_j
— depends only on content, not position.
"""
from __future__ import annotations

from manim import (
    Arrow,
    BLUE,
    Create,
    DOWN,
    DashedLine,
    FadeIn,
    GREEN,
    GREY,
    LEFT,
    MathTex,
    ORANGE,
    RED,
    RoundedRectangle,
    Scene,
    Square,
    Tex,
    VGroup,
    WHITE,
    YELLOW,
    Write,
    Indicate,
    SurroundingRectangle,
)


COLOR_TOKEN = BLUE
COLOR_ATTN = YELLOW
COLOR_OUTPUT = GREEN
COLOR_WARN = RED


class NoPositionInfo(Scene):
    """Self-attention is permutation-equivariant — no position info."""

    # --- Layout ---
    Y_FORMULA = 3.50
    Y_OUTPUT = 2.00         # output (top)
    Y_ATTN = -0.10          # attention matrix (middle)
    Y_TOKENS = -2.40        # input tokens (bottom)
    MATRIX_CELL = 0.55

    # Two examples side by side
    X_LEFT = -3.50          # "dog bites man"
    X_RIGHT = 3.50          # "man bites dog"
    X_STEP = 1.40           # spacing between tokens

    T_STEP = 0.6            # animation step time

    # Content-based attention (asymmetric: dog != man).
    # attn(query_word, key_word) — depends ONLY on content.
    # dog:   strong attention to bites, moderate to man
    # bites: attends to both nouns (dog slightly more)
    # man:   strong attention to bites, moderate to dog
    ATTN_CONTENT = {
        ("dog", "dog"):   0.20, ("dog", "bites"):   0.55, ("dog", "man"):   0.25,
        ("bites", "dog"): 0.50, ("bites", "bites"): 0.10, ("bites", "man"): 0.40,
        ("man", "dog"):   0.30, ("man", "bites"):   0.55, ("man", "man"):   0.15,
    }

    def _weights_for(self, words: list[str]) -> list[list[float]]:
        """Build attention matrix for a given word order."""
        return [[self.ATTN_CONTENT[(words[i], words[j])]
                 for j in range(3)] for i in range(3)]

    def _make_token_boxes(self, words: list[str], x_center: float,
                          y: float) -> list[VGroup]:
        groups = []
        for i, w in enumerate(words):
            x = x_center + (i - 1) * self.X_STEP
            box = RoundedRectangle(
                corner_radius=0.08, width=1.10, height=0.50,
                color=COLOR_TOKEN, stroke_width=2,
            ).set_fill(COLOR_TOKEN, opacity=0.15).move_to([x, y, 0])
            lbl = Tex(rf"\textit{{{w}}}").scale(0.80).move_to(box.get_center())
            groups.append(VGroup(box, lbl))
        return groups

    def _make_attn_matrix(self, x_center: float, y_center: float,
                          weights: list[list[float]],
                          row_labels: list[str],
                          col_labels: list[str],
                          ) -> tuple[VGroup, list[MathTex], list[MathTex]]:
        """Build a 3x3 attention weight matrix with row/col labels."""
        all_cells = VGroup()
        w_max = max(max(r) for r in weights)
        for i in range(3):
            for j in range(3):
                cx = x_center + (j - 1) * self.MATRIX_CELL
                cy = y_center + (1 - i) * self.MATRIX_CELL
                w = weights[i][j]
                opacity = 0.08 + 0.82 * w / w_max
                cell = Square(
                    side_length=self.MATRIX_CELL,
                    color=COLOR_ATTN, stroke_width=1.5,
                ).set_fill(COLOR_ATTN, opacity=opacity)
                cell.move_to([cx, cy, 0])
                all_cells.add(cell)

        # Row labels (left) — query indices
        r_lbls = []
        for i, w in enumerate(row_labels):
            lbl = (MathTex(rf"q_{{\text{{{w[0]}}}}}")
                   .scale(0.58)
                   .move_to([x_center - 1.5 * self.MATRIX_CELL - 0.30,
                             y_center + (1 - i) * self.MATRIX_CELL, 0]))
            r_lbls.append(lbl)

        # Col labels (below) — key indices
        c_lbls = []
        for j, w in enumerate(col_labels):
            lbl = (MathTex(rf"k_{{\text{{{w[0]}}}}}")
                   .scale(0.58)
                   .move_to([x_center + (j - 1) * self.MATRIX_CELL,
                             y_center - 1.5 * self.MATRIX_CELL - 0.20, 0]))
            c_lbls.append(lbl)

        return all_cells, r_lbls, c_lbls

    def _make_output_boxes(self, words: list[str], x_center: float,
                           y: float) -> list[VGroup]:
        groups = []
        for i, w in enumerate(words):
            x = x_center + (i - 1) * self.X_STEP
            box = RoundedRectangle(
                corner_radius=0.08, width=1.10, height=0.50,
                color=COLOR_OUTPUT, stroke_width=2,
            ).set_fill(COLOR_OUTPUT, opacity=0.15).move_to([x, y, 0])
            txt = MathTex(rf"z_{{\text{{{w}}}}}").scale(0.63).move_to(box.get_center())
            groups.append(VGroup(box, txt))
        return groups

    def construct(self) -> None:
        # ============== Formula ==============
        formula = (
            MathTex(
                r"z_i = \sum_j \text{softmax}\!\left("
                r"\frac{q_i \cdot k_j}{\sqrt{d}}\right) v_j"
            )
            .scale(0.52)
            .move_to([0, self.Y_FORMULA, 0])
        )
        subtitle = (
            MathTex(r"\text{depends only on \textbf{content}, not position}")
            .scale(0.45)
            .next_to(formula, DOWN, buff=0.12)
            .set_color(YELLOW)
        )
        self.play(Write(formula), run_time=0.7)
        self.play(FadeIn(subtitle), run_time=0.4)
        self.wait(0.4)

        # ============== Left: "dog bites man" ==============
        words_l = ["dog", "bites", "man"]
        toks_l = self._make_token_boxes(words_l, self.X_LEFT, self.Y_TOKENS)
        weights_l = self._weights_for(words_l)

        title_l = (Tex(r"\textit{dog bites man}")
                   .scale(0.55)
                   .move_to([self.X_LEFT, self.Y_TOKENS - 0.55, 0])
                   .set_color(COLOR_TOKEN))

        self.play(*[FadeIn(t) for t in toks_l], FadeIn(title_l), run_time=0.4)

        attn_group_l, rlbls_l, clbls_l = self._make_attn_matrix(
            self.X_LEFT, self.Y_ATTN, weights_l, words_l, words_l)

        in_arrows_l = [
            Arrow(start=toks_l[i].get_top(),
                  end=[self.X_LEFT + (i - 1) * self.X_STEP,
                       self.Y_ATTN - 1.5 * self.MATRIX_CELL - 0.40, 0],
                  buff=0.08, stroke_width=1.5, tip_length=0.07, color=GREY)
            for i in range(3)]

        attn_lbl_l = (MathTex(r"A_1").scale(0.50)
                      .next_to(attn_group_l, LEFT, buff=0.70))

        self.play(
            *[Create(a) for a in in_arrows_l],
            FadeIn(attn_group_l), FadeIn(attn_lbl_l),
            *[FadeIn(l) for l in rlbls_l + clbls_l],
            run_time=self.T_STEP,
        )

        # Left output
        out_l = self._make_output_boxes(words_l, self.X_LEFT, self.Y_OUTPUT)
        out_arrows_l = [
            Arrow(start=[self.X_LEFT + (i - 1) * self.X_STEP,
                         self.Y_ATTN + 1.5 * self.MATRIX_CELL + 0.10, 0],
                  end=out_l[i].get_bottom(),
                  buff=0.08, stroke_width=1.5, tip_length=0.07, color=COLOR_OUTPUT)
            for i in range(3)]

        self.play(
            *[FadeIn(o) for o in out_l],
            *[Create(a) for a in out_arrows_l],
            run_time=self.T_STEP,
        )
        self.wait(0.3)

        # ============== Separator ==============
        sep = DashedLine(
            start=[0, self.Y_FORMULA - 0.50, 0],
            end=[0, self.Y_TOKENS - 0.40, 0],
            stroke_width=1.5, color=GREY, dash_length=0.10,
        ).set_opacity(0.4)
        self.play(FadeIn(sep), run_time=0.2)

        # ============== Right: "man bites dog" ==============
        words_r = ["man", "bites", "dog"]
        toks_r = self._make_token_boxes(words_r, self.X_RIGHT, self.Y_TOKENS)
        weights_r = self._weights_for(words_r)

        title_r = (Tex(r"\textit{man bites dog}")
                   .scale(0.55)
                   .move_to([self.X_RIGHT, self.Y_TOKENS - 0.55, 0])
                   .set_color(COLOR_TOKEN))

        self.play(*[FadeIn(t) for t in toks_r], FadeIn(title_r), run_time=0.4)

        attn_group_r, rlbls_r, clbls_r = self._make_attn_matrix(
            self.X_RIGHT, self.Y_ATTN, weights_r, words_r, words_r)

        in_arrows_r = [
            Arrow(start=toks_r[i].get_top(),
                  end=[self.X_RIGHT + (i - 1) * self.X_STEP,
                       self.Y_ATTN - 1.5 * self.MATRIX_CELL - 0.40, 0],
                  buff=0.08, stroke_width=1.5, tip_length=0.07, color=GREY)
            for i in range(3)]

        attn_lbl_r = (MathTex(r"A_2").scale(0.50)
                      .next_to(attn_group_r, LEFT, buff=0.70))

        self.play(
            *[Create(a) for a in in_arrows_r],
            FadeIn(attn_group_r), FadeIn(attn_lbl_r),
            *[FadeIn(l) for l in rlbls_r + clbls_r],
            run_time=self.T_STEP,
        )

        # Right output — SAME words, just reordered
        out_r = self._make_output_boxes(words_r, self.X_RIGHT, self.Y_OUTPUT)
        out_arrows_r = [
            Arrow(start=[self.X_RIGHT + (i - 1) * self.X_STEP,
                         self.Y_ATTN + 1.5 * self.MATRIX_CELL + 0.10, 0],
                  end=out_r[i].get_bottom(),
                  buff=0.08, stroke_width=1.5, tip_length=0.07, color=COLOR_OUTPUT)
            for i in range(3)]

        self.play(
            *[FadeIn(o) for o in out_r],
            *[Create(a) for a in out_arrows_r],
            run_time=self.T_STEP,
        )
        self.wait(0.5)

        # ============== Highlight: matrices ARE different ==============
        neq_matrix = (MathTex(r"\neq").scale(0.90)
                      .move_to([0, self.Y_ATTN, 0])
                      .set_color(WHITE))
        self.play(FadeIn(neq_matrix), run_time=0.3)

        # Flash matrices to show they're different
        self.play(
            Indicate(attn_group_l, color=ORANGE, scale_factor=1.03),
            Indicate(attn_group_r, color=ORANGE, scale_factor=1.03),
            run_time=0.8,
        )
        self.wait(0.3)

        # ============== But outputs ARE the same! ==============
        # z_dog(left) = z_dog(right), z_man(left) = z_man(right)
        # Highlight matching outputs
        # Left:  z_dog(pos0) z_bites(pos1) z_man(pos2)
        # Right: z_man(pos0) z_bites(pos1) z_dog(pos2)

        # Highlight z_dog on both sides
        highlight_l0 = SurroundingRectangle(
            out_l[0], color=COLOR_WARN, stroke_width=3, buff=0.06)
        highlight_r2 = SurroundingRectangle(
            out_r[2], color=COLOR_WARN, stroke_width=3, buff=0.06)

        eq_out = (MathTex(r"z_{\text{dog}} = z_{\text{dog}}").scale(0.65)
                  .move_to([0, self.Y_OUTPUT + 0.50, 0])
                  .set_color(COLOR_WARN))

        self.play(
            Create(highlight_l0), Create(highlight_r2),
            FadeIn(eq_out),
            run_time=self.T_STEP,
        )
        self.wait(0.5)

        # Same for z_man
        highlight_l2 = SurroundingRectangle(
            out_l[2], color=COLOR_WARN, stroke_width=3, buff=0.06)
        highlight_r0 = SurroundingRectangle(
            out_r[0], color=COLOR_WARN, stroke_width=3, buff=0.06)

        eq_out2 = (MathTex(r"z_{\text{man}} = z_{\text{man}}").scale(0.65)
                   .move_to([0, self.Y_OUTPUT - 0.05, 0])
                   .set_color(COLOR_WARN))

        self.play(
            Create(highlight_l2), Create(highlight_r0),
            FadeIn(eq_out2),
            run_time=self.T_STEP,
        )
        self.wait(0.5)

        # ============== Conclusion ==============
        conclusion = (
            MathTex(
                r"\text{different order, same } z_i",
                r"\implies",
                r"\text{need positional encoding!}",
            )
            .scale(0.48)
            .move_to([0, -3.40, 0])
        )
        conclusion[0].set_color(COLOR_WARN)
        conclusion[2].set_color(YELLOW)

        self.play(FadeIn(conclusion), run_time=self.T_STEP)
        self.wait(2.0)
