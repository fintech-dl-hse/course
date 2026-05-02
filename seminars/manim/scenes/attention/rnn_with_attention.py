"""RNN + Attention solves the NIAH bottleneck.

Step-by-step attention computation:
1. Build RNN chain: tokens → h_1..h_11
2. Extract query q = h_11
3. Compute scores: s_i = q · h_i  (dashed lines to ALL h_i)
4. Softmax → attention weights α_i  (bar chart, only needle labeled)
5. Weighted sum: draw lines from ALL h_i to context c, then highlight needle
6. Answer: c ≈ 0.72·h_4 → 7392
"""
from __future__ import annotations

from manim import (
    Arrow,
    Create,
    DashedLine,
    FadeIn,
    FadeOut,
    GREEN,
    GREY,
    LEFT,
    Line,
    MathTex,
    ORANGE,
    PURPLE,
    RED,
    RIGHT,
    Rectangle,
    RoundedRectangle,
    Scene,
    SurroundingRectangle,
    Tex,
    Transform,
    UP,
    VGroup,
    WHITE,
    YELLOW,
    Write,
    Indicate,
)


COLOR_HIDDEN = GREEN
COLOR_NEEDLE = RED
COLOR_QUERY = PURPLE
COLOR_ATTN = YELLOW
COLOR_OUTPUT = ORANGE


class RNNWithAttention(Scene):
    """Attention over RNN hidden states — step by step."""

    TOKENS = ["The", "pw", "is", "7392",
              "...", "...",
              "What", "was", "the", "pw", "?"]
    N = 11
    NEEDLE_IDX = 3  # "7392"

    # Layout — tighter vertical spacing to reduce dead space
    Y_TITLE = 3.50
    Y_RESULT = 2.80        # result line c ≈ ... → 7392
    Y_CONTEXT = 2.25       # context vector c
    Y_WEIGHTS = 0.70       # attention weight bars (lowered)
    Y_STEP_LBL = 0.00      # step labels (only during animation)
    Y_QUERY = -0.50        # query q position
    Y_H = -1.20            # hidden states (moved up more)
    Y_TOKENS = -2.50       # tokens (moved up)
    Y_CONCLUSION = -3.40

    X_START = -5.50
    X_STEP = 1.00

    CELL_W = 0.52
    CELL_H = 0.42

    # Attention weights (after softmax). Needle dominates.
    ATTN_W = [0.02, 0.03, 0.04, 0.72,
              0.02, 0.02,
              0.04, 0.03, 0.02, 0.03, 0.03]

    # Raw scores (before softmax, for display)
    SCORES = [0.1, 0.3, 0.5, 4.8,
              0.2, 0.1,
              0.6, 0.4, 0.2, 0.3, 0.3]

    def _xs(self) -> list[float]:
        return [self.X_START + i * self.X_STEP for i in range(self.N)]

    def _h_cell(self, x: float, y: float, idx: int, color=COLOR_HIDDEN) -> VGroup:
        box = RoundedRectangle(
            corner_radius=0.04, width=self.CELL_W, height=self.CELL_H,
            color=color, stroke_width=2,
        ).set_fill(color, opacity=0.15).move_to([x, y, 0])
        lbl = MathTex(rf"h_{{{idx}}}").scale(0.43).move_to(box.get_center())
        return VGroup(box, lbl)

    def construct(self) -> None:
        xs = self._xs()

        # ============== Title ==============
        title = (Tex(r"\textbf{RNN + Attention}")
                 .scale(0.58).move_to([0, self.Y_TITLE, 0]))
        self.play(Write(title), run_time=0.5)

        # ============== Tokens (bottom) ==============
        tok_mobs = []
        for i, tok in enumerate(self.TOKENS):
            is_needle = (i == self.NEEDLE_IDX)
            is_filler = tok == "..."
            is_question = i >= 6
            color = (COLOR_NEEDLE if is_needle
                     else GREY if is_filler
                     else YELLOW if is_question
                     else WHITE)
            if is_filler:
                # Grey dash as "skipped tokens" indicator (not text → no lint)
                mob = Line(
                    LEFT * 0.20, RIGHT * 0.20,
                    color=GREY, stroke_width=2.5,
                ).set_opacity(0.5)
            else:
                mob = Tex(rf"\textit{{{tok}}}").scale(0.63).set_color(color)
            mob.move_to([xs[i], self.Y_TOKENS, 0])
            tok_mobs.append(mob)
        self.play(*[FadeIn(t) for t in tok_mobs], run_time=0.4)

        # ============== Hidden states h_1..h_N ==============
        h_cells = []
        for i in range(self.N):
            color = COLOR_NEEDLE if i == self.NEEDLE_IDX else COLOR_HIDDEN
            h = self._h_cell(xs[i], self.Y_H, i + 1, color)
            h_cells.append(h)

        h_arrows = [
            Arrow(start=tok_mobs[i].get_top(), end=h_cells[i][0].get_bottom(),
                  buff=0.10, stroke_width=1.2, tip_length=0.05, color=GREY)
            for i in range(self.N)]

        h_chain = [
            Arrow(start=h_cells[i][0].get_right(), end=h_cells[i + 1][0].get_left(),
                  buff=0.03, stroke_width=1.5, tip_length=0.05, color=COLOR_HIDDEN)
            for i in range(self.N - 1)]

        self.play(
            *[FadeIn(h) for h in h_cells],
            *[Create(a) for a in h_arrows],
            *[Create(a) for a in h_chain],
            run_time=0.7,
        )
        self.wait(0.3)

        # ======================================================
        # STEP 1: Extract query q = h_T
        # ======================================================
        step_lbl = (
            Tex(r"\textbf{Step 1:} query $q = h_{11}$")
            .scale(0.48).set_color(COLOR_QUERY)
            .move_to([0, self.Y_STEP_LBL, 0]))
        self.play(FadeIn(step_lbl), run_time=0.4)

        # q above h_11
        q_label = (
            MathTex(r"q").scale(0.58).set_color(COLOR_QUERY)
            .move_to([xs[-1], self.Y_QUERY, 0]))
        q_arrow = Arrow(
            start=h_cells[-1][0].get_top(), end=q_label.get_bottom(),
            buff=0.08, stroke_width=2, tip_length=0.08, color=COLOR_QUERY,
        )
        self.play(Create(q_arrow), FadeIn(q_label), run_time=0.5)
        self.wait(0.4)

        # ======================================================
        # STEP 2: Compute scores s_i = q · h_i
        # ======================================================
        step2_lbl = (
            Tex(r"\textbf{Step 2:} scores $s_i = q \cdot h_i$")
            .scale(0.48).set_color(WHITE)
            .move_to([0, self.Y_STEP_LBL, 0]))
        self.play(Transform(step_lbl, step2_lbl), run_time=0.3)

        # Dashed lines from q to every h_i + score labels above each h
        score_lines = []
        score_labels = []
        for i in range(self.N):
            is_needle = (i == self.NEEDLE_IDX)
            skip_line = (i == self.N - 1)  # skip h_11, q IS h_11

            if not skip_line:
                line = DashedLine(
                    start=q_label.get_left() + LEFT * 0.05,
                    end=h_cells[i][0].get_top() + UP * 0.06,
                    stroke_width=2.0 if is_needle else 1.2,
                    color=COLOR_NEEDLE if is_needle else GREY,
                    dash_length=0.06,
                ).set_opacity(0.8 if is_needle else 0.4)
            else:
                line = None
            score_lines.append(line)

            # Score value label above h_i
            score_val = f"{self.SCORES[i]:.1f}"
            s_color = COLOR_NEEDLE if is_needle else WHITE
            s_lbl = (MathTex(score_val).scale(0.42).set_color(s_color)
                     .next_to(h_cells[i][0], UP, buff=0.42))
            if skip_line:
                s_lbl.shift(LEFT * 0.45)
            score_labels.append(s_lbl)

        # Show lines in 3 groups: left (0-2), needle (3), rest (4-10)
        self.play(
            *[Create(score_lines[i]) for i in range(3)],
            *[FadeIn(score_labels[i]) for i in range(3)],
            run_time=0.5,
        )
        # Needle — emphasized
        self.play(
            Create(score_lines[3]),
            FadeIn(score_labels[3]),
            run_time=0.5,
        )
        # Rest (skip None)
        self.play(
            *[Create(score_lines[i]) for i in range(4, self.N)
              if score_lines[i] is not None],
            *[FadeIn(score_labels[i]) for i in range(4, self.N)],
            run_time=0.5,
        )
        self.wait(0.3)

        # Highlight needle score
        self.play(
            Indicate(score_labels[self.NEEDLE_IDX],
                     color=COLOR_NEEDLE, scale_factor=1.3),
            run_time=0.5,
        )
        self.wait(0.3)

        # ======================================================
        # STEP 3: Softmax → attention weights α_i
        # ======================================================
        step3_lbl = (
            Tex(r"\textbf{Step 3:} $\alpha_i = \mathrm{softmax}(s_i)$")
            .scale(0.48).set_color(COLOR_ATTN)
            .move_to([0, self.Y_STEP_LBL, 0]))
        self.play(Transform(step_lbl, step3_lbl), run_time=0.3)

        # Fade out score lines and q stuff
        self.play(
            *[FadeOut(sl) for sl in score_lines if sl is not None],
            FadeOut(q_arrow), FadeOut(q_label),
            run_time=0.3,
        )

        # Bar chart — min bar height visible as a bar, not a dash
        max_bar_h = 0.85
        min_bar_h = 0.22  # bumped up so non-needle bars look like bars
        w_max = max(self.ATTN_W)
        bars = []
        bar_labels = []
        for i in range(self.N):
            w = self.ATTN_W[i]
            bar_h = max(min_bar_h, max_bar_h * w / w_max)
            is_needle = (i == self.NEEDLE_IDX)
            color = COLOR_NEEDLE if is_needle else COLOR_ATTN

            bar = RoundedRectangle(
                corner_radius=0.03,
                width=self.CELL_W * 0.75, height=bar_h,
                color=color, stroke_width=1.5,
            ).set_fill(color, opacity=0.6 if is_needle else 0.30)
            bar.move_to([xs[i], self.Y_WEIGHTS + bar_h / 2, 0])
            bars.append(bar)

            # Only label the needle bar
            if is_needle:
                w_lbl = (MathTex(r"0.72").scale(0.48).set_color(COLOR_NEEDLE)
                         .next_to(bar, UP, buff=0.06))
                bar_labels.append(w_lbl)

        # Fade scores → show bars
        self.play(
            *[FadeOut(sl) for sl in score_labels],
            *[FadeIn(b) for b in bars],
            *[FadeIn(bl) for bl in bar_labels],
            run_time=0.7,
        )

        # α_i label on the left
        alpha_label = (
            MathTex(r"\alpha_i").scale(0.55).set_color(COLOR_ATTN)
            .move_to([xs[0] - 0.80, self.Y_WEIGHTS + max_bar_h / 2, 0]))
        self.play(FadeIn(alpha_label), run_time=0.3)
        self.wait(0.3)

        # ======================================================
        # STEP 4: Weighted sum c = Σ α_i h_i
        # ======================================================
        step4_lbl = (
            Tex(r"\textbf{Step 4:} $c = \sum_i \alpha_i \, h_i$")
            .scale(0.48).set_color(COLOR_OUTPUT)
            .move_to([0, self.Y_STEP_LBL, 0]))
        self.play(Transform(step_lbl, step4_lbl), run_time=0.3)

        # Context label with surrounding box
        ctx_x = 0.0
        context_text = (
            MathTex(r"c").scale(0.70).set_color(COLOR_OUTPUT)
            .move_to([ctx_x, self.Y_CONTEXT, 0]))
        context_box = SurroundingRectangle(
            context_text, buff=0.08, color=COLOR_OUTPUT, stroke_width=1.5,
        ).set_fill(COLOR_OUTPUT, opacity=0.10)
        context_group = VGroup(context_box, context_text)

        # Lines from each bar top to context c — ALL of them
        ctx_lines = []
        for i in range(self.N):
            w = self.ATTN_W[i]
            bar_h = max(min_bar_h, max_bar_h * w / w_max)
            bar_top_y = self.Y_WEIGHTS + bar_h

            line = Line(
                start=[xs[i], bar_top_y + 0.03, 0],
                end=[ctx_x, self.Y_CONTEXT - 0.18, 0],
                stroke_width=1.0 + 4.0 * w,
                color=COLOR_OUTPUT,
            ).set_opacity(0.25 + 0.75 * (w / w_max))
            ctx_lines.append(line)

        # Show all context lines at once
        self.play(
            FadeIn(context_group),
            *[Create(cl) for cl in ctx_lines],
            run_time=0.8,
        )
        self.wait(0.4)

        # *** HIGHLIGHT the needle line ***
        anims = []
        for i, cl in enumerate(ctx_lines):
            if i == self.NEEDLE_IDX:
                new_line = cl.copy().set_color(COLOR_NEEDLE).set_opacity(1.0)
                new_line.set_stroke(width=5.0)
                anims.append(Transform(cl, new_line))
            else:
                # Keep non-needle lines semi-visible
                anims.append(cl.animate.set_opacity(0.25))
        self.play(*anims, run_time=0.6)

        # Highlight needle bar + cell
        self.play(
            Indicate(bars[self.NEEDLE_IDX], color=COLOR_NEEDLE, scale_factor=1.1),
            Indicate(h_cells[self.NEEDLE_IDX], color=COLOR_NEEDLE, scale_factor=1.1),
            run_time=0.6,
        )
        self.wait(0.3)

        # ============== Result (larger, prominent) ==============
        # Fade out step label
        self.play(FadeOut(step_lbl), run_time=0.2)

        result_label = (
            MathTex(
                r"c \approx 0.72 \cdot h_4",
                r"\quad \Rightarrow \quad",
                r"\textbf{7392}",
            ).scale(0.58).move_to([0, self.Y_RESULT, 0]))
        result_label[0].set_color(COLOR_OUTPUT)
        result_label[1].set_color(WHITE)
        result_label[2].set_color(COLOR_NEEDLE)

        self.play(FadeIn(result_label), run_time=0.5)
        self.wait(0.5)

        # ============== Conclusion ==============
        conclusion = (
            MathTex(
                r"\text{Attention} \to \text{direct access to all } h_i"
                r"\text{ --- needle found!}")
            .scale(0.45).set_color(COLOR_ATTN)
            .move_to([0, self.Y_CONCLUSION, 0]))
        self.play(FadeIn(conclusion), run_time=0.5)
        self.wait(2.0)
