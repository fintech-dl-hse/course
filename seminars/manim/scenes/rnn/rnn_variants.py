"""RNN Variants: Two-layer, Bidirectional, and 2-layer Bidirectional.

Three-column comparison side by side.
Bottom-to-top flow, 3 timesteps ("the", "cat", "sat").
High-level — no weight matrices or projections.

Animation order follows correct computation graph:
- Forward cells: left-to-right, one token at a time.
- Backward cells: right-to-left, one token at a time.
- Each cell appears with its input arrow + chain arrow from previous cell.
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
    PURPLE,
    RIGHT,
    RoundedRectangle,
    Scene,
    Tex,
    UP,
    VGroup,
    WHITE,
    YELLOW,
    Write,
)

from shared.neural import TensorColumn


COLOR_L1 = BLUE
COLOR_L2 = ORANGE
COLOR_HIDDEN = GREEN
COLOR_CONCAT = YELLOW
COLOR_BWD = "#CC66FF"     # purple-ish for backward


class RNNVariants(Scene):
    """Three-column: two-layer / bidirectional / 2-layer bidirectional."""

    TOKENS = ["the", "cat", "sat"]
    N = 3

    CELL_W = 0.75
    CELL_H = 0.44

    # Vertical
    Y_TOKEN = -3.30
    Y_R1 = -1.70            # row 1 (layer 1 / fwd+bwd 1)
    Y_R2 = -0.10            # row 2 (layer 2 / fwd+bwd 2)
    Y_OUTPUT = 1.30
    Y_TITLE = 3.40
    Y_SUBTITLE = 2.70

    # Three columns
    X_COL = [-4.60, 0.00, 4.60]
    X_STEP = 1.30

    # Animation speed
    T_CELL = 0.5        # time to show one cell
    T_FAST = 0.3        # fast fade
    T_PAUSE = 0.4       # pause between stages

    def _xs(self, col: int) -> list[float]:
        c = self.X_COL[col]
        return [c + (i - 1) * self.X_STEP for i in range(self.N)]

    def _cell(self, x: float, y: float, label: str, color) -> VGroup:
        box = RoundedRectangle(
            corner_radius=0.06, width=self.CELL_W, height=self.CELL_H,
            color=color, stroke_width=2,
        ).set_fill(color, opacity=0.12).move_to([x, y, 0])
        lbl = Tex(rf"\textbf{{{label}}}").scale(0.55).move_to(box.get_center())
        return VGroup(box, lbl)

    def _h_arrow(self, cell_from: VGroup, cell_to: VGroup, color,
                 reverse: bool = False):
        """Single horizontal chain arrow between two cells."""
        if reverse:
            return Arrow(
                start=cell_from[0].get_left(), end=cell_to[0].get_right(),
                buff=0.04, stroke_width=2, color=color, tip_length=0.07,
            )
        return Arrow(
            start=cell_from[0].get_right(), end=cell_to[0].get_left(),
            buff=0.04, stroke_width=2, color=color, tip_length=0.07,
        )

    # ------------------------------------------------------------------
    # Token-by-token animation helpers
    # ------------------------------------------------------------------

    def _animate_fwd_row(self, cells, input_arrows, color, run_time=None):
        """Animate a forward row left-to-right, cell by cell."""
        rt = run_time or self.T_CELL
        for i in range(self.N):
            anims = [FadeIn(cells[i]), Create(input_arrows[i])]
            if i > 0:
                h_arr = self._h_arrow(cells[i - 1], cells[i], color)
                anims.append(Create(h_arr))
            self.play(*anims, run_time=rt)

    def _animate_bwd_row(self, cells, input_arrows, color, run_time=None):
        """Animate a backward row right-to-left, cell by cell."""
        rt = run_time or self.T_CELL
        for idx in range(self.N - 1, -1, -1):
            anims = [FadeIn(cells[idx]), Create(input_arrows[idx])]
            if idx < self.N - 1:
                h_arr = self._h_arrow(cells[idx + 1], cells[idx], color,
                                      reverse=True)
                anims.append(Create(h_arr))
            self.play(*anims, run_time=rt)

    def construct(self) -> None:
        # ================ Title ================
        title = (
            Tex(r"\textbf{RNN Variants}")
            .scale(0.65)
            .move_to([0, self.Y_TITLE, 0])
        )
        self.play(Write(title), run_time=0.6)

        # Column titles
        col_titles = [
            Tex(r"\textbf{2-Layer}").scale(0.58),
            Tex(r"\textbf{Bidirectional}").scale(0.58),
            Tex(r"\textbf{2-Layer BiDir}").scale(0.58),
        ]
        for i, ct in enumerate(col_titles):
            ct.move_to([self.X_COL[i], self.Y_SUBTITLE, 0])

        # Separators
        seps = []
        for sx in [-2.30, 2.30]:
            s = DashedLine(
                start=[sx, self.Y_TITLE - 0.20, 0], end=[sx, -3.50, 0],
                stroke_width=1, color=GREY, dash_length=0.10,
            ).set_opacity(0.35)
            seps.append(s)

        self.play(
            *[FadeIn(ct) for ct in col_titles],
            *[FadeIn(s) for s in seps],
            run_time=0.5,
        )
        self.wait(self.T_PAUSE)

        # ================================================================
        # COLUMN 1: Two-layer RNN
        # ================================================================
        xs = self._xs(0)

        # Tokens
        toks1 = [Tex(rf"\textit{{{t}}}").scale(0.62).move_to([xs[i], self.Y_TOKEN, 0])
                  for i, t in enumerate(self.TOKENS)]
        self.play(*[FadeIn(t) for t in toks1], run_time=self.T_FAST)

        # Layer 1 — forward, token by token
        l1 = [self._cell(xs[i], self.Y_R1, "RNN", COLOR_L1) for i in range(self.N)]
        l1_in = [Arrow(start=toks1[i].get_top(), end=l1[i][0].get_bottom(),
                        buff=0.10, stroke_width=1.5, tip_length=0.07)
                  for i in range(self.N)]
        self._animate_fwd_row(l1, l1_in, COLOR_L1)

        # Layer 2 — forward, token by token
        l2 = [self._cell(xs[i], self.Y_R2, "RNN", COLOR_L2) for i in range(self.N)]
        l2_up = [Arrow(start=l1[i][0].get_top(), end=l2[i][0].get_bottom(),
                        buff=0.06, stroke_width=1.5, color=COLOR_HIDDEN, tip_length=0.07)
                  for i in range(self.N)]
        self._animate_fwd_row(l2, l2_up, COLOR_L2)

        # Output
        out1 = [MathTex(rf"h^2_{i+1}").scale(0.58).set_color(COLOR_L2)
                .move_to([xs[i], self.Y_OUTPUT, 0]) for i in range(self.N)]
        out1_arr = [Arrow(start=l2[i][0].get_top(), end=out1[i].get_bottom(),
                          buff=0.10, stroke_width=1.5, color=COLOR_L2, tip_length=0.07)
                    for i in range(self.N)]
        self.play(*[FadeIn(o) for o in out1], *[Create(a) for a in out1_arr],
                  run_time=self.T_CELL)

        note1 = MathTex(r"\text{depth}").scale(0.58).move_to(
            [self.X_COL[0], self.Y_OUTPUT + 0.40, 0])
        self.play(FadeIn(note1), run_time=self.T_FAST)
        self.wait(self.T_PAUSE)

        # ================================================================
        # COLUMN 2: Bidirectional RNN (1 layer)
        # ================================================================
        xs = self._xs(1)
        Y_FWD = self.Y_R1 - 0.25
        Y_BWD = self.Y_R1 + 0.25

        toks2 = [Tex(rf"\textit{{{t}}}").scale(0.62).move_to([xs[i], self.Y_TOKEN, 0])
                  for i, t in enumerate(self.TOKENS)]
        self.play(*[FadeIn(t) for t in toks2], run_time=self.T_FAST)

        # Forward — left to right
        fwd = [self._cell(xs[i], Y_FWD, "F", COLOR_L1) for i in range(self.N)]
        fwd_in = [Arrow(start=toks2[i].get_top(), end=fwd[i][0].get_bottom(),
                         buff=0.10, stroke_width=1.5, tip_length=0.07)
                   for i in range(self.N)]
        self._animate_fwd_row(fwd, fwd_in, COLOR_L1)

        # Backward — right to left (input from same tokens via FWD)
        bwd = [self._cell(xs[i], Y_BWD, "B", COLOR_BWD) for i in range(self.N)]
        bwd_up = [Arrow(start=fwd[i][0].get_top(), end=bwd[i][0].get_bottom(),
                         buff=0.03, stroke_width=1, color=GREY, tip_length=0.06)
                   for i in range(self.N)]
        self._animate_bwd_row(bwd, bwd_up, COLOR_BWD)

        # Concat output
        out2 = [MathTex(rf"[\vec{{h}};\overleftarrow{{h}}]_{i+1}")
                .scale(0.48).set_color(COLOR_CONCAT)
                .move_to([xs[i], self.Y_OUTPUT, 0]) for i in range(self.N)]
        out2_arr = [Arrow(start=bwd[i][0].get_top(), end=out2[i].get_bottom(),
                          buff=0.10, stroke_width=1.5, color=COLOR_CONCAT, tip_length=0.07)
                    for i in range(self.N)]
        self.play(*[FadeIn(o) for o in out2], *[Create(a) for a in out2_arr],
                  run_time=self.T_CELL)

        note2 = MathTex(r"\text{full context}").scale(0.58).move_to(
            [self.X_COL[1], self.Y_OUTPUT + 0.40, 0])
        self.play(FadeIn(note2), run_time=self.T_FAST)
        self.wait(self.T_PAUSE)

        # ================================================================
        # COLUMN 3: 2-Layer Bidirectional RNN
        # ================================================================
        xs = self._xs(2)

        # Row 1: FWD1 + BWD1
        Y_F1 = self.Y_R1 - 0.25
        Y_B1 = self.Y_R1 + 0.25
        # Row 2: FWD2 + BWD2
        Y_F2 = self.Y_R2 - 0.25
        Y_B2 = self.Y_R2 + 0.25

        toks3 = [Tex(rf"\textit{{{t}}}").scale(0.62).move_to([xs[i], self.Y_TOKEN, 0])
                  for i, t in enumerate(self.TOKENS)]
        self.play(*[FadeIn(t) for t in toks3], run_time=self.T_FAST)

        # Layer 1 forward — left to right
        f1 = [self._cell(xs[i], Y_F1, "F", COLOR_L1) for i in range(self.N)]
        f1_in = [Arrow(start=toks3[i].get_top(), end=f1[i][0].get_bottom(),
                        buff=0.10, stroke_width=1.5, tip_length=0.07)
                  for i in range(self.N)]
        self._animate_fwd_row(f1, f1_in, COLOR_L1)

        # Layer 1 backward — right to left
        b1 = [self._cell(xs[i], Y_B1, "B", COLOR_BWD) for i in range(self.N)]
        b1_up = [Arrow(start=f1[i][0].get_top(), end=b1[i][0].get_bottom(),
                        buff=0.03, stroke_width=1, color=GREY, tip_length=0.06)
                  for i in range(self.N)]
        self._animate_bwd_row(b1, b1_up, COLOR_BWD)

        # L1 label
        l1_lbl = MathTex(r"\text{L1}").scale(0.60).set_color(COLOR_L1)
        l1_lbl.next_to(VGroup(f1[0], b1[0]), LEFT, buff=0.08)

        # Layer 2 forward — left to right
        f2 = [self._cell(xs[i], Y_F2, "F", COLOR_L1) for i in range(self.N)]
        f2_up = [Arrow(start=b1[i][0].get_top(), end=f2[i][0].get_bottom(),
                        buff=0.06, stroke_width=1.5, color=COLOR_HIDDEN, tip_length=0.07)
                  for i in range(self.N)]

        self.play(FadeIn(l1_lbl), run_time=self.T_FAST)
        self._animate_fwd_row(f2, f2_up, COLOR_L1)

        # Layer 2 backward — right to left
        b2 = [self._cell(xs[i], Y_B2, "B", COLOR_BWD) for i in range(self.N)]
        b2_up = [Arrow(start=f2[i][0].get_top(), end=b2[i][0].get_bottom(),
                        buff=0.03, stroke_width=1, color=GREY, tip_length=0.06)
                  for i in range(self.N)]

        l2_lbl = MathTex(r"\text{L2}").scale(0.60).set_color(COLOR_L2)
        l2_lbl.next_to(VGroup(f2[0], b2[0]), LEFT, buff=0.08)

        self._animate_bwd_row(b2, b2_up, COLOR_BWD)
        self.play(FadeIn(l2_lbl), run_time=self.T_FAST)

        # Output
        out3 = [MathTex(rf"[\vec{{h}}^2;\overleftarrow{{h}}^2]_{i+1}")
                .scale(0.46).set_color(COLOR_CONCAT)
                .move_to([xs[i], self.Y_OUTPUT, 0]) for i in range(self.N)]
        out3_arr = [Arrow(start=b2[i][0].get_top(), end=out3[i].get_bottom(),
                          buff=0.10, stroke_width=1.5, color=COLOR_CONCAT, tip_length=0.07)
                    for i in range(self.N)]
        self.play(*[FadeIn(o) for o in out3], *[Create(a) for a in out3_arr],
                  run_time=self.T_CELL)

        note3 = MathTex(r"\text{depth + context}").scale(0.58).move_to(
            [self.X_COL[2], self.Y_OUTPUT + 0.40, 0])
        self.play(FadeIn(note3), run_time=self.T_FAST)
        self.wait(self.T_PAUSE)

        # ================ Bottom summary ================
        summary = (
            MathTex(
                r"\text{stacked} \to \text{depth}"
                r"\quad | \quad"
                r"\text{bidir} \to \text{past + future}"
                r"\quad | \quad"
                r"\text{both} \to \text{BERT, ELMo}"
            )
            .scale(0.42)
            .move_to([0, -3.65, 0])
        )
        self.play(FadeIn(summary), run_time=0.5)
        self.wait(2.0)
