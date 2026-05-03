"""Positional-encoding scene for seminar 09 (V09 of the curriculum catalog).

Motivates positional encoding via permutation invariance of self-attention.

1. Title + subtitle formula.
2. Phase A — two side-by-side tracks (Track A / Track B) showing that
   self-attention without PE produces identical output sets regardless of
   token order. Bottom-to-top flow: tokens at bottom, outputs at top.
3. Phase B — introduce PE: three "x_t + PE_t = x~_t" panels.
4. Phase C — re-run the two tracks WITH PE; outputs now differ.

Color convention:
- BLUE — input embeddings x_t.
- PURPLE — self-attention outputs z_t.
- GREY_B — positional encoding PE_t.
- TEAL — enriched embedding x~_t = x_t + PE_t.
"""
from __future__ import annotations

from typing import Any

from manim import (
    Arrow,
    BLUE,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    GREY_B,
    LEFT,
    MathTex,
    PURPLE,
    RIGHT,
    Scene,
    Square,
    SurroundingRectangle,
    Tex,
    TEAL,
    UP,
    VGroup,
    VMobject,
    WHITE,
    YELLOW,
    Write,
)

from shared.neural import LabeledBox, TensorColumn, arrow_between


def _vertical_arrow(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    """Arrow that always attaches to top/bottom edges (forced vertical)."""
    defaults: dict[str, Any] = {"buff": 0.06, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[1] >= b.get_center()[1]:
        start, end = a.get_bottom(), b.get_top()
    else:
        start, end = a.get_top(), b.get_bottom()
    return Arrow(start=start, end=end, **defaults)


class PositionalEncoding(Scene):
    """V09: permutation invariance of self-attention and the PE fix."""

    TENSOR_DIM = 4
    CELL = 0.26
    PE_CELL = 0.28

    # --- Vertical layout (bottom-to-top flow) ---
    Y_TITLE = 3.65
    Y_SUBTITLE = 3.25

    # Track A (left half of screen)
    # Bottom-to-top: tokens -> x -> self-attn box -> z outputs
    Y_TOKEN = -3.20
    Y_X = -2.30
    Y_BOX = -0.90
    Y_Z = 0.40
    Y_Z_LABEL = 1.30

    # Punchline between z outputs and title
    Y_PUNCHLINE = 2.00

    # Horizontal: Track A on left, Track B on right
    X_TRACK_A = -3.60
    X_TRACK_B = 3.60
    X_TOK_SPACING = 1.60

    # Self-attention box width (spans 3 columns)
    BOX_W = 4.20
    BOX_H = 0.65

    # Phase B (PE introduction) — 3 panels centered
    Y_PE_ROW = 0.40
    Y_PE_LABELS = -0.80
    Y_PE_CAPTION = -2.40
    X_PE_SPACING = 4.40

    # --- Toy vector patterns (opacity triplets → now 4-tuples) ---
    X_PATTERNS = [
        (0.85, 0.30, 0.30, 0.50),   # x_1 "the": top-heavy
        (0.30, 0.85, 0.30, 0.40),   # x_2 "cat": middle-heavy
        (0.30, 0.30, 0.85, 0.70),   # x_3 "sat": bottom-heavy
    ]
    Z_PATTERNS = [
        (0.70, 0.50, 0.30, 0.55),   # z_1
        (0.40, 0.75, 0.40, 0.50),   # z_2
        (0.30, 0.50, 0.70, 0.60),   # z_3
    ]
    PE_PATTERNS = [
        (0.85, 0.55, 0.20, 0.40),   # PE_1
        (0.55, 0.20, 0.85, 0.30),   # PE_2
        (0.20, 0.85, 0.55, 0.70),   # PE_3
    ]
    Z_TILDE_A_PATTERNS = [
        (0.85, 0.20, 0.40, 0.60),
        (0.30, 0.85, 0.45, 0.50),
        (0.45, 0.35, 0.85, 0.70),
    ]
    Z_TILDE_B_PATTERNS = [
        (0.30, 0.85, 0.55, 0.45),
        (0.85, 0.50, 0.25, 0.65),
        (0.55, 0.30, 0.85, 0.40),
    ]

    TOKENS = ["the", "cat", "sat"]
    B_TOKEN_ORDER = [2, 0, 1]  # sat, the, cat

    # ----------------------------- helpers -----------------------------
    def _build_tensor_with_pattern(
        self,
        pattern: tuple[float, ...],
        color: str,
        cell_size: float | None = None,
    ) -> TensorColumn:
        cs = cell_size if cell_size is not None else self.CELL
        col = TensorColumn(
            dim=self.TENSOR_DIM,
            cell_size=cs,
            color=color,
            fill_opacity=0.30,
        )
        for k, cell in enumerate(col.cells):
            cell.set_fill(color, opacity=float(pattern[k]))
        return col

    def _tok_xs(self, track_cx: float) -> list[float]:
        """Return 3 x-positions centered on track_cx."""
        return [track_cx + (j - 1) * self.X_TOK_SPACING for j in range(3)]

    # ----------------------------- main -----------------------------
    def construct(self) -> None:
        # ===================== Title =====================
        title = (
            Tex(r"Self-attention is permutation invariant")
            .scale(0.65)
            .move_to([0.0, self.Y_TITLE, 0.0])
        )
        subtitle = (
            MathTex(r"\text{fix: } \tilde{x}_t = x_t + PE_t")
            .scale(0.58)
            .move_to([0.0, self.Y_SUBTITLE, 0.0])
        )
        self.play(Write(title), run_time=0.5)
        self.play(FadeIn(subtitle), run_time=0.3)
        self.wait(0.2)

        # ===================== Phase A: two tracks WITHOUT PE =====================
        track_a = self._build_track(
            track_cx=self.X_TRACK_A,
            tokens=[self.TOKENS[i] for i in [0, 1, 2]],
            x_indices=[0, 1, 2],
            z_indices=[0, 1, 2],
            track_label_str=r"\text{Track A}",
            with_pe=False,
        )
        track_b = self._build_track(
            track_cx=self.X_TRACK_B,
            tokens=[self.TOKENS[i] for i in self.B_TOKEN_ORDER],
            x_indices=list(self.B_TOKEN_ORDER),
            z_indices=list(self.B_TOKEN_ORDER),
            track_label_str=r"\text{Track B}",
            with_pe=False,
        )

        self._animate_track_in(track_a)
        self._animate_track_in(track_b)
        self.wait(0.3)

        # Highlight matching outputs: z_1 in Track A == z_1 in Track B etc.
        # Brief yellow flash on the z output columns
        z_a_cols = [c for c, _ in track_a["z_cols"]]
        z_b_cols = [c for c, _ in track_b["z_cols"]]
        highlights_a = [
            SurroundingRectangle(c, color=YELLOW, buff=0.04, stroke_width=2.5)
            for c in z_a_cols
        ]
        highlights_b = [
            SurroundingRectangle(c, color=YELLOW, buff=0.04, stroke_width=2.5)
            for c in z_b_cols
        ]
        self.play(
            *[Create(h) for h in highlights_a + highlights_b],
            run_time=0.4,
        )

        punchline = (
            Tex(r"outputs are identical up to permutation")
            .scale(0.58)
            .move_to([0.0, self.Y_PUNCHLINE, 0.0])
        )
        self.play(FadeIn(punchline), run_time=0.4)
        self.wait(0.8)

        # ===================== Clear Phase A =====================
        a_mobs = self._track_mobs(track_a)
        b_mobs = self._track_mobs(track_b)
        cleanup = [FadeOut(m) for m in (
            *a_mobs, *b_mobs, punchline,
            *highlights_a, *highlights_b,
        )]
        self.play(*cleanup, run_time=0.5)
        for m in (*a_mobs, *b_mobs, punchline, *highlights_a, *highlights_b):
            self.remove(m)

        # ===================== Phase B: introduce PE =====================
        pe_panels: list[dict[str, Any]] = []
        for t in range(3):
            cx = (t - 1) * self.X_PE_SPACING
            pe_panels.append(self._build_pe_panel(t, cx))

        for panel in pe_panels:
            self.play(
                FadeIn(panel["x_col"]),
                FadeIn(panel["x_lbl"]),
                run_time=0.25,
            )
            self.play(
                FadeIn(panel["plus"]),
                FadeIn(panel["pe_col"]),
                FadeIn(panel["pe_lbl"]),
                run_time=0.25,
            )
            self.play(
                FadeIn(panel["eq"]),
                FadeIn(panel["xt_col"]),
                FadeIn(panel["xt_lbl"]),
                run_time=0.25,
            )
        self.wait(0.4)

        pe_caption = (
            Tex(r"each position now carries a unique signature")
            .scale(0.55)
            .move_to([0.0, self.Y_PE_CAPTION, 0.0])
        )
        self.play(FadeIn(pe_caption), run_time=0.3)
        self.wait(0.7)

        all_pe_mobs: list[VMobject] = [pe_caption]
        for panel in pe_panels:
            all_pe_mobs.extend(panel["all_mobs"])
        self.play(*[FadeOut(m) for m in all_pe_mobs], run_time=0.4)
        for m in all_pe_mobs:
            self.remove(m)

        # ===================== Phase C: two tracks WITH PE =====================
        track_a2 = self._build_track(
            track_cx=self.X_TRACK_A,
            tokens=[self.TOKENS[i] for i in [0, 1, 2]],
            x_indices=[0, 1, 2],
            z_indices=[0, 1, 2],
            track_label_str=r"\text{Track A (with PE)}",
            with_pe=True,
            z_patterns_override=self.Z_TILDE_A_PATTERNS,
        )
        track_b2 = self._build_track(
            track_cx=self.X_TRACK_B,
            tokens=[self.TOKENS[i] for i in self.B_TOKEN_ORDER],
            x_indices=list(self.B_TOKEN_ORDER),
            z_indices=list(self.B_TOKEN_ORDER),
            track_label_str=r"\text{Track B (with PE)}",
            with_pe=True,
            z_patterns_override=self.Z_TILDE_B_PATTERNS,
        )
        self._animate_track_in(track_a2)
        self._animate_track_in(track_b2)

        # Highlight that outputs NOW DIFFER
        z_a2_cols = [c for c, _ in track_a2["z_cols"]]
        z_b2_cols = [c for c, _ in track_b2["z_cols"]]
        diff_highlights = []
        for c in z_a2_cols:
            diff_highlights.append(
                SurroundingRectangle(c, color=TEAL, buff=0.04, stroke_width=2.5)
            )
        for c in z_b2_cols:
            diff_highlights.append(
                SurroundingRectangle(c, color=PURPLE, buff=0.04, stroke_width=2.5)
            )
        self.play(*[Create(h) for h in diff_highlights], run_time=0.4)

        diff_label = (
            Tex(r"outputs now differ --- order matters!")
            .scale(0.58)
            .move_to([0.0, self.Y_PUNCHLINE, 0.0])
        )
        self.play(FadeIn(diff_label), run_time=0.4)
        self.wait(1.0)

    # ----------------------------- track builder -----------------------------
    def _build_track(
        self,
        *,
        track_cx: float,
        tokens: list[str],
        x_indices: list[int],
        z_indices: list[int],
        track_label_str: str,
        with_pe: bool,
        z_patterns_override: list[tuple[float, ...]] | None = None,
    ) -> dict[str, Any]:
        x_color = TEAL if with_pe else BLUE
        z_color = PURPLE

        tok_xs = self._tok_xs(track_cx)

        # Token labels (bottom)
        tok_mobs: list[Tex] = []
        for k, tok in enumerate(tokens):
            t = (
                Tex(rf"\textit{{``{tok}''}}")
                .scale(0.58)
                .move_to([tok_xs[k], self.Y_TOKEN, 0.0])
            )
            tok_mobs.append(t)

        # Embedding columns x_t (above tokens)
        x_cols: list[tuple[TensorColumn, MathTex]] = []
        for k, orig_idx in enumerate(x_indices):
            label = f"\\tilde{{x}}_{{{orig_idx + 1}}}" if with_pe else f"x_{orig_idx + 1}"
            col = self._build_tensor_with_pattern(
                self.X_PATTERNS[orig_idx], x_color
            )
            col.move_to([tok_xs[k], self.Y_X, 0.0])
            lbl = (
                MathTex(label)
                .scale(0.58)
                .next_to(col, RIGHT, buff=0.08)
            )
            x_cols.append((col, lbl))

        # Self-attention box (middle)
        attn_box = LabeledBox(
            label=r"\mathrm{self\text{-}attn}",
            width=self.BOX_W,
            height=self.BOX_H,
            label_scale=0.58,
            color=GREY_B,
        ).move_to([track_cx, self.Y_BOX, 0.0])

        # Output z columns (top)
        z_cols: list[tuple[TensorColumn, MathTex]] = []
        z_pats = z_patterns_override if z_patterns_override is not None else self.Z_PATTERNS
        for k, orig_idx in enumerate(z_indices):
            zcol = self._build_tensor_with_pattern(
                z_pats[orig_idx], z_color
            )
            zcol.move_to([tok_xs[k], self.Y_Z, 0.0])
            zlbl = (
                MathTex(rf"z_{orig_idx + 1}")
                .scale(0.58)
                .next_to(zcol, UP, buff=0.08)
            )
            z_cols.append((zcol, zlbl))

        # Track label above everything
        track_lbl = (
            MathTex(track_label_str)
            .scale(0.55)
            .move_to([track_cx, self.Y_Z_LABEL, 0.0])
        )

        # Arrows: x → box (bottom-to-top)
        x_to_box_arrows: list[Arrow] = []
        for col, _lbl in x_cols:
            arr = _vertical_arrow(
                col, attn_box, buff=0.08, tip_length=0.13,
                color=WHITE, stroke_width=2.5,
            )
            x_to_box_arrows.append(arr)

        # Arrows: box → z (bottom-to-top)
        box_to_z_arrows: list[Arrow] = []
        for zcol, _zlbl in z_cols:
            arr = _vertical_arrow(
                attn_box, zcol, buff=0.08, tip_length=0.13,
                color=WHITE, stroke_width=2.5,
            )
            box_to_z_arrows.append(arr)

        return {
            "tok_mobs": tok_mobs,
            "x_cols": x_cols,
            "attn_box": attn_box,
            "z_cols": z_cols,
            "track_lbl": track_lbl,
            "x_to_box_arrows": x_to_box_arrows,
            "box_to_z_arrows": box_to_z_arrows,
        }

    def _animate_track_in(self, track: dict[str, Any]) -> None:
        """Bottom-to-top: tokens → x → box → z."""
        self.play(
            *[FadeIn(t) for t in track["tok_mobs"]],
            FadeIn(track["track_lbl"]),
            run_time=0.30,
        )
        self.play(
            *[FadeIn(c) for c, _ in track["x_cols"]],
            *[FadeIn(lbl) for _, lbl in track["x_cols"]],
            run_time=0.30,
        )
        self.play(
            FadeIn(track["attn_box"]),
            *[Create(a) for a in track["x_to_box_arrows"]],
            run_time=0.40,
        )
        self.play(
            *[Create(a) for a in track["box_to_z_arrows"]],
            *[FadeIn(c) for c, _ in track["z_cols"]],
            *[FadeIn(lbl) for _, lbl in track["z_cols"]],
            run_time=0.45,
        )

    def _track_mobs(self, track: dict[str, Any]) -> list[VMobject]:
        flat: list[VMobject] = []
        flat.extend(track["tok_mobs"])
        for c, lbl in track["x_cols"]:
            flat.append(c)
            flat.append(lbl)
        flat.append(track["attn_box"])
        for c, lbl in track["z_cols"]:
            flat.append(c)
            flat.append(lbl)
        flat.append(track["track_lbl"])
        flat.extend(track["x_to_box_arrows"])
        flat.extend(track["box_to_z_arrows"])
        return flat

    # ----------------------------- PE panel builder -----------------------------
    def _build_pe_panel(self, t: int, cx: float) -> dict[str, Any]:
        col_dx = 1.15
        x_col = self._build_tensor_with_pattern(
            self.X_PATTERNS[t], BLUE, cell_size=self.PE_CELL
        ).move_to([cx - col_dx, self.Y_PE_ROW, 0.0])
        pe_col = self._build_tensor_with_pattern(
            self.PE_PATTERNS[t], GREY_B, cell_size=self.PE_CELL
        ).move_to([cx, self.Y_PE_ROW, 0.0])
        xt_pattern = tuple(
            min(1.0, self.X_PATTERNS[t][k] + self.PE_PATTERNS[t][k])
            for k in range(self.TENSOR_DIM)
        )
        xt_col = self._build_tensor_with_pattern(
            xt_pattern, TEAL, cell_size=self.PE_CELL
        ).move_to([cx + col_dx, self.Y_PE_ROW, 0.0])

        plus = (
            MathTex(r"+")
            .scale(1.30)
            .move_to([cx - col_dx / 2.0, self.Y_PE_ROW, 0.0])
        )
        eq = (
            MathTex(r"=")
            .scale(1.30)
            .move_to([cx + col_dx / 2.0, self.Y_PE_ROW, 0.0])
        )

        x_lbl = (
            MathTex(rf"x_{t + 1}")
            .scale(0.65)
            .next_to(x_col, DOWN, buff=0.12)
        )
        pe_lbl = (
            MathTex(rf"PE_{t + 1}")
            .scale(0.65)
            .next_to(pe_col, DOWN, buff=0.12)
        )
        xt_lbl = (
            MathTex(rf"\tilde{{x}}_{t + 1}")
            .scale(0.65)
            .next_to(xt_col, DOWN, buff=0.12)
        )

        all_mobs = [x_col, pe_col, xt_col, plus, eq, x_lbl, pe_lbl, xt_lbl]
        return {
            "x_col": x_col, "pe_col": pe_col, "xt_col": xt_col,
            "plus": plus, "eq": eq,
            "x_lbl": x_lbl, "pe_lbl": pe_lbl, "xt_lbl": xt_lbl,
            "all_mobs": all_mobs,
        }
