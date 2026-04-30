"""Positional-encoding scene for seminar 09 (V09 of the curriculum catalog).

Мотивирует positional encoding через permutation invariance self-attention'а.

1. Сверху — заголовок
   ``\\text{Self-attention is permutation invariant}``
   и подзаголовок ``\\text{fix: } \\tilde{x}_t = x_t + PE_t``.
2. Phase A — две параллельные «дорожки» self-attention БЕЗ positional encoding.
   - Track A: токены ``the / cat / sat`` → эмбеддинги ``x_1, x_2, x_3``.
   - Track B: те же токены, переставленные в порядок ``sat / the / cat`` →
     эмбеддинги ``x_3, x_1, x_2`` (те же векторы, переставленные).
   Каждая дорожка проходит через self-attention блок и выдаёт выходы
   ``z_t``. Вывод: множества выходов совпадают (``{z_1, z_2, z_3}`` против
   ``{z_3, z_1, z_2}`` — те же векторы, в другом порядке). Punchline:
   ``\\text{outputs are identical up to permutation}``.
3. Phase B — введение positional encoding. Снизу — три ``PE_1, PE_2, PE_3``
   столбца, каждый с уникальной "синусоидальной" заливкой ячеек (различная
   opacity по ячейкам — характерные паттерны на каждой позиции). Затем
   анимация ``\\tilde{x}_t = x_t + PE_t`` для каждого ``t``: pure-визуальный
   "+" между двумя столбцами и резалт-столбец справа.
4. Phase C — re-run на тех же двух дорожках, но уже с ``\\tilde{x}``. Теперь
   self-attention видит, что у токенов разные позиционные сигналы, и выходы
   различаются (визуально — выделены разным паттерном подсветки).

Сцена использует общие примитивы ``shared.neural`` (TensorColumn,
LabeledBox, arrow_between). Локальная копия ``_horizontal_arrow`` (как в
``ScaledDotProductAttention`` и ``RNNForward``) обеспечивает корректную
маршрутизацию горизонтальных стрелок.

Цветовая конвенция:
- ``BLUE`` — обычные эмбеддинги ``x_t``.
- ``PURPLE`` — выходы self-attention ``z_t``.
- ``GREY_B`` — positional encoding ``PE_t`` (ненасыщенный, чтобы не
  конфликтовать с Q/K/V палитрой из V06/V07).
- ``TEAL`` — итоговый ``\\tilde{x}_t = x_t + PE_t`` (новый цвет — сигнал,
  что это "обогащённый позицией" эмбеддинг).
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
    Tex,
    TEAL,
    UP,
    VGroup,
    VMobject,
    WHITE,
    Write,
)

from shared.neural import LabeledBox, TensorColumn, arrow_between


def _horizontal_arrow(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    """Arrow that always attaches to right/left edges (forced horizontal)."""
    defaults: dict[str, Any] = {"buff": 0.08, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


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

    # Layout constants tuned for 720p (-qm) frame ±(7.11, 4.0).
    TENSOR_DIM = 3
    CELL = 0.20            # tensor-column cell size in two-track phases
    PE_CELL = 0.24         # bigger cells in PE-introduction phase

    # Title anchors.
    Y_TITLE = 3.65
    Y_SUBTITLE = 3.25

    # Per-track row positions (Phase A and Phase C share these).
    # Track A occupies upper half (y > 0), Track B occupies lower half.
    # Computed so z-labels clear the punchline and title clears Track A tokens.
    Y_TRACK_A_TOKEN = 2.65
    Y_TRACK_A_X = 2.10
    Y_TRACK_A_BOX = 1.40
    Y_TRACK_A_Z = 0.70

    Y_TRACK_B_TOKEN = -0.95
    Y_TRACK_B_X = -1.50
    Y_TRACK_B_BOX = -2.20
    Y_TRACK_B_Z = -2.90

    # Punchline sits between Track A z-label band (~y=0.25) and Track B
    # token band (~y=-0.95) with margin on both sides.
    Y_PUNCHLINE = -0.15

    # Phase B (PE introduction). Center of the "x_t + PE_t = ~x_t" diagram.
    Y_PE_ROW = 0.60
    Y_PE_LABELS = -0.85    # captions like "x_t" / "PE_t" / "~x_t"
    Y_PE_CAPTION = -2.55   # final caption

    # Horizontal layout for tokens in a track (3 tokens spaced).
    X_TOK_BASE = -2.10
    X_TOK_SPACING = 1.05
    # Wider spacing to the self-attention box and outputs.
    X_BOX_LEFT = -2.10     # left edge of box stretches across tokens
    X_BOX_RIGHT = 2.10
    X_Z_BASE = -2.10
    X_Z_SPACING = 1.05

    # Phase B group geometry.
    X_PE_GROUP_BASE = -5.20
    X_PE_GROUP_SPACING = 3.50

    # ------- Toy "vector content" to make permutation-invariance visible. -------
    # Each x vector is a triplet of opacities for its 3 cells. Attention
    # (without PE) is permutation-invariant in the sense that running the
    # SAME multiset of inputs through self-attention produces the SAME multiset
    # of outputs. We pick three visually distinct patterns so the viewer
    # can see "z_1 from track A == z_1 from track B" up to permutation.
    X_PATTERNS = [
        (0.85, 0.30, 0.30),   # x_1 (token "the"): top-heavy
        (0.30, 0.85, 0.30),   # x_2 (token "cat"): middle-heavy
        (0.30, 0.30, 0.85),   # x_3 (token "sat"): bottom-heavy
    ]
    # Output z_t patterns (without PE). We make them mixtures that differ
    # from the inputs (so the viewer sees self-attention "did something")
    # but are still position-independent: track B at slot k has the SAME
    # output as track A's permuted slot.
    Z_PATTERNS = [
        (0.70, 0.50, 0.30),   # z_1
        (0.40, 0.75, 0.40),   # z_2
        (0.30, 0.50, 0.70),   # z_3
    ]
    # PE patterns — sinusoidal-flavored (varying per-cell opacity, distinct
    # per position). Grey shades; small enough to not be confused with x_t.
    PE_PATTERNS = [
        (0.85, 0.55, 0.20),   # PE_1
        (0.55, 0.20, 0.85),   # PE_2
        (0.20, 0.85, 0.55),   # PE_3
    ]
    # After x + PE, outputs become position-DEPENDENT. Track A outputs and
    # Track B outputs differ now (different cells highlighted).
    Z_TILDE_A_PATTERNS = [
        (0.85, 0.20, 0.40),   # z_1 (track A)
        (0.30, 0.85, 0.45),   # z_2 (track A)
        (0.45, 0.35, 0.85),   # z_3 (track A)
    ]
    Z_TILDE_B_PATTERNS = [
        (0.30, 0.85, 0.55),   # z_1 (track B) — different from A
        (0.85, 0.50, 0.25),   # z_2 (track B)
        (0.55, 0.30, 0.85),   # z_3 (track B)
    ]

    TOKENS = ["the", "cat", "sat"]
    # Permutation σ for Track B: (t=1 -> token "sat", t=2 -> "the", t=3 -> "cat")
    # Using 0-indexed: B_TOKEN_ORDER[k] is the original index of the token
    # placed at slot k in Track B.
    B_TOKEN_ORDER = [2, 0, 1]

    # ----------------------------- helpers -----------------------------
    def _build_tensor_with_pattern(
        self,
        pattern: tuple[float, float, float],
        color: str,
        cell_size: float | None = None,
    ) -> TensorColumn:
        """Build a TensorColumn whose individual cells have opacities from `pattern`.

        TensorColumn uses uniform fill_opacity by default; we override each
        cell after construction so the column carries a "per-position
        signature" the viewer can recognize across phases.
        """
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

    # ----------------------------- main -----------------------------
    def construct(self) -> None:
        # ===================== Title =====================
        title = (
            Tex(r"Self-attention is permutation invariant")
            .scale(0.62)
            .move_to([0.0, self.Y_TITLE, 0.0])
        )
        subtitle = (
            MathTex(r"\text{fix: } \tilde{x}_t = x_t + PE_t")
            .scale(0.55)
            .move_to([0.0, self.Y_SUBTITLE, 0.0])
        )
        self.play(Write(title), run_time=0.5)
        self.play(FadeIn(subtitle), run_time=0.3)
        self.wait(0.2)

        # ===================== Phase A: two tracks WITHOUT PE =====================
        # Track A objects.
        track_a = self._build_track(
            tokens=[self.TOKENS[i] for i in [0, 1, 2]],
            x_indices=[0, 1, 2],     # x_1, x_2, x_3 in slot order
            z_indices=[0, 1, 2],     # outputs z_1, z_2, z_3
            y_token=self.Y_TRACK_A_TOKEN,
            y_x=self.Y_TRACK_A_X,
            y_box=self.Y_TRACK_A_BOX,
            y_z=self.Y_TRACK_A_Z,
            track_label_str=r"\text{Track A}",
            with_pe=False,
        )
        track_b = self._build_track(
            tokens=[self.TOKENS[i] for i in self.B_TOKEN_ORDER],
            x_indices=list(self.B_TOKEN_ORDER),  # x_3, x_1, x_2 (same vectors)
            z_indices=list(self.B_TOKEN_ORDER),  # outputs reordered identically
            y_token=self.Y_TRACK_B_TOKEN,
            y_x=self.Y_TRACK_B_X,
            y_box=self.Y_TRACK_B_BOX,
            y_z=self.Y_TRACK_B_Z,
            track_label_str=r"\text{Track B}",
            with_pe=False,
        )

        # Animate Phase A in two batches (track A then track B).
        self._animate_track_in(track_a)
        self._animate_track_in(track_b)
        self.wait(0.4)

        # Punchline label between the tracks.
        punchline = (
            Tex(r"outputs are identical up to permutation")
            .scale(0.55)
            .move_to([0.0, self.Y_PUNCHLINE, 0.0])
        )
        self.play(FadeIn(punchline), run_time=0.4)
        self.wait(1.0)

        # ===================== Clear Phase A =====================
        a_mobs = self._track_mobs(track_a)
        b_mobs = self._track_mobs(track_b)
        cleanup = [FadeOut(m) for m in (*a_mobs, *b_mobs, punchline)]
        self.play(*cleanup, run_time=0.5)
        for m in (*a_mobs, *b_mobs, punchline):
            self.remove(m)

        # ===================== Phase B: introduce PE and the addition =====================
        # Build three "x_t + PE_t = ~x_t" panels side-by-side.
        pe_panels: list[dict[str, Any]] = []
        for t in range(3):
            cx = self.X_PE_GROUP_BASE + t * self.X_PE_GROUP_SPACING
            pe_panels.append(self._build_pe_panel(t, cx))

        # Animate per-panel: appear x, appear PE with "+", appear ~x with "=".
        for panel in pe_panels:
            self.play(
                FadeIn(panel["x_col"]),
                FadeIn(panel["x_lbl"]),
                run_time=0.30,
            )
            self.play(
                FadeIn(panel["plus"]),
                FadeIn(panel["pe_col"]),
                FadeIn(panel["pe_lbl"]),
                run_time=0.30,
            )
            self.play(
                FadeIn(panel["eq"]),
                FadeIn(panel["xt_col"]),
                FadeIn(panel["xt_lbl"]),
                run_time=0.30,
            )
        self.wait(0.5)

        pe_caption = (
            Tex(r"each position now carries a unique signature")
            .scale(0.50)
            .move_to([0.0, self.Y_PE_CAPTION, 0.0])
        )
        self.play(FadeIn(pe_caption), run_time=0.3)
        self.wait(0.8)

        # Cleanup Phase B.
        all_pe_mobs: list[VMobject] = [pe_caption]
        for panel in pe_panels:
            all_pe_mobs.extend(panel["all_mobs"])
        self.play(*[FadeOut(m) for m in all_pe_mobs], run_time=0.4)
        for m in all_pe_mobs:
            self.remove(m)

        # ===================== Phase C: two tracks WITH PE =====================
        track_a2 = self._build_track(
            tokens=[self.TOKENS[i] for i in [0, 1, 2]],
            x_indices=[0, 1, 2],
            z_indices=[0, 1, 2],
            y_token=self.Y_TRACK_A_TOKEN,
            y_x=self.Y_TRACK_A_X,
            y_box=self.Y_TRACK_A_BOX,
            y_z=self.Y_TRACK_A_Z,
            track_label_str=r"\text{Track A (with PE)}",
            with_pe=True,
            z_patterns_override=self.Z_TILDE_A_PATTERNS,
        )
        track_b2 = self._build_track(
            tokens=[self.TOKENS[i] for i in self.B_TOKEN_ORDER],
            x_indices=list(self.B_TOKEN_ORDER),
            z_indices=list(self.B_TOKEN_ORDER),
            y_token=self.Y_TRACK_B_TOKEN,
            y_x=self.Y_TRACK_B_X,
            y_box=self.Y_TRACK_B_BOX,
            y_z=self.Y_TRACK_B_Z,
            track_label_str=r"\text{Track B (with PE)}",
            with_pe=True,
            # Track B outputs at each slot DIFFER from Track A's same slot —
            # this is the whole point: order now matters.
            z_patterns_override=self.Z_TILDE_B_PATTERNS,
        )
        self._animate_track_in(track_a2)
        self._animate_track_in(track_b2)

        diff_label = (
            Tex(r"outputs now differ --- order matters")
            .scale(0.55)
            .move_to([0.0, self.Y_PUNCHLINE, 0.0])
        )
        self.play(FadeIn(diff_label), run_time=0.4)
        self.wait(0.8)

        final_caption = (
            Tex(r"position info encoded directly in the embedding")
            .scale(0.50)
            .to_edge(DOWN, buff=0.20)
        )
        self.play(FadeIn(final_caption), run_time=0.3)
        self.wait(1.0)

    # ----------------------------- track builder -----------------------------
    def _build_track(
        self,
        *,
        tokens: list[str],
        x_indices: list[int],
        z_indices: list[int],
        y_token: float,
        y_x: float,
        y_box: float,
        y_z: float,
        track_label_str: str,
        with_pe: bool,
        z_patterns_override: list[tuple[float, float, float]] | None = None,
    ) -> dict[str, Any]:
        """Build all mobjects for one track without animating them.

        Returns a dict carrying mobjects + helper lists for animation.
        """
        x_color = TEAL if with_pe else BLUE
        z_color = PURPLE

        # Token-strip texts.
        tok_mobs: list[Tex] = []
        for k, tok in enumerate(tokens):
            tx = self.X_TOK_BASE + k * self.X_TOK_SPACING
            t = (
                Tex(rf"\textit{{``{tok}''}}")
                .scale(0.58)
                .move_to([tx, y_token, 0.0])
            )
            tok_mobs.append(t)

        # Embedding columns x_t (with original index baked into the label and
        # the cell pattern so a permuted track shows the same cell signatures).
        # Labels go to the RIGHT of each column so they don't sit on the path
        # of the vertical arrow x_t -> self-attn box (lint pattern E2/check_arrow_path_clear).
        x_cols: list[TensorColumn] = []
        for k, orig_idx in enumerate(x_indices):
            cx = self.X_TOK_BASE + k * self.X_TOK_SPACING
            label = f"\\tilde{{x}}_{{{orig_idx + 1}}}" if with_pe else f"x_{orig_idx + 1}"
            col = self._build_tensor_with_pattern(
                self.X_PATTERNS[orig_idx], x_color
            )
            col.move_to([cx, y_x, 0.0])
            lbl = (
                MathTex(label)
                .scale(0.50)
                .next_to(col, RIGHT, buff=0.06)
            )
            x_cols.append((col, lbl))  # type: ignore[arg-type]

        # Self-attention box stretched across the 3 columns.
        box_w = (self.X_BOX_RIGHT - self.X_BOX_LEFT) + 0.40
        box_cx = (self.X_BOX_RIGHT + self.X_BOX_LEFT) / 2.0
        attn_box = LabeledBox(
            label=r"\mathrm{self\text{-}attn}",
            width=box_w,
            height=0.55,
            label_scale=0.50,
            color=GREY_B,
        ).move_to([box_cx, y_box, 0.0])

        # Output z columns.
        z_cols: list[TensorColumn] = []
        z_pats = z_patterns_override if z_patterns_override is not None else self.Z_PATTERNS
        for k, orig_idx in enumerate(z_indices):
            zx = self.X_Z_BASE + k * self.X_Z_SPACING
            zcol = self._build_tensor_with_pattern(
                z_pats[orig_idx], z_color
            )
            zcol.move_to([zx, y_z, 0.0])
            zlbl = (
                MathTex(rf"z_{orig_idx + 1}")
                .scale(0.50)
                .next_to(zcol, DOWN, buff=0.06)
            )
            z_cols.append((zcol, zlbl))  # type: ignore[arg-type]

        # Track label off to the right side.
        # Place it next to the rightmost x column to identify the track.
        track_lbl = (
            MathTex(track_label_str)
            .scale(0.45)
            .move_to([self.X_TOK_BASE + 2 * self.X_TOK_SPACING + 1.65, y_x, 0.0])
        )

        # Arrows from each x column to the attention box (vertical) and from
        # the attention box to each z column (vertical). The box spans all 3
        # x columns horizontally so each arrow stays clear of siblings.
        x_to_box_arrows: list[Arrow] = []
        for col, _lbl in x_cols:
            arr = _vertical_arrow(
                col, attn_box, buff=0.08, tip_length=0.13, color=WHITE,
                stroke_width=2.5,
            )
            x_to_box_arrows.append(arr)
        box_to_z_arrows: list[Arrow] = []
        for zcol, _zlbl in z_cols:
            arr = _vertical_arrow(
                attn_box, zcol, buff=0.08, tip_length=0.13, color=WHITE,
                stroke_width=2.5,
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
        """Sequence the appearance of one track: tokens → x → box → z."""
        tok_mobs = track["tok_mobs"]
        x_cols = track["x_cols"]
        attn_box = track["attn_box"]
        z_cols = track["z_cols"]
        track_lbl = track["track_lbl"]
        x_arrows = track["x_to_box_arrows"]
        z_arrows = track["box_to_z_arrows"]

        self.play(*[FadeIn(t) for t in tok_mobs], FadeIn(track_lbl), run_time=0.30)
        self.play(
            *[FadeIn(c) for c, _ in x_cols],
            *[FadeIn(lbl) for _, lbl in x_cols],
            run_time=0.30,
        )
        self.play(
            FadeIn(attn_box),
            *[Create(a) for a in x_arrows],
            run_time=0.40,
        )
        self.play(
            *[Create(a) for a in z_arrows],
            *[FadeIn(c) for c, _ in z_cols],
            *[FadeIn(lbl) for _, lbl in z_cols],
            run_time=0.45,
        )

    def _track_mobs(self, track: dict[str, Any]) -> list[VMobject]:
        """Flatten a track into a single mobject list (for cleanup)."""
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
        """Build one "x_t + PE_t = ~x_t" panel centered at x = cx.

        Layout within the panel: three TensorColumns side-by-side with a
        "+" between (col 0, 1) and an "=" between (col 1, 2). Each column
        has a label below it. All three sit at y = Y_PE_ROW.
        """
        # Internal panel offsets — the three columns plus the two operator
        # symbols. Total horizontal span ≈ 2.5 units; we want centers at
        # cx - 1.0, cx, cx + 1.0 with operators in between.
        col_dx = 1.05    # distance between adjacent columns
        x_col = self._build_tensor_with_pattern(
            self.X_PATTERNS[t], BLUE, cell_size=self.PE_CELL
        ).move_to([cx - col_dx, self.Y_PE_ROW, 0.0])
        pe_col = self._build_tensor_with_pattern(
            self.PE_PATTERNS[t], GREY_B, cell_size=self.PE_CELL
        ).move_to([cx, self.Y_PE_ROW, 0.0])
        # ~x is per-cell: x_pattern[k] + PE_pattern[k] (clamped to 0..1).
        xt_pattern = tuple(
            min(1.0, self.X_PATTERNS[t][k] + self.PE_PATTERNS[t][k])
            for k in range(self.TENSOR_DIM)
        )
        xt_col = self._build_tensor_with_pattern(
            xt_pattern, TEAL, cell_size=self.PE_CELL
        ).move_to([cx + col_dx, self.Y_PE_ROW, 0.0])

        plus = (
            MathTex(r"+")
            .scale(0.65)
            .move_to([cx - col_dx / 2.0, self.Y_PE_ROW, 0.0])
        )
        eq = (
            MathTex(r"=")
            .scale(0.65)
            .move_to([cx + col_dx / 2.0, self.Y_PE_ROW, 0.0])
        )

        x_lbl = (
            MathTex(rf"x_{t + 1}")
            .scale(0.55)
            .next_to(x_col, DOWN, buff=0.10)
        )
        pe_lbl = (
            MathTex(rf"PE_{t + 1}")
            .scale(0.55)
            .next_to(pe_col, DOWN, buff=0.10)
        )
        xt_lbl = (
            MathTex(rf"\tilde{{x}}_{t + 1}")
            .scale(0.55)
            .next_to(xt_col, DOWN, buff=0.10)
        )

        all_mobs = [x_col, pe_col, xt_col, plus, eq, x_lbl, pe_lbl, xt_lbl]
        return {
            "x_col": x_col,
            "pe_col": pe_col,
            "xt_col": xt_col,
            "plus": plus,
            "eq": eq,
            "x_lbl": x_lbl,
            "pe_lbl": pe_lbl,
            "xt_lbl": xt_lbl,
            "all_mobs": all_mobs,
        }
