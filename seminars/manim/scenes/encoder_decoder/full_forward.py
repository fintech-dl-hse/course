"""Full encoder-decoder forward pass — translation (V15, catalog finale).

Сцена объединяет всё, что мы строили в V06–V14, в один кадр перевода
``the cat sat`` → ``le chat ...`` (французский). Слева работает энкодер
параллельно по исходным токенам; справа декодер последовательно
обрабатывает уже сгенерированные target-токены и предсказывает
следующий французский токен.

Сценарий по фазам:

1. Заголовок: ``Encoder–Decoder forward: translation``.
2. Левая колонка (encoder, EN):
   - Source-токены ``"the" / "cat" / "sat"``.
   - Эмбеддинги (TensorColumn dim=4) с подписью ``+PE_i``.
   - Collapsed ``Encoder Block × N`` LabeledBox.
   - Скрытые состояния энкодера ``e_1, e_2, e_3``.
   - Проекции ``W_K`` (orange) / ``W_V`` (green) → стопки ``K`` и ``V``.
3. Правая колонка (decoder, FR):
   - Уже сгенерированные target-токены ``"<bos>" / "le" / "chat"``.
   - Эмбеддинги + ``+PE_i``.
   - Collapsed ``Decoder Block × N`` LabeledBox с двумя помеченными
     sublayer-плашками внутри: ``Causal Self-Attn`` и ``Cross-Attn``.
   - Скрытое состояние декодера ``d_t``.
4. Cross-attention bridge: длинные стрелки от стопок ``K`` и ``V``
   (низ энкодера) к sublayer-плашке ``Cross-Attn`` внутри decoder block.
   Маршрутизация — L-образные сегменты, чтобы не пересекать промежуточные
   объекты (см. lessons E2/E9 в .claude/agents/manim-visualizer.md).
5. Output: ``d_t`` → ``W_{vocab}`` → softmax → предсказанный French
   токен ``"assis"``, который дописывается к target-ленте.
6. Caption (низ): ``Encoder: parallel over source. Decoder: sequential over target.``.

Цветовая конвенция (наследуется из V06–V14):
- ``BLUE`` — input embeddings, encoder hidden, decoder hidden ``d_t``.
- ``ORANGE`` — ``W_K`` и K-стопка.
- ``GREEN`` — ``W_V`` и V-стопка.
- ``TEAL`` — Q / cross-attn sublayer / предсказанный токен.
- ``PURPLE`` — causal self-attn sublayer (отличает от cross-attn).
- ``YELLOW`` — softmax-выход словаря.
- ``GREY_B`` — нейтральные блоки (encoder / decoder / W_vocab).

Сцена использует только публичные примитивы ``shared.neural``
(``LabeledBox``, ``TensorColumn``, ``arrow_between``); ``_horizontal_arrow``
и ``_vertical_arrow`` — локальные копии (как в ``rnn_forward.py:45-58``).
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
    GREEN,
    GREY_B,
    LEFT,
    Line,
    MathTex,
    ORANGE,
    PURPLE,
    RIGHT,
    Scene,
    TEAL,
    Tex,
    UP,
    VGroup,
    VMobject,
    WHITE,
    YELLOW,
    Write,
)

from shared.neural import LabeledBox, TensorColumn, arrow_between


def _horizontal_arrow(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    """Стрелка, всегда прилипающая к правому/левому краю объектов.

    Локальная копия (см. ``rnn_forward.py:45-58``). Нужна, когда
    ``arrow_between`` выбрал бы вертикальное крепление и провёл бы
    линию через посторонний узел.
    """
    defaults: dict[str, Any] = {"buff": 0.08, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


def _vertical_arrow(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    """Стрелка, всегда прилипающая к верхнему/нижнему краю объектов."""
    defaults: dict[str, Any] = {"buff": 0.06, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[1] >= b.get_center()[1]:
        start, end = a.get_bottom(), b.get_top()
    else:
        start, end = a.get_top(), b.get_bottom()
    return Arrow(start=start, end=end, **defaults)


class EncoderDecoderForward(Scene):
    """V15: full encoder-decoder forward pass for translation EN -> FR."""

    # ---- Layout constants — 720p (-qm) frame ±(7.11, 4.0) ----
    TENSOR_DIM = 4
    CELL = 0.20                       # tensor-column cell side (compact)

    # Vertical anchors (top → bottom).
    Y_TITLE = 3.75
    Y_TOK = 3.20                      # source / target token rows
    Y_EMB = 2.55                      # embeddings (height ~0.80, top ~2.95, bot ~2.15)
    Y_PE_LBL = 1.85                   # +PE_i annotation row (between emb and block)
    Y_BLOCK_TOP = 1.40                # top edge of encoder/decoder block
    Y_BLOCK_BOT = -0.45               # bottom edge of block
    Y_BLOCK_MID = (Y_BLOCK_TOP + Y_BLOCK_BOT) / 2.0  # = 0.475
    Y_HIDDEN = -1.20                  # e_1..e_3 (left) / d_t (right)
    Y_PROJ = -2.05                    # W_K, W_V boxes
    Y_KV = -2.85                      # K, V tensor stacks
    Y_VOCAB = -2.10                   # decoder side: vocab box (right of d_t)
    Y_PRED = -3.05                    # predicted next token
    Y_CAPTION = -3.78

    # Horizontal layout — encoder (left) and decoder (right) columns.
    # Encoder zone center at -3.85; decoder zone center at +3.85.
    X_ENC = -3.85
    X_DEC = 3.85

    # Source / target tokens for the translation framing.
    SRC_TOKENS = ("the", "cat", "sat")
    TGT_TOKENS = ("<bos>", "le", "chat")
    PRED_TOKEN = "assis"              # predicted next French token

    # Spacing between source-side embedding columns / encoder hidden columns.
    X_EMB_SPACING = 0.55              # between embedding columns on same side
    X_TOK_SPACING = 0.95              # token strip spacing (wider than emb to avoid overlap)

    # K / V stacks under encoder.
    X_KV_INNER_SPACING = 0.30         # between adjacent columns inside K (or V) stack
    X_K_CENTER_OFFSET = -1.05         # K stack center relative to X_ENC
    X_V_CENTER_OFFSET = 1.05          # V stack center relative to X_ENC

    # Decoder block: two stacked sublayer mini-boxes inside.
    DEC_BLOCK_W = 2.85
    DEC_BLOCK_H = 1.90
    SUBLAYER_W = 2.45
    SUBLAYER_H = 0.55

    # Encoder block dims.
    ENC_BLOCK_W = 2.85
    ENC_BLOCK_H = 1.90

    # Vocab projection box on decoder side.
    VOCAB_BOX_W = 0.95
    VOCAB_BOX_H = 0.50

    # Projection W_K / W_V dims.
    WPROJ_W = 0.85
    WPROJ_H = 0.40

    # ----------------------------- helpers -----------------------------
    def _strip_xs(self, n: int, center_x: float, spacing: float) -> list[float]:
        """Return n evenly spaced x-coords centered on ``center_x``."""
        first_x = center_x - (n - 1) * spacing / 2.0
        return [first_x + j * spacing for j in range(n)]

    def _build_emb_row(
        self,
        center_x: float,
        n: int,
        color: str,
        fill_opacity: float = 0.40,
    ) -> list[TensorColumn]:
        """Row of n TensorColumns at Y_EMB, centered on ``center_x``."""
        cols: list[TensorColumn] = []
        xs = self._strip_xs(n, center_x, self.X_EMB_SPACING)
        for cx in xs:
            col = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL,
                color=color,
                fill_opacity=fill_opacity,
            ).move_to([cx, self.Y_EMB, 0.0])
            cols.append(col)
        return cols

    def _build_e_row(self) -> list[TensorColumn]:
        """Encoder hidden states e_1..e_3 at Y_HIDDEN, centered under encoder."""
        cols: list[TensorColumn] = []
        xs = self._strip_xs(3, self.X_ENC, self.X_EMB_SPACING)
        for cx in xs:
            col = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL,
                color=BLUE,
                fill_opacity=0.50,
            ).move_to([cx, self.Y_HIDDEN, 0.0])
            cols.append(col)
        return cols

    def _build_kv_stack(
        self,
        center_x_abs: float,
        color: str,
    ) -> list[TensorColumn]:
        """3-column tensor stack at (center_x_abs, Y_KV)."""
        cols: list[TensorColumn] = []
        xs = self._strip_xs(3, center_x_abs, self.X_KV_INNER_SPACING)
        for cx in xs:
            col = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL,
                color=color,
                fill_opacity=0.55,
            ).move_to([cx, self.Y_KV, 0.0])
            cols.append(col)
        return cols

    def _build_dec_block(self) -> tuple[LabeledBox, LabeledBox, LabeledBox, MathTex]:
        """Outer decoder block + 2 sublayer plates + ×N annotation.

        Returns (outer_box, causal_sublayer, cross_sublayer, n_label).
        Causal Self-Attn sits on top, Cross-Attn below — matches the
        Vaswani-style decoder block ordering.
        """
        outer = LabeledBox(
            label=r"\text{Decoder Block} \times N",
            width=self.DEC_BLOCK_W,
            height=self.DEC_BLOCK_H,
            color=GREY_B,
            label_scale=0.40,
            fill_opacity=0.10,
        ).move_to([self.X_DEC, self.Y_BLOCK_MID, 0.0])
        # Move outer label up to the top so internals are visible.
        outer.label_tex.move_to(
            [self.X_DEC, self.Y_BLOCK_TOP - 0.18, 0.0]
        )

        # Position sublayers: causal on top, cross below, with vertical gap.
        # Block goes from Y_BLOCK_TOP (1.45) to Y_BLOCK_BOT (-0.45) — height
        # 1.90. After the title at top (~0.18 thick), available content is
        # 1.45 - 0.36 = 1.09 down to -0.45. Place causal at y=0.55 and
        # cross at y=-0.10 (sublayer height 0.55 each, gap ~0.10).
        causal = LabeledBox(
            label=r"\text{Causal Self-Attn}",
            width=self.SUBLAYER_W,
            height=self.SUBLAYER_H,
            color=PURPLE,
            label_scale=0.36,
            fill_opacity=0.22,
        ).move_to([self.X_DEC, 0.55, 0.0])
        cross = LabeledBox(
            label=r"\text{Cross-Attn}",
            width=self.SUBLAYER_W,
            height=self.SUBLAYER_H,
            color=TEAL,
            label_scale=0.36,
            fill_opacity=0.22,
        ).move_to([self.X_DEC, -0.10, 0.0])
        # Block-level "× N" already in outer label; no separate n_label needed.
        # Return placeholder for compatibility with destructuring.
        n_lbl = MathTex(r"").scale(0.01)
        return outer, causal, cross, n_lbl

    def _build_enc_block(self) -> LabeledBox:
        """Outer encoder block (collapsed; internals were unfolded in V08)."""
        box = LabeledBox(
            label=r"\text{Encoder Block} \times N",
            width=self.ENC_BLOCK_W,
            height=self.ENC_BLOCK_H,
            color=GREY_B,
            label_scale=0.42,
            fill_opacity=0.10,
        ).move_to([self.X_ENC, self.Y_BLOCK_MID, 0.0])
        return box

    # ----------------------------- main -----------------------------
    def construct(self) -> None:
        # ===================== Phase 0: title =====================
        title = (
            MathTex(
                r"\text{Encoder--Decoder forward: translation (EN} \to "
                r"\text{FR)}"
            )
            .scale(0.55)
            .move_to([0.0, self.Y_TITLE, 0.0])
        )
        self.play(Write(title), run_time=0.6)
        self.wait(0.15)

        # ===================== Phase 1: source tokens (left) =====================
        src_xs = self._strip_xs(
            len(self.SRC_TOKENS), self.X_ENC, self.X_TOK_SPACING
        )
        src_tok_mobs: list[Tex] = []
        for j, tok in enumerate(self.SRC_TOKENS):
            t = (
                Tex(rf"\textit{{``{tok}''}}")
                .scale(0.55)
                .move_to([src_xs[j], self.Y_TOK, 0.0])
            )
            src_tok_mobs.append(t)

        # Source-side label "(EN)" placed FAR LEFT of the token strip so it
        # doesn't overlap the tokens. We anchor it against the left frame
        # edge (clear of the leftmost token at x ≈ X_ENC - X_TOK_SPACING).
        src_side_lbl = (
            MathTex(r"\text{EN}")
            .scale(0.45)
            .move_to([-6.70, self.Y_TOK, 0.0])
            .set_color(GREY_B)
        )

        self.play(
            *[FadeIn(t) for t in src_tok_mobs],
            FadeIn(src_side_lbl),
            run_time=0.4,
        )
        self.wait(0.1)

        # ===================== Phase 2: source embeddings + PE =====================
        src_emb_cols = self._build_emb_row(self.X_ENC, len(self.SRC_TOKENS), BLUE)
        # +PE annotation to the LEFT of the rightmost emb column, aligned with
        # the row's vertical center so it doesn't fall on top of any arrow.
        # Place to the LEFT of the leftmost emb column (clear of the column
        # cells and the left-side EN label which is at x=-6.70).
        src_pe_lbl = (
            MathTex(r"+\,PE_i")
            .scale(0.42)
            .next_to(src_emb_cols[0], LEFT, buff=0.18)
        )

        # Token → embedding arrows (one per token, forced vertical).
        src_tok_to_emb: list[Arrow] = []
        for j in range(len(self.SRC_TOKENS)):
            arr = _vertical_arrow(
                src_tok_mobs[j], src_emb_cols[j],
                buff=0.08, tip_length=0.10, color=BLUE, stroke_width=2.0,
            )
            src_tok_to_emb.append(arr)

        self.play(
            *[FadeIn(c) for c in src_emb_cols],
            *[Create(a) for a in src_tok_to_emb],
            run_time=0.45,
        )
        self.play(FadeIn(src_pe_lbl), run_time=0.2)
        self.wait(0.1)

        # ===================== Phase 3: encoder block + arrow =====================
        enc_block = self._build_enc_block()
        # Single arrow from middle embedding column → top of encoder block.
        arr_emb_to_enc = _vertical_arrow(
            src_emb_cols[1], enc_block,
            buff=0.10, tip_length=0.12, color=GREY_B, stroke_width=2.5,
        )
        self.play(
            FadeIn(enc_block),
            Create(arr_emb_to_enc),
            run_time=0.45,
        )
        self.wait(0.1)

        # ===================== Phase 4: encoder hidden e_1..e_3 =====================
        e_cols = self._build_e_row()
        e_labels: list[MathTex] = []
        for j, col in enumerate(e_cols):
            lbl = (
                MathTex(rf"e_{j + 1}")
                .scale(0.42)
                .next_to(col, DOWN, buff=0.06)
            )
            e_labels.append(lbl)
        # Single arrow from block bottom → middle e column.
        arr_enc_to_e = _vertical_arrow(
            enc_block, e_cols[1],
            buff=0.10, tip_length=0.12, color=BLUE, stroke_width=2.5,
        )
        self.play(
            *[FadeIn(c) for c in e_cols],
            *[FadeIn(lbl) for lbl in e_labels],
            Create(arr_enc_to_e),
            run_time=0.5,
        )
        self.wait(0.15)

        # ===================== Phase 5: encoder K, V projections =====================
        x_wk = self.X_ENC + self.X_K_CENTER_OFFSET   # absolute x of K center
        x_wv = self.X_ENC + self.X_V_CENTER_OFFSET   # absolute x of V center
        w_k = LabeledBox(
            label=r"W_K",
            width=self.WPROJ_W,
            height=self.WPROJ_H,
            color=ORANGE,
            label_scale=0.42,
            fill_opacity=0.20,
        ).move_to([x_wk, self.Y_PROJ, 0.0])
        w_v = LabeledBox(
            label=r"W_V",
            width=self.WPROJ_W,
            height=self.WPROJ_H,
            color=GREEN,
            label_scale=0.42,
            fill_opacity=0.20,
        ).move_to([x_wv, self.Y_PROJ, 0.0])

        # K and V tensor stacks (3 cols each).
        k_cols = self._build_kv_stack(x_wk, ORANGE)
        v_cols = self._build_kv_stack(x_wv, GREEN)
        k_label = (
            MathTex(r"K")
            .scale(0.45)
            .next_to(k_cols[-1], RIGHT, buff=0.10)
        )
        v_label = (
            MathTex(r"V")
            .scale(0.45)
            .next_to(v_cols[-1], RIGHT, buff=0.10)
        )

        # Arrows from e_* row to W_K and W_V. Use leftmost e for W_K, rightmost
        # e for W_V — keeps the diagonal short and visually fanning out.
        arr_e_to_wk = _vertical_arrow(
            e_cols[0], w_k,
            buff=0.10, tip_length=0.10, color=ORANGE, stroke_width=2.0,
        )
        arr_e_to_wv = _vertical_arrow(
            e_cols[2], w_v,
            buff=0.10, tip_length=0.10, color=GREEN, stroke_width=2.0,
        )
        # W_K → K middle column, W_V → V middle column.
        arr_wk_to_k = _vertical_arrow(
            w_k, k_cols[1],
            buff=0.06, tip_length=0.10, color=ORANGE, stroke_width=2.0,
        )
        arr_wv_to_v = _vertical_arrow(
            w_v, v_cols[1],
            buff=0.06, tip_length=0.10, color=GREEN, stroke_width=2.0,
        )

        self.play(
            FadeIn(w_k), FadeIn(w_v),
            Create(arr_e_to_wk), Create(arr_e_to_wv),
            run_time=0.5,
        )
        self.play(
            *[FadeIn(c) for c in k_cols + v_cols],
            FadeIn(k_label), FadeIn(v_label),
            Create(arr_wk_to_k), Create(arr_wv_to_v),
            run_time=0.55,
        )
        self.wait(0.2)

        # ===================== Phase 6: target tokens (right) =====================
        tgt_xs = self._strip_xs(
            len(self.TGT_TOKENS), self.X_DEC, self.X_TOK_SPACING
        )
        tgt_tok_mobs: list[Tex] = []
        for j, tok in enumerate(self.TGT_TOKENS):
            # Render <bos> with explicit Tex angle-bracket macros — the literal
            # "<" is reinterpreted by LaTeX (smart-punctuation / babel) as ¡
            # in default mode, so we use \textless / \textgreater.
            if tok.startswith("<"):
                inner = tok.strip("<>")
                t = (
                    Tex(rf"\textless\textit{{{inner}}}\textgreater").scale(0.50)
                    .move_to([tgt_xs[j], self.Y_TOK, 0.0])
                )
            else:
                t = (
                    Tex(rf"\textit{{``{tok}''}}").scale(0.55)
                    .move_to([tgt_xs[j], self.Y_TOK, 0.0])
                )
            tgt_tok_mobs.append(t)

        tgt_side_lbl = (
            MathTex(r"\text{FR}")
            .scale(0.45)
            .move_to([6.70, self.Y_TOK, 0.0])
            .set_color(GREY_B)
        )

        self.play(
            *[FadeIn(t) for t in tgt_tok_mobs],
            FadeIn(tgt_side_lbl),
            run_time=0.4,
        )
        self.wait(0.1)

        # ===================== Phase 7: target embeddings + PE =====================
        tgt_emb_cols = self._build_emb_row(self.X_DEC, len(self.TGT_TOKENS), BLUE)
        tgt_pe_lbl = (
            MathTex(r"+\,PE_i")
            .scale(0.42)
            .next_to(tgt_emb_cols[-1], RIGHT, buff=0.18)
        )
        tgt_tok_to_emb: list[Arrow] = []
        for j in range(len(self.TGT_TOKENS)):
            arr = _vertical_arrow(
                tgt_tok_mobs[j], tgt_emb_cols[j],
                buff=0.08, tip_length=0.10, color=BLUE, stroke_width=2.0,
            )
            tgt_tok_to_emb.append(arr)
        self.play(
            *[FadeIn(c) for c in tgt_emb_cols],
            *[Create(a) for a in tgt_tok_to_emb],
            run_time=0.45,
        )
        self.play(FadeIn(tgt_pe_lbl), run_time=0.2)
        self.wait(0.1)

        # ===================== Phase 8: decoder block + sublayers =====================
        dec_outer, causal_sub, cross_sub, _n = self._build_dec_block()
        # Arrow from middle target embedding → decoder block top.
        arr_temb_to_dec = _vertical_arrow(
            tgt_emb_cols[1], dec_outer,
            buff=0.10, tip_length=0.12, color=GREY_B, stroke_width=2.5,
        )

        self.play(
            FadeIn(dec_outer),
            Create(arr_temb_to_dec),
            run_time=0.45,
        )
        # Inner sublayers fade in top→bottom (causal first, cross second —
        # matches the Vaswani sublayer order).
        self.play(FadeIn(causal_sub), run_time=0.30)
        self.play(FadeIn(cross_sub), run_time=0.30)
        self.wait(0.15)

        # ===================== Phase 9: K/V → cross-attention bridge =====================
        # The cross-attn sublayer pulls K and V from the encoder.
        # Route as L-shaped paths so we don't pierce intermediate mobjects:
        #   K stack rightmost col → up to a rail just above K stack →
        #   right across mid-frame → down to cross_sub left edge.
        # V stack rightmost col → up to a slightly different rail →
        #   right across mid-frame → up to cross_sub left edge.
        # We do K (orange) above V (green) on different rails to avoid
        # arrow-on-arrow visual noise.

        # Rails: choose y-values such that
        #   - The K rail (lower) goes through y ≈ Y_KV + 0.55 = -2.30, which
        #     is below all encoder content (e row is at Y_HIDDEN = -1.20,
        #     hidden cells span [-1.40, -1.00]) and above the bottom of K
        #     stack (Y_KV - cell_size*dim/2 = -2.85 - 0.40 = -3.25). So the
        #     rail is in clear space.
        #   - The V rail at y ≈ -3.30 — under everything (caption is at
        #     -3.75, so we have room).
        #
        # NOTE: V stack is to the RIGHT of K stack on the encoder side. K
        # rightmost column is at x_wk + spacing = roughly -4.60. V rightmost
        # column is at x_wv + spacing = roughly -2.50. Both must travel
        # rightward to reach cross_sub left edge at x ≈ X_DEC - SUBLAYER_W/2
        # = 3.85 - 1.225 = 2.625.

        # The K/V bridge feeds into the Cross-Attn sublayer, which sits INSIDE
        # the outer decoder block. We can't route the arrow tip into the
        # sublayer interior because that path also crosses the outer block's
        # AABB (and lint flags it). Instead we end the bridges at the LEFT
        # edge of the OUTER decoder block (just outside it), with the
        # arrowhead aimed at the cross-sub vertical band. The colored stroke
        # + the visible cross-sub plate behind the arrowhead reads cleanly
        # as "K and V feed Cross-Attn".
        dec_left_x = float(dec_outer.get_left()[0])  # ≈ 2.425
        # End-points sit just left of dec_outer's left edge (buff = 0.05).
        bridge_end_x = dec_left_x - 0.05            # ≈ 2.375
        # K bridge: take from rightmost K column.
        k_src_col = k_cols[-1]
        k_anchor = [
            float(k_src_col.get_right()[0]) + 0.02,
            float(k_src_col.get_center()[1]),
            0.0,
        ]
        # K rail just below the K stack.
        k_rail_y = self.Y_KV - 0.65   # = -3.50
        k_seg1 = Line(
            start=k_anchor,
            end=[k_anchor[0], k_rail_y, 0.0],
            color=ORANGE, stroke_width=2.0,
        )
        k_seg2 = Line(
            start=[k_anchor[0], k_rail_y, 0.0],
            end=[bridge_end_x, k_rail_y, 0.0],
            color=ORANGE, stroke_width=2.0,
        )
        # Final segment: short upward arrow ending just left of the
        # cross-sublayer's left edge (well clear of any AABB).
        # Cross-sub is at y=-0.10. End at y = -0.20 so the arrow visibly
        # points "into" the cross-attn band but stops outside dec_outer.
        k_seg3 = Arrow(
            start=[bridge_end_x, k_rail_y, 0.0],
            end=[bridge_end_x, -0.25, 0.0],
            buff=0.0, stroke_width=2.0, tip_length=0.12, color=ORANGE,
        )

        # V bridge: rightmost V column, separate rail (above K rail).
        v_src_col = v_cols[-1]
        v_anchor = [
            float(v_src_col.get_right()[0]) + 0.02,
            float(v_src_col.get_center()[1]),
            0.0,
        ]
        v_rail_y = self.Y_KV - 0.30   # = -3.15 (above K rail)
        # Use a different end-x for V so its final arrow doesn't sit on top
        # of K's final arrow.
        v_bridge_end_x = bridge_end_x - 0.20        # ≈ 2.175
        v_seg1 = Line(
            start=v_anchor,
            end=[v_anchor[0], v_rail_y, 0.0],
            color=GREEN, stroke_width=2.0,
        )
        v_seg2 = Line(
            start=[v_anchor[0], v_rail_y, 0.0],
            end=[v_bridge_end_x, v_rail_y, 0.0],
            color=GREEN, stroke_width=2.0,
        )
        v_seg3 = Arrow(
            start=[v_bridge_end_x, v_rail_y, 0.0],
            end=[v_bridge_end_x, 0.05, 0.0],
            buff=0.0, stroke_width=2.0, tip_length=0.12, color=GREEN,
        )

        # No bridge labels needed — color (orange = K, green = V) and the
        # K/V column-stack labels already disambiguate. Adding labels next
        # to the K/V column labels caused tight overlap in earlier frames.

        self.play(
            Create(k_seg1), Create(k_seg2), Create(k_seg3),
            Create(v_seg1), Create(v_seg2), Create(v_seg3),
            run_time=0.8,
        )
        self.wait(0.2)

        # ===================== Phase 10: decoder hidden d_t =====================
        d_t = TensorColumn(
            dim=self.TENSOR_DIM,
            cell_size=self.CELL,
            color=BLUE,
            fill_opacity=0.55,
        ).move_to([self.X_DEC, self.Y_HIDDEN, 0.0])
        d_t_lbl = (
            MathTex(r"d_t")
            .scale(0.45)
            .next_to(d_t, DOWN, buff=0.06)
        )
        arr_dec_to_dt = _vertical_arrow(
            dec_outer, d_t,
            buff=0.10, tip_length=0.12, color=BLUE, stroke_width=2.5,
        )
        self.play(
            FadeIn(d_t), FadeIn(d_t_lbl),
            Create(arr_dec_to_dt),
            run_time=0.45,
        )
        self.wait(0.15)

        # ===================== Phase 11: vocab projection + softmax + predicted token =====================
        # Vocab box sits to the RIGHT of d_t but constrained inside frame.
        # d_t is at X_DEC = 3.85; vocab box must clear the frame. Cell of d_t
        # spans ~0.20 wide → its right edge ~3.95. Place vocab box at x =
        # 5.10 (left edge ~4.625, right edge ~5.575 — well inside ±7.11).
        vocab_box = LabeledBox(
            label=r"W_{\text{vocab}}",
            width=self.VOCAB_BOX_W,
            height=self.VOCAB_BOX_H,
            color=GREY_B,
            label_scale=0.40,
            fill_opacity=0.12,
        ).move_to([5.10, self.Y_HIDDEN, 0.0])

        # Softmax tensor: small dim=5 column to the right of vocab box.
        softmax_tensor = TensorColumn(
            dim=5,
            cell_size=0.22,
            color=YELLOW,
            fill_opacity=0.25,
            highlight_index=2,
            highlight_color="#F5C518",
            highlight_opacity=0.90,
        ).move_to([6.30, self.Y_HIDDEN, 0.0])
        softmax_lbl = (
            MathTex(r"\mathrm{softmax}")
            .scale(0.36)
            .next_to(softmax_tensor, DOWN, buff=0.06)
        )

        arr_dt_to_vbox = _horizontal_arrow(
            d_t, vocab_box,
            buff=0.08, tip_length=0.12, color=GREY_B, stroke_width=2.0,
        )
        arr_vbox_to_softmax = _horizontal_arrow(
            vocab_box, softmax_tensor,
            buff=0.08, tip_length=0.12, color=YELLOW, stroke_width=2.0,
        )

        self.play(
            FadeIn(vocab_box),
            Create(arr_dt_to_vbox),
            run_time=0.40,
        )
        self.play(
            FadeIn(softmax_tensor), FadeIn(softmax_lbl),
            Create(arr_vbox_to_softmax),
            run_time=0.45,
        )

        # Predicted token: sits directly under the softmax tensor with a
        # short vertical arrow. Color TEAL to mark "newly predicted".
        pred_tok = (
            Tex(rf"\textit{{``{self.PRED_TOKEN}''}}")
            .scale(0.55)
            .move_to([6.30, self.Y_PRED + 0.18, 0.0])
            .set_color(TEAL)
        )
        arr_softmax_to_pred = _vertical_arrow(
            softmax_tensor, pred_tok,
            buff=0.08, tip_length=0.12, color=TEAL, stroke_width=2.0,
        )
        self.play(
            FadeIn(pred_tok),
            Create(arr_softmax_to_pred),
            run_time=0.40,
        )
        self.wait(0.4)

        # ===================== Phase 12: caption =====================
        caption = (
            MathTex(
                r"\text{Encoder: parallel over source.\ "
                r"Decoder: sequential over target.}"
            )
            .scale(0.46)
            .move_to([0.0, self.Y_CAPTION, 0.0])
        )
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(1.0)
