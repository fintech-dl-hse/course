"""Encoder-decoder cross-attention scene (V14).

Сцена показывает «мост» между энкодером и декодером в seq2seq трансформере:
декодер строит свои queries из текущего скрытого состояния ``d_t``, а keys
и values берутся из выходов энкодера ``e_1, e_2, e_3``. После softmax по
attention-скорам декодер получает контекстный вектор ``c_t``, который
вливается в его cross-attention sublayer.

Сценарий по фазам:

1. Заголовок: ``Cross-attention: decoder Q × encoder K, V``.
2. Левая зона (encoder): исходные токены ``"the" / "cat" / "sat"`` →
   collapsed ``Encoder Stack`` LabeledBox → ряд из трёх скрытых тензоров
   ``e_1, e_2, e_3`` (TensorColumn dim=4).
3. Из ``e_*`` две проекции через цветные ``W_K`` (orange) и ``W_V`` (green) —
   получаются столбцы ``K`` (3 колонки) и ``V`` (3 колонки), идентичной
   формы по числу позиций исходного предложения.
4. Правая зона (decoder): уже сгенерированные target-токены ``"le" / "chat"``,
   collapsed ``Decoder Block`` LabeledBox, текущее скрытое состояние ``d_t``.
5. ``d_t`` проецируется через ``W_Q`` (blue) → одна Q-колонка.
6. Центр кадра — собственно cross-attention:
   - ``Q · K^\\top`` → строка из 3 raw-скоров;
   - ``softmax`` → строка из 3 весов с подсветкой максимума (позиция 1 = «cat»);
   - взвешенная сумма по ``V`` → контекстный вектор ``c_t`` (TensorColumn dim=4).
7. ``c_t`` вливается обратно в decoder block — стрелка из центрального
   ``c_t`` поднимается в decoder-block LabeledBox.
8. Финальный caption: ``Decoder uses encoder context via Q from decoder, K/V from encoder``.

Цветовая конвенция (наследуется из V06–V08):
- ``TEAL`` (≈ cyan) — Q-сторона декодера, ``W_Q``, Q-тензор.
- ``ORANGE`` — K-сторона энкодера, ``W_K``, K-тензоры.
- ``GREEN`` — V-сторона энкодера, ``W_V``, V-тензоры.
- ``BLUE`` — abstract embeddings / encoder hidden ``e_*`` / decoder hidden ``d_t``.
- ``PURPLE`` — выходной контекстный вектор ``c_t``.
- ``GREY_B`` — нейтральные блоки (encoder stack, decoder block).

Сцена использует только публичные примитивы ``shared.neural``
(``LabeledBox``, ``TensorColumn``, ``arrow_between``); ``_horizontal_arrow`` —
локальная копия (как в ``rnn_forward.py:45-58``), чтобы вертикально-близкие
объекты не получали top/bottom-крепление с пересечением соседних узлов.
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
    Rectangle,
    RIGHT,
    Scene,
    Square,
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

    Локальная копия (см. ``rnn_forward.py:45-58``). Нужна, когда центры объектов
    близки по x, и ``arrow_between`` выбрал бы вертикальное крепление, проводя
    линию через посторонние узлы.
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


class CrossAttention(Scene):
    """V14: encoder–decoder cross-attention. Q from decoder, K/V from encoder."""

    # ---- Layout constants — 720p (-qm) frame ±(7.11, 4.0) ----
    TENSOR_DIM = 4
    CELL = 0.22                       # tensor-column cell side (compact — many columns)

    # Vertical anchors (top → bottom).
    Y_TITLE = 3.65
    Y_SRC_TOK = 3.05                  # source tokens row (encoder side)
    Y_TGT_TOK = 3.05                  # target tokens row (decoder side)
    Y_BLOCK = 2.05                    # Encoder stack / Decoder block boxes
    Y_HIDDEN = 0.85                   # e_1..e_3 / d_t tensors row
    Y_PROJ = -0.30                    # W_K, W_V, W_Q boxes
    Y_QKV = -1.50                     # K, V columns (encoder) and Q column (decoder)
    Y_SCORES = -2.45                  # 1×3 raw-score row (will become softmax weights)
    Y_CTX = -3.45                     # context vector c_t (sits at the bottom)
    Y_CAPTION = -3.85                 # final caption

    # Horizontal anchors.
    # Encoder zone: leftmost ~third of frame.
    X_ENC_BLOCK = -5.40               # Encoder stack center
    X_E_BASE = -6.40                  # leftmost e_j column
    X_E_SPACING = 0.60                # between e_1, e_2, e_3
    X_WK = -5.85
    X_WV = -4.10
    X_K_BASE = -6.20                  # leftmost K column (3 cols, ORANGE)
    X_KV_SPACING = 0.46
    X_V_BASE = -4.45                  # leftmost V column (3 cols, GREEN)

    # Decoder zone: rightmost ~third of frame.
    X_DEC_BLOCK = 5.40                # Decoder block center
    X_DT = 5.40                       # d_t column center
    X_WQ = 5.40
    X_Q = 5.40                        # Q column center

    # Center: attention computation.
    X_SCORE_BASE = -0.70              # leftmost score cell
    X_SCORE_SPACING = 0.50            # cell side + gap
    SCORE_CELL = 0.42                 # softmax/score cell size
    X_CTX = 0.0                       # c_t (context vector) center

    # Source/target token strings for the translation framing.
    SRC_TOKENS = ("the", "cat", "sat")
    TGT_TOKENS = ("le", "chat")
    # The decoder is currently predicting *the next* target token. Position 1
    # ("cat") is the strongest source for the attention pattern below — it
    # explains why the decoder, when generating the next French token after
    # "le chat", focuses on the English subject "cat".
    ATTN_DOMINANT = 1

    # Toy attention scores Q · K^T (pre-softmax) and post-softmax weights.
    # Position 1 ("cat") is dominant — pedagogical example.
    RAW_SCORES = (0.6, 2.4, 0.7)
    ATTN_WEIGHTS = (0.10, 0.78, 0.12)

    # ----------------------------- helpers -----------------------------
    def _src_token_xs(self) -> list[float]:
        """Token strip x-coordinates for the source sentence (above e_*)."""
        # Strip is wider than the e-column spacing so quoted words don't overlap.
        spacing = 0.85
        base = self.X_E_BASE + (len(self.SRC_TOKENS) - 1) * (self.X_E_SPACING - spacing) / 2.0
        # Center the strip on the encoder block instead.
        center_x = self.X_ENC_BLOCK
        n = len(self.SRC_TOKENS)
        first_x = center_x - (n - 1) * spacing / 2.0
        return [first_x + j * spacing for j in range(n)]

    def _tgt_token_xs(self) -> list[float]:
        spacing = 0.95
        center_x = self.X_DEC_BLOCK
        n = len(self.TGT_TOKENS) + 1   # +1 for the placeholder "?" being predicted
        first_x = center_x - (n - 1) * spacing / 2.0
        return [first_x + j * spacing for j in range(n)]

    def _build_e_columns(self) -> list[TensorColumn]:
        """Encoder hidden states e_1..e_3 in BLUE."""
        cols: list[TensorColumn] = []
        # Position e-row centered under encoder block.
        x_center = self.X_ENC_BLOCK
        n = 3
        first_x = x_center - (n - 1) * self.X_E_SPACING / 2.0
        for j in range(n):
            cx = first_x + j * self.X_E_SPACING
            col = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL,
                color=BLUE,
                fill_opacity=0.40,
            ).move_to([cx, self.Y_HIDDEN, 0.0])
            cols.append(col)
        return cols

    def _build_kv_columns(
        self, x_center: float, color: str,
    ) -> list[TensorColumn]:
        """3-column tensor stack centered at ``x_center`` (used for K and V)."""
        cols: list[TensorColumn] = []
        n = 3
        first_x = x_center - (n - 1) * self.X_KV_SPACING / 2.0
        for j in range(n):
            cx = first_x + j * self.X_KV_SPACING
            col = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL,
                color=color,
                fill_opacity=0.45,
            ).move_to([cx, self.Y_QKV, 0.0])
            cols.append(col)
        return cols

    def _build_score_row(self) -> list[Square]:
        """Raw-score / softmax-weight 1×3 cell row at center of frame."""
        cells: list[Square] = []
        for j in range(3):
            cx = self.X_SCORE_BASE + j * self.X_SCORE_SPACING
            raw = self.RAW_SCORES[j]
            raw_max = max(self.RAW_SCORES)
            op = 0.20 + 0.55 * raw / raw_max
            cell = Square(
                side_length=self.SCORE_CELL,
                color=ORANGE,
                stroke_width=1.8,
            ).set_fill(ORANGE, opacity=op)
            cell.move_to([cx, self.Y_SCORES, 0.0])
            cells.append(cell)
        return cells

    # ----------------------------- main -----------------------------
    def construct(self) -> None:
        # ===================== Phase 0: title =====================
        title = (
            MathTex(
                r"\text{Cross-attention: decoder } Q \times "
                r"\text{ encoder } K,\, V"
            )
            .scale(0.55)
            .move_to([0.0, self.Y_TITLE, 0.0])
        )
        self.play(Write(title), run_time=0.6)
        self.wait(0.2)

        # ===================== Phase 1: encoder side — tokens + stack =====================
        src_xs = self._src_token_xs()
        src_tok_mobs: list[Tex] = []
        for j, tok in enumerate(self.SRC_TOKENS):
            t = (
                Tex(rf"\textit{{``{tok}''}}")
                .scale(0.55)
                .move_to([src_xs[j], self.Y_SRC_TOK, 0.0])
            )
            src_tok_mobs.append(t)

        # Combined "Encoder (EN)" label so the language flavor is conveyed
        # without an extra free-floating MathTex that would either crowd the
        # token strip or clip the frame edge.
        enc_block = LabeledBox(
            label=r"\text{Encoder (EN)}",
            width=2.50,
            height=0.78,
            color=GREY_B,
            label_scale=0.55,
            fill_opacity=0.12,
        ).move_to([self.X_ENC_BLOCK, self.Y_BLOCK, 0.0])

        # Tokens → encoder block (one arrow from middle source token to block).
        arr_src_to_enc = _vertical_arrow(
            src_tok_mobs[1], enc_block,
            buff=0.10, tip_length=0.13, color=GREY_B, stroke_width=2.5,
        )

        self.play(
            *[FadeIn(t) for t in src_tok_mobs],
            run_time=0.4,
        )
        self.play(FadeIn(enc_block), Create(arr_src_to_enc), run_time=0.4)
        self.wait(0.15)

        # ===================== Phase 2: encoder hidden states e_1..e_3 =====================
        e_cols = self._build_e_columns()
        e_labels: list[MathTex] = []
        # Place all e labels stacked to the right of the rightmost e column
        # to stay clear of both the enc_block→e vertical arrows and the
        # diagonal e→W_K/W_V arrows.
        e_lbl_x = e_cols[-1].get_right()[0] + 0.18
        for j in range(3):
            lbl = (
                MathTex(rf"e_{j + 1}")
                .scale(0.65)
                .move_to([e_lbl_x, self.Y_HIDDEN + (1 - j) * 0.30, 0.0])
            )
            e_labels.append(lbl)

        # Encoder block → middle e column (single arrow representing the row).
        # arrow_between picks top/bottom attachment because |dy|>|dx| here.
        arr_enc_to_e = arrow_between(
            enc_block, e_cols[1],
            buff=0.10, tip_length=0.13, color=BLUE, stroke_width=2.5,
        )
        self.play(
            *[FadeIn(c) for c in e_cols],
            *[FadeIn(lbl) for lbl in e_labels],
            Create(arr_enc_to_e),
            run_time=0.5,
        )
        self.wait(0.2)

        # ===================== Phase 3: decoder side — tokens + block + d_t =====================
        tgt_xs = self._tgt_token_xs()
        tgt_tok_mobs: list[Tex] = []
        # Generated target tokens in white; the predicted slot rendered as "?".
        all_tgt = list(self.TGT_TOKENS) + ["?"]
        for j, tok in enumerate(all_tgt):
            color = TEAL if j == len(all_tgt) - 1 else WHITE
            t = (
                Tex(rf"\textit{{``{tok}''}}")
                .scale(0.55)
                .move_to([tgt_xs[j], self.Y_TGT_TOK, 0.0])
                .set_color(color)
            )
            tgt_tok_mobs.append(t)

        dec_block = LabeledBox(
            label=r"\text{Decoder (FR)}",
            width=2.50,
            height=0.78,
            color=GREY_B,
            label_scale=0.55,
            fill_opacity=0.12,
        ).move_to([self.X_DEC_BLOCK, self.Y_BLOCK, 0.0])

        # Already-generated target tokens flow up into the decoder block.
        arr_tgt_to_dec = _vertical_arrow(
            tgt_tok_mobs[1], dec_block,
            buff=0.10, tip_length=0.13, color=GREY_B, stroke_width=2.5,
        )

        d_t = TensorColumn(
            dim=self.TENSOR_DIM,
            cell_size=self.CELL,
            color=BLUE,
            fill_opacity=0.40,
        ).move_to([self.X_DT, self.Y_HIDDEN, 0.0])
        d_t_lbl = (
            MathTex(r"d_t")
            .scale(0.62)
            .next_to(d_t, RIGHT, buff=0.10)
        )
        arr_dec_to_dt = _vertical_arrow(
            dec_block, d_t,
            buff=0.10, tip_length=0.13, color=BLUE, stroke_width=2.5,
        )

        self.play(
            *[FadeIn(t) for t in tgt_tok_mobs],
            run_time=0.4,
        )
        self.play(FadeIn(dec_block), Create(arr_tgt_to_dec), run_time=0.4)
        self.play(FadeIn(d_t), FadeIn(d_t_lbl), Create(arr_dec_to_dt), run_time=0.4)
        self.wait(0.2)

        # ===================== Phase 4: encoder K/V projections =====================
        # W_K and W_V boxes between encoder e-row and K/V tensor row.
        w_k = LabeledBox(
            label=r"W_K",
            width=0.85,
            height=0.42,
            color=ORANGE,
            label_scale=0.48,
            fill_opacity=0.18,
        ).move_to([self.X_WK, self.Y_PROJ, 0.0])
        w_v = LabeledBox(
            label=r"W_V",
            width=0.85,
            height=0.42,
            color=GREEN,
            label_scale=0.48,
            fill_opacity=0.18,
        ).move_to([self.X_WV, self.Y_PROJ, 0.0])

        # K, V tensor stacks (3 columns each, ORANGE / GREEN).
        k_cols = self._build_kv_columns(self.X_WK, ORANGE)
        v_cols = self._build_kv_columns(self.X_WV, GREEN)
        k_label = (
            MathTex(r"K")
            .scale(0.62)
            .next_to(k_cols[-1], DOWN, buff=0.10)
        )
        v_label = (
            MathTex(r"V")
            .scale(0.62)
            .next_to(v_cols[-1], DOWN, buff=0.10)
        )

        # Arrows: e centroid → W_K, e centroid → W_V (encoder hidden fans into both).
        arr_e_to_wk = _vertical_arrow(
            e_cols[0], w_k,
            buff=0.10, tip_length=0.13, color=ORANGE, stroke_width=2.5,
        )
        arr_e_to_wv = _vertical_arrow(
            e_cols[2], w_v,
            buff=0.10, tip_length=0.13, color=GREEN, stroke_width=2.5,
        )
        # Arrows: W_K → middle K column, W_V → middle V column.
        arr_wk_to_k = _vertical_arrow(
            w_k, k_cols[1],
            buff=0.08, tip_length=0.12, color=ORANGE, stroke_width=2.2,
        )
        arr_wv_to_v = _vertical_arrow(
            w_v, v_cols[1],
            buff=0.08, tip_length=0.12, color=GREEN, stroke_width=2.2,
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
            run_time=0.6,
        )
        self.wait(0.2)

        # ===================== Phase 5: decoder Q projection =====================
        w_q = LabeledBox(
            label=r"W_Q",
            width=0.85,
            height=0.42,
            color=TEAL,
            label_scale=0.48,
            fill_opacity=0.18,
        ).move_to([self.X_WQ, self.Y_PROJ, 0.0])

        q_col = TensorColumn(
            dim=self.TENSOR_DIM,
            cell_size=self.CELL,
            color=TEAL,
            fill_opacity=0.50,
        ).move_to([self.X_Q, self.Y_QKV, 0.0])
        q_label = (
            MathTex(r"Q")
            .scale(0.62)
            .next_to(q_col, DOWN, buff=0.10)
        )

        arr_dt_to_wq = _vertical_arrow(
            d_t, w_q,
            buff=0.10, tip_length=0.13, color=TEAL, stroke_width=2.5,
        )
        arr_wq_to_q = _vertical_arrow(
            w_q, q_col,
            buff=0.08, tip_length=0.12, color=TEAL, stroke_width=2.2,
        )

        self.play(
            FadeIn(w_q), Create(arr_dt_to_wq),
            run_time=0.45,
        )
        self.play(
            FadeIn(q_col), FadeIn(q_label),
            Create(arr_wq_to_q),
            run_time=0.45,
        )
        self.wait(0.2)

        # ===================== Phase 6: Q · K^T → raw scores (centre) =====================
        score_cells = self._build_score_row()
        score_lbl = (
            MathTex(r"Q\, K^\top")
            .scale(0.58)
            .next_to(score_cells[0], LEFT, buff=0.60)
        )
        # Long horizontal "feeder" arrows: Q → score-row left edge, K-rightmost
        # column → score-row left edge (we use Lines from each side that meet
        # near the score row to suggest "Q meets K"). Two short forced-
        # horizontal arrows from Q (right side) to score row, and from K
        # rightmost column (left side, but really: from K row centroid through
        # the gap) to score row. To avoid arrow_path_clear flagging long
        # diagonals, we route as 2-segment elbows.

        # Q → scores: from Q column (top side, then over to the leftmost score cell).
        # Since score row is at y=Y_SCORES (=-2.45) and Q is at y=Y_QKV (=-1.50),
        # they are *near* the same vertical band, so a 2-segment elbow is small.
        q_pt = q_col.get_left()                     # right side faces center
        # Wait — Q is on the right (X_Q=5.40), score cells span around x=0.
        # So Q's LEFT side faces the centre. Use that.
        q_anchor = [float(q_col.get_left()[0]) - 0.02,
                    float(q_col.get_center()[1]),
                    0.0]
        # Target: rightmost score cell's right edge.
        target_q = [float(score_cells[-1].get_right()[0]) + 0.02,
                    float(score_cells[-1].get_center()[1]),
                    0.0]
        # Two-segment L: from Q horizontally inward to x=target_q[0]+offset,
        # then down/up to target. Since Q.y > target.y (Y_QKV=-1.50 vs
        # Y_SCORES=-2.45), we go over then down.
        elbow_x_q = target_q[0] + 0.30
        seg_q1 = Line(
            start=q_anchor,
            end=[elbow_x_q, q_anchor[1], 0.0],
            color=TEAL, stroke_width=2.2,
        )
        seg_q2 = Arrow(
            start=[elbow_x_q, q_anchor[1], 0.0],
            end=[elbow_x_q, target_q[1] + 0.02, 0.0],
            buff=0.0, stroke_width=2.2, tip_length=0.12, color=TEAL,
        )

        # K rightmost column → score row left edge (mirror of Q feeder).
        # K rightmost col is at K_BASE + 2*KV_SPACING ≈ -5.28 (centered on X_WK=-5.85).
        k_anchor = [float(k_cols[-1].get_right()[0]) + 0.02,
                    float(k_cols[-1].get_center()[1]),
                    0.0]
        target_k = [float(score_cells[0].get_left()[0]) - 0.02,
                    float(score_cells[0].get_center()[1]),
                    0.0]
        elbow_x_k = target_k[0] - 0.30
        seg_k1 = Line(
            start=k_anchor,
            end=[elbow_x_k, k_anchor[1], 0.0],
            color=ORANGE, stroke_width=2.2,
        )
        seg_k2 = Arrow(
            start=[elbow_x_k, k_anchor[1], 0.0],
            end=[elbow_x_k, target_k[1] + 0.02, 0.0],
            buff=0.0, stroke_width=2.2, tip_length=0.12, color=ORANGE,
        )

        self.play(
            *[FadeIn(c) for c in score_cells],
            FadeIn(score_lbl),
            Create(seg_q1), Create(seg_q2),
            Create(seg_k1), Create(seg_k2),
            run_time=0.7,
        )
        self.wait(0.3)

        # ===================== Phase 7: softmax → attention weights =====================
        # Re-color score cells per softmax weights, change to YELLOW, and
        # outline the dominant cell (position 1 = "cat") in PURPLE.
        weight_max = max(self.ATTN_WEIGHTS)
        soft_anims = []
        for j, cell in enumerate(score_cells):
            w = self.ATTN_WEIGHTS[j]
            op = 0.20 + 0.70 * w / weight_max
            soft_anims.append(
                cell.animate.set_fill(YELLOW, opacity=op).set_stroke(YELLOW)
            )

        # Replace "Q K^T" label with "softmax(...)" — fade old, fade new to
        # avoid double-formula at midpoint frame sampling.
        softmax_lbl = (
            MathTex(r"\mathrm{softmax}")
            .scale(0.58)
            .move_to(score_lbl.get_center())
        )
        self.play(
            *soft_anims,
            FadeOut(score_lbl),
            run_time=0.45,
        )
        self.remove(score_lbl)
        self.play(FadeIn(softmax_lbl), run_time=0.25)

        dom_cell = score_cells[self.ATTN_DOMINANT]
        dom_outline = Rectangle(
            width=dom_cell.width + 0.10,
            height=dom_cell.height + 0.10,
            color=PURPLE,
            stroke_width=3,
        ).move_to(dom_cell.get_center())
        self.play(Create(dom_outline), run_time=0.30)
        self.wait(0.4)

        # ===================== Phase 8: weighted sum over V → c_t =====================
        c_t = TensorColumn(
            dim=self.TENSOR_DIM,
            cell_size=self.CELL,
            color=PURPLE,
            fill_opacity=0.55,
        ).move_to([self.X_CTX, self.Y_CTX, 0.0])
        c_t_lbl = (
            MathTex(r"c_t")
            .scale(0.62)
            .next_to(c_t, RIGHT, buff=0.12)
        )

        # Visualize the weighted sum: V columns "pour into" c_t with opacity
        # proportional to attention weight. We do a short ghost-pour like in
        # scaled_dot_product.py phase 5, but condensed.
        ghost_cols: list[TensorColumn] = []
        for j, vc in enumerate(v_cols):
            w = self.ATTN_WEIGHTS[j]
            ghost = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL,
                color=PURPLE,
                fill_opacity=0.10 + 0.55 * w / weight_max,
            ).move_to(vc.get_center())
            ghost_cols.append(ghost)

        self.play(*[FadeIn(g) for g in ghost_cols], run_time=0.30)
        self.play(
            *[g.animate.move_to(c_t.get_center()) for g in ghost_cols],
            FadeIn(c_t),
            FadeIn(c_t_lbl),
            run_time=0.7,
        )
        self.play(*[FadeOut(g) for g in ghost_cols], run_time=0.20)
        for g in ghost_cols:
            self.remove(g)
        self.wait(0.3)

        # ===================== Phase 9: c_t flows back into decoder block =====================
        # 3-segment elbow: from c_t up to a rail near Y_BLOCK, then over to
        # the column directly below dec_block, then a final short arrow whose
        # tip lands ON dec_block.get_bottom() (not inside the box). This
        # keeps the arrow's sampled path entirely outside dec_block's AABB,
        # while the arrow's *endpoint* is < 0.35 from dec_block's center
        # (geometric tolerance for `check_arrow_path_clear`).
        c_top = [float(c_t.get_top()[0]),
                 float(c_t.get_top()[1]) + 0.02,
                 0.0]
        rail_y = self.Y_BLOCK - 0.65       # well below decoder block bottom
        dec_bottom_pt = [
            float(dec_block.get_bottom()[0]),
            float(dec_block.get_bottom()[1]),
            0.0,
        ]
        # Three segments: up, over, up to dec_block bottom centre.
        seg_ctx_up = Line(
            start=c_top,
            end=[c_top[0], rail_y, 0.0],
            color=PURPLE, stroke_width=2.5,
        )
        seg_ctx_over = Line(
            start=[c_top[0], rail_y, 0.0],
            end=[dec_bottom_pt[0], rail_y, 0.0],
            color=PURPLE, stroke_width=2.5,
        )
        seg_ctx_in = Arrow(
            start=[dec_bottom_pt[0], rail_y, 0.0],
            end=dec_bottom_pt,
            buff=0.05, stroke_width=2.5, tip_length=0.14, color=PURPLE,
        )

        self.play(
            Create(seg_ctx_up), Create(seg_ctx_over), Create(seg_ctx_in),
            run_time=0.7,
        )
        # Brief flash on decoder block to signal "context fused in".
        self.play(
            dec_block.box.animate.set_stroke(PURPLE, width=3),
            run_time=0.35,
        )
        self.wait(0.4)

        # ===================== Phase 10: caption =====================
        caption = (
            MathTex(
                r"\text{Decoder uses encoder context: }",
                r"Q \text{ from decoder, } K, V \text{ from encoder}",
            )
            .scale(0.48)
            .move_to([0.0, self.Y_CAPTION, 0.0])
        )
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(1.0)
