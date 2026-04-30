"""BERT Masked Language Model objective scene (V10).

Сцена визуализирует цель MLM: для входной последовательности из 5 токенов
``["the", "cat", "[MASK]", "on", "mat"]`` показывается, как маскированная
позиция ``k=2`` восстанавливается из двунаправленного контекста через
один блок энкодера и vocabulary-projection голову.

Ключевые элементы:

1. Заголовок — формула ``L_{MLM} = -\\log P(\\text{token}_k \\mid
   \\text{context}_{\\neq k})``.
2. Строка из 5 входных токенов; ``[MASK]`` подсвечен красной рамкой.
3. Каждый токен → embedding ``x_i`` (TensorColumn dim=4) + бейдж ``+PE_i``.
4. Все 5 эмбеддингов идут в один большой ``EncoderBlock`` LabeledBox; внутри
   рисуется упрощённая bidirectional attention сетка (несколько перекрёстных
   стрелок), показывающая, что каждая позиция смотрит на каждую.
5. Выход — 5 контекстуальных скрытых тензоров ``h_0..h_4``; ``h_2``
   подсвечен (жёлтый) как маскированная позиция.
6. ``h_2`` → orange ``W_{vocab}`` LabeledBox → softmax ``y_2`` (TensorColumn
   с 5 ячейками, argmax-ячейка подсвечена).
7. Финальная стрелка из argmax-ячейки ведёт в предсказанный токен «sat»,
   совпадающий с истинным замаскированным словом.

Сцена использует общие примитивы ``shared.neural`` (TensorColumn,
LabeledBox, arrow_between). Локальный ``_horizontal_arrow`` — копия
паттерна из ``rnn_forward.py`` / ``encoder_block.py`` — нужен, чтобы
длинные горизонтальные стрелки от строк колонок к энкодеру не
переключались на верх/низ-крепление.
"""
from __future__ import annotations

from typing import Any

from manim import (
    Arrow,
    BLUE,
    Create,
    DOWN,
    FadeIn,
    GREY_B,
    Line,
    MathTex,
    ORANGE,
    PURPLE,
    RED,
    RIGHT,
    Rectangle,
    Scene,
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

    Локальная копия паттерна из ``rnn_forward.py`` — нужна, когда
    ``arrow_between`` выбрал бы вертикальное крепление и линия прошла
    бы через посторонний узел.
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
    defaults: dict[str, Any] = {"buff": 0.08, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[1] >= b.get_center()[1]:
        start, end = a.get_bottom(), b.get_top()
    else:
        start, end = a.get_top(), b.get_bottom()
    return Arrow(start=start, end=end, **defaults)


class BERTMaskedLM(Scene):
    """V10: BERT Masked Language Model — предсказание [MASK] из контекста."""

    # ---- Layout constants — 720p (-qm) frame ±(7.11, 4.0) ----
    NUM_TOKENS = 5
    MASK_INDEX = 2
    TENSOR_DIM = 4
    CELL = 0.22

    # Vertical anchors.
    Y_TITLE = 3.55
    Y_TOKEN = 2.75
    Y_PE = 2.10
    Y_X = 1.30
    Y_ENC = 0.05      # encoder block center
    Y_H = -1.45
    Y_W = -2.40       # W_vocab box center
    Y_Y = -3.25       # softmax + predicted token row

    # Horizontal layout — 5 token columns centered around x=0.
    X_SPACING = 2.40

    # Encoder block dims (single big LabeledBox spanning the row).
    ENC_W = 11.0
    ENC_H = 1.10

    # PE badge dims.
    PE_W = 0.55
    PE_H = 0.34

    # W_vocab dims.
    W_VOCAB_W = 1.20
    W_VOCAB_H = 0.55

    # Colors.
    COLOR_X = BLUE
    COLOR_H = PURPLE
    COLOR_MASK = RED
    COLOR_HIGHLIGHT = YELLOW
    COLOR_PE = GREY_B
    COLOR_W = ORANGE

    def _x_for(self, i: int) -> float:
        """X-координата центра i-го столбца (i=0..4)."""
        return (i - (self.NUM_TOKENS - 1) / 2.0) * self.X_SPACING

    def construct(self) -> None:
        # ===================== Phase 0: title =====================
        title = MathTex(
            r"\mathcal{L}_{\mathrm{MLM}} = -\log P(\text{token}_k \mid "
            r"\text{context}_{\neq k})"
        ).scale(0.55)
        title.move_to([0.0, self.Y_TITLE, 0.0])
        self.play(Write(title), run_time=0.6)
        self.wait(0.2)

        # ===================== Phase 1: input tokens =====================
        token_strings = ["the", "cat", "[MASK]", "on", "mat"]
        token_mobs: list[VGroup] = []
        for i, tok in enumerate(token_strings):
            x = self._x_for(i)
            if i == self.MASK_INDEX:
                # [MASK] visually distinct — red italic + red border box.
                txt = (
                    Tex(rf"\textit{{[MASK]}}", color=self.COLOR_MASK)
                    .scale(0.55)
                    .move_to([x, self.Y_TOKEN, 0.0])
                )
                border = Rectangle(
                    width=txt.width + 0.20,
                    height=txt.height + 0.16,
                    color=self.COLOR_MASK,
                    stroke_width=2.0,
                ).move_to(txt.get_center())
                grp = VGroup(border, txt)
            else:
                txt = (
                    Tex(rf"\textit{{``{tok}''}}")
                    .scale(0.55)
                    .move_to([x, self.Y_TOKEN, 0.0])
                )
                grp = VGroup(txt)
            token_mobs.append(grp)

        self.play(*[FadeIn(t) for t in token_mobs], run_time=0.5)
        self.wait(0.2)

        # ===================== Phase 2: x_i embeddings + PE badges =====================
        x_cols: list[TensorColumn] = []
        x_labels: list[MathTex] = []
        pe_badges: list[LabeledBox] = []
        tok_to_x_arrows: list[Arrow] = []

        for i in range(self.NUM_TOKENS):
            x = self._x_for(i)
            col = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL,
                color=self.COLOR_X,
                fill_opacity=0.40,
            ).move_to([x, self.Y_X, 0.0])
            x_cols.append(col)

            lbl = MathTex(rf"x_{i}").scale(0.50)
            lbl.next_to(col, RIGHT, buff=0.08)
            x_labels.append(lbl)

            # PE badge — small grey labeled box "+PE_i" between token and x_i.
            pe = LabeledBox(
                label=rf"+PE_{i}",
                width=self.PE_W,
                height=self.PE_H,
                color=self.COLOR_PE,
                label_scale=0.36,
                fill_opacity=0.15,
            ).move_to([x, self.Y_PE, 0.0])
            pe_badges.append(pe)

            arr = _vertical_arrow(
                token_mobs[i], col,
                buff=0.05, tip_length=0.10,
                color=WHITE, stroke_width=2.0,
            )
            tok_to_x_arrows.append(arr)

        self.play(
            *[FadeIn(p) for p in pe_badges],
            run_time=0.4,
        )
        self.play(
            *[FadeIn(c) for c in x_cols],
            *[FadeIn(l) for l in x_labels],
            *[Create(a) for a in tok_to_x_arrows],
            run_time=0.6,
        )
        self.wait(0.3)

        # ===================== Phase 3: encoder block =====================
        encoder = LabeledBox(
            label=r"\mathrm{Encoder\ Block}",
            width=self.ENC_W,
            height=self.ENC_H,
            color=PURPLE,
            label_scale=0.42,
            fill_opacity=0.10,
        ).move_to([0.0, self.Y_ENC, 0.0])
        # Push the label to the left edge so the bidirectional crisscross has
        # room in the center of the box.
        encoder.label_tex.move_to(
            [encoder.box.get_left()[0] + 1.0, self.Y_ENC, 0.0]
        )

        # Arrows: each x_i (top edge) → encoder (top edge).
        x_to_enc_arrows: list[Arrow] = []
        for col in x_cols:
            arr = _vertical_arrow(
                col, encoder,
                buff=0.06, tip_length=0.10,
                color=self.COLOR_X, stroke_width=2.0,
            )
            x_to_enc_arrows.append(arr)

        self.play(FadeIn(encoder), run_time=0.4)
        self.play(*[Create(a) for a in x_to_enc_arrows], run_time=0.5)

        # Bidirectional attention pattern inside encoder: short crisscross
        # lines between 5 anchor points along the encoder's vertical center.
        # Use Lines (not Arrows) so they don't trip arrow-path-clear.
        attn_y = self.Y_ENC
        attn_x_inset = 1.85  # leftmost attention anchor x (after the label)
        attn_pts: list[list[float]] = []
        # 5 anchor points spread across the right ~70% of the encoder box.
        n_pts = 5
        for j in range(n_pts):
            ax = attn_x_inset + j * 1.05
            attn_pts.append([ax, attn_y, 0.0])

        # A handful of crisscross lines: a few "k attends to all" pairs +
        # a few near-neighbour pairs. Keep total <= 8 lines so the picture
        # stays readable.
        attn_pairs = [
            (2, 0), (2, 1), (2, 3), (2, 4),  # masked position attends to all
            (0, 4), (1, 3), (0, 2), (3, 4),  # a few extra ties
        ]
        attn_lines: list[Line] = []
        for a_idx, b_idx in attn_pairs:
            ln = Line(
                start=attn_pts[a_idx],
                end=attn_pts[b_idx],
                color=GREY_B,
                stroke_width=1.6,
            ).set_opacity(0.55)
            attn_lines.append(ln)

        # Tiny dots at each anchor to make it read as a graph.
        attn_dots: list[VMobject] = []
        for p in attn_pts:
            dot = Rectangle(
                width=0.10, height=0.10,
                color=GREY_B, stroke_width=1.5,
            ).set_fill(GREY_B, opacity=0.60).move_to(p)
            attn_dots.append(dot)

        self.play(
            *[FadeIn(d) for d in attn_dots],
            *[Create(l) for l in attn_lines],
            run_time=0.7,
        )
        self.wait(0.3)

        # ===================== Phase 4: contextual hidden h_i =====================
        h_cols: list[TensorColumn] = []
        h_labels: list[MathTex] = []
        enc_to_h_arrows: list[Arrow] = []
        h_glow: Rectangle | None = None  # separate top-level mobject for h_2

        for i in range(self.NUM_TOKENS):
            x = self._x_for(i)
            # Highlight a cell in h_2 (yellow inside the column).
            highlight = 1 if i == self.MASK_INDEX else None
            # For the masked position use a brighter base color so the column
            # itself stands out — no extra wrapping border (would create a
            # bigger BB that swallows arrow endpoints).
            base_opacity = 0.65 if i == self.MASK_INDEX else 0.40
            col = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL,
                color=self.COLOR_H,
                fill_opacity=base_opacity,
                highlight_index=highlight,
                highlight_color="#F5C518",
                highlight_opacity=0.95,
            ).move_to([x, self.Y_H, 0.0])
            h_cols.append(col)

            lbl = MathTex(rf"h_{i}").scale(0.50)
            lbl.next_to(col, RIGHT, buff=0.08)
            h_labels.append(lbl)

            arr = _vertical_arrow(
                encoder, col,
                buff=0.06, tip_length=0.10,
                color=self.COLOR_H, stroke_width=2.0,
            )
            enc_to_h_arrows.append(arr)

        # Build a thin yellow glow ring around h_2 as a separate top-level
        # mobject. It's geometrically TIGHT to the column so its AABB equals
        # the column's AABB ± a hair, which lets endpoint detection still
        # treat the column as an arrow endpoint.
        h2_col = h_cols[self.MASK_INDEX]
        h_glow = Rectangle(
            width=h2_col.width + 0.10,
            height=h2_col.height + 0.10,
            color=self.COLOR_HIGHLIGHT,
            stroke_width=2.5,
        ).move_to(h2_col.get_center())

        self.play(
            *[FadeIn(c) for c in h_cols],
            *[FadeIn(l) for l in h_labels],
            *[Create(a) for a in enc_to_h_arrows],
            FadeIn(h_glow),
            run_time=0.7,
        )
        self.wait(0.3)

        # ===================== Phase 5: h_2 → W_vocab → y_2 =====================
        h2_inner = h_cols[self.MASK_INDEX]

        # W_vocab box centered at x=0 (under h_2), at Y_W.
        x_mask = self._x_for(self.MASK_INDEX)
        w_vocab = LabeledBox(
            label=r"W_{\mathrm{vocab}}",
            width=self.W_VOCAB_W,
            height=self.W_VOCAB_H,
            color=self.COLOR_W,
            label_scale=0.40,
            fill_opacity=0.20,
        ).move_to([x_mask, self.Y_W, 0.0])

        # softmax y_2 — 5-cell horizontal-ish column to the right of W_vocab.
        # Argmax index = 1 (the "sat" cell).
        argmax_idx = 1
        y2 = TensorColumn(
            dim=5,
            cell_size=self.CELL,
            color=self.COLOR_W,
            fill_opacity=0.30,
            highlight_index=argmax_idx,
            highlight_color="#F5C518",
            highlight_opacity=0.95,
        ).move_to([x_mask + 1.50, self.Y_Y + 0.30, 0.0])
        # Move y2 up so its center sits between W_vocab and predicted-token row.
        y2.move_to([x_mask + 1.50, (self.Y_W + self.Y_Y) / 2.0, 0.0])
        y2_lbl = MathTex(r"y_2").scale(0.50).next_to(y2, RIGHT, buff=0.08)

        # Vocabulary stub labels next to y2 cells. We use a single Tex with
        # a tall character ("dog" has descender) to anchor a uniform baseline,
        # then scale so even short ascender-only words (``ran``, ``ate``) clear
        # the lint min-height (0.14). Non-italic — italic letters have a
        # smaller rendered BB which trips the lint on 3-char words.
        vocab_labels = ["dog", "sat", "ran", "is", "ate"]
        vocab_label_mobs: list[Tex] = []
        for k, w in enumerate(vocab_labels):
            t = (
                Tex(rf"{w}")
                .scale(0.62)
                .next_to(y2.cells[k], [-1.0, 0.0, 0.0], buff=0.14)
            )
            vocab_label_mobs.append(t)

        # Predicted token "sat" — to the right of y2, on Y_Y.
        pred_token = (
            Tex(rf"\textit{{``sat''}}", color=self.COLOR_HIGHLIGHT)
            .scale(0.65)
            .move_to([x_mask + 4.20, y2.get_center()[1], 0.0])
        )

        # Arrows: h_2 → W_vocab; W_vocab → y_2; y_2[argmax] → pred_token.
        arr_h2_to_w = _vertical_arrow(
            h2_inner, w_vocab,
            buff=0.08, tip_length=0.12, color=self.COLOR_W,
            stroke_width=2.5,
        )
        # W_vocab is below-left of y2; use horizontal-style arrow with elbow
        # avoidance: just go straight from w_vocab right edge to y2 left edge
        # (they're at slightly different y, but |dx|≈1.0 > |dy|≈0.42 so
        # arrow_between will pick horizontal attachment — safe).
        arr_w_to_y = arrow_between(
            w_vocab, y2,
            buff=0.10, tip_length=0.12,
            color=self.COLOR_W, stroke_width=2.5,
        )
        # y_2 argmax cell → predicted token. Use horizontal arrow from the
        # argmax cell to the pred_token text.
        arr_y_to_pred = _horizontal_arrow(
            y2.cells[argmax_idx], pred_token,
            buff=0.10, tip_length=0.12,
            color=self.COLOR_HIGHLIGHT, stroke_width=2.5,
        )

        self.play(
            FadeIn(w_vocab),
            Create(arr_h2_to_w),
            run_time=0.45,
        )
        self.play(
            FadeIn(y2),
            FadeIn(y2_lbl),
            *[FadeIn(t) for t in vocab_label_mobs],
            Create(arr_w_to_y),
            run_time=0.55,
        )
        self.play(
            FadeIn(pred_token),
            Create(arr_y_to_pred),
            run_time=0.45,
        )

        # Caption: prediction matches the original masked token.
        caption = (
            MathTex(
                r"\arg\max_v y_2[v] = \text{``sat''} \;\checkmark"
            )
            .scale(0.50)
            .to_edge(DOWN, buff=0.18)
        )
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(1.0)
