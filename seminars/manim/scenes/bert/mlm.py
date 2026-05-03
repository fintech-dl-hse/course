"""BERT Masked Language Model objective scene (V10).

Visualizes MLM: for 5 input tokens ["the", "cat", "[MASK]", "on", "mat"],
the masked position k=2 is recovered from bidirectional context through
an encoder block and vocabulary-projection head.

Key improvements over V1:
- PE shown as inline "+PE_i" annotations (no separate badge boxes)
- Bidirectional attention pattern centered inside encoder block
- Clean vertical prediction chain: h_2 -> W_vocab -> y_2 -> "sat"
- Better spacing and readability

Color convention:
- BLUE -- input embeddings x_i.
- PURPLE -- encoder block, contextual hidden h_i.
- RED -- [MASK] token highlight.
- YELLOW -- highlighted masked position h_2, argmax cell.
- ORANGE -- W_vocab projection.
- GREY_B -- PE annotations, attention pattern.
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
    LEFT,
    Rectangle,
    Scene,
    SurroundingRectangle,
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
    defaults: dict[str, Any] = {"buff": 0.08, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


def _vertical_arrow(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    defaults: dict[str, Any] = {"buff": 0.08, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[1] >= b.get_center()[1]:
        start, end = a.get_bottom(), b.get_top()
    else:
        start, end = a.get_top(), b.get_bottom()
    return Arrow(start=start, end=end, **defaults)


class BERTMaskedLM(Scene):
    """V10: BERT Masked Language Model -- predict [MASK] from context."""

    NUM_TOKENS = 5
    MASK_INDEX = 2
    TENSOR_DIM = 4
    CELL = 0.24

    # Vertical layout
    Y_TITLE = 3.55
    Y_TOKEN = 2.80
    Y_PE_ANNOT = 2.25         # +PE_i inline annotation
    Y_X = 1.50
    Y_ENC_TOP = 0.60          # encoder block top
    Y_ENC_BOT = -0.30         # encoder block bottom
    Y_ENC_MID = 0.15          # encoder block center
    Y_H = -1.15
    Y_W = -2.15               # W_vocab
    Y_Y = -3.00               # softmax y_2
    Y_PRED = -3.70            # predicted token

    # Horizontal layout -- 5 columns centered at x=0
    X_SPACING = 2.20

    # Encoder block
    ENC_W = 10.50
    ENC_H = 0.95

    # W_vocab dims
    W_VOCAB_W = 1.10
    W_VOCAB_H = 0.50

    # Colors
    COLOR_X = BLUE
    COLOR_H = PURPLE
    COLOR_MASK = RED
    COLOR_HIGHLIGHT = YELLOW
    COLOR_PE = GREY_B
    COLOR_W = ORANGE

    def _x_for(self, i: int) -> float:
        return (i - (self.NUM_TOKENS - 1) / 2.0) * self.X_SPACING

    def construct(self) -> None:
        # ===================== Title =====================
        title = MathTex(
            r"\mathcal{L}_{\mathrm{MLM}} = -\log P(\text{token}_k \mid "
            r"\text{context}_{\neq k})"
        ).scale(0.55)
        title.move_to([0.0, self.Y_TITLE, 0.0])
        self.play(Write(title), run_time=0.6)
        self.wait(0.2)

        # ===================== Input tokens =====================
        token_strings = ["the", "cat", "[MASK]", "on", "mat"]
        token_mobs: list[VGroup] = []
        for i, tok in enumerate(token_strings):
            x = self._x_for(i)
            if i == self.MASK_INDEX:
                txt = (
                    Tex(rf"\textit{{[MASK]}}", color=self.COLOR_MASK)
                    .scale(0.58)
                    .move_to([x, self.Y_TOKEN, 0.0])
                )
                border = Rectangle(
                    width=txt.width + 0.20,
                    height=txt.height + 0.16,
                    color=self.COLOR_MASK,
                    stroke_width=2.5,
                ).move_to(txt.get_center())
                grp = VGroup(border, txt)
            else:
                txt = (
                    Tex(rf"\textit{{``{tok}''}}")
                    .scale(0.58)
                    .move_to([x, self.Y_TOKEN, 0.0])
                )
                grp = VGroup(txt)
            token_mobs.append(grp)

        self.play(*[FadeIn(t) for t in token_mobs], run_time=0.5)
        self.wait(0.2)

        # ===================== PE annotations + embeddings =====================
        x_cols: list[TensorColumn] = []
        x_labels: list[MathTex] = []
        pe_annotations: list[MathTex] = []
        tok_to_x_arrows: list[Arrow] = []

        for i in range(self.NUM_TOKENS):
            x = self._x_for(i)

            # +PE_i annotation (inline text, no box)
            pe_annot = (
                MathTex(rf"+PE_{i}")
                .scale(0.45)
                .move_to([x, self.Y_PE_ANNOT, 0.0])
                .set_color(self.COLOR_PE)
            )
            pe_annotations.append(pe_annot)

            # Embedding column
            col = TensorColumn(
                dim=self.TENSOR_DIM, cell_size=self.CELL,
                color=self.COLOR_X, fill_opacity=0.40,
            ).move_to([x, self.Y_X, 0.0])
            x_cols.append(col)

            # Label
            lbl = MathTex(rf"x_{i}").scale(0.58)
            lbl.next_to(col, RIGHT, buff=0.08)
            x_labels.append(lbl)

            # Arrow from token to embedding
            arr = _vertical_arrow(
                token_mobs[i], col,
                buff=0.18, tip_length=0.10,
                color=WHITE, stroke_width=2.0,
            )
            tok_to_x_arrows.append(arr)

        self.play(
            *[FadeIn(p) for p in pe_annotations],
            run_time=0.3,
        )
        self.play(
            *[FadeIn(c) for c in x_cols],
            *[FadeIn(l) for l in x_labels],
            *[Create(a) for a in tok_to_x_arrows],
            run_time=0.6,
        )
        self.wait(0.2)

        # ===================== Encoder block =====================
        encoder = LabeledBox(
            label=r"\mathrm{Encoder\ Block}",
            width=self.ENC_W, height=self.ENC_H,
            color=PURPLE, label_scale=0.52, fill_opacity=0.10,
        ).move_to([0.0, self.Y_ENC_MID, 0.0])

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

        # Bidirectional attention pattern -- centered inside encoder
        # 5 evenly spaced anchor points across the encoder
        attn_y = self.Y_ENC_MID
        anchor_positions: list[float] = [self._x_for(i) for i in range(5)]

        attn_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # adjacent
            (0, 2), (2, 4),                    # skip-1
            (0, 4), (1, 3),                    # long-range
        ]
        attn_lines: list[Line] = []
        for a_idx, b_idx in attn_pairs:
            ln = Line(
                start=[anchor_positions[a_idx], attn_y, 0.0],
                end=[anchor_positions[b_idx], attn_y, 0.0],
                color=GREY_B, stroke_width=1.8,
            ).set_opacity(0.50)
            attn_lines.append(ln)

        # Small dots at anchors
        attn_dots: list[VMobject] = []
        for ax in anchor_positions:
            dot = Rectangle(
                width=0.12, height=0.12,
                color=WHITE, stroke_width=1.5,
            ).set_fill(WHITE, opacity=0.55).move_to([ax, attn_y, 0.0])
            attn_dots.append(dot)

        self.play(
            *[FadeIn(d) for d in attn_dots],
            *[Create(l) for l in attn_lines],
            run_time=0.6,
        )
        self.wait(0.3)

        # ===================== Contextual hidden h_i =====================
        h_cols: list[TensorColumn] = []
        h_labels: list[MathTex] = []
        enc_to_h_arrows: list[Arrow] = []

        for i in range(self.NUM_TOKENS):
            x = self._x_for(i)
            highlight = 1 if i == self.MASK_INDEX else None
            base_opacity = 0.65 if i == self.MASK_INDEX else 0.40
            col = TensorColumn(
                dim=self.TENSOR_DIM, cell_size=self.CELL,
                color=self.COLOR_H, fill_opacity=base_opacity,
                highlight_index=highlight,
                highlight_color="#F5C518", highlight_opacity=0.95,
            ).move_to([x, self.Y_H, 0.0])
            h_cols.append(col)

            lbl = MathTex(rf"h_{i}").scale(0.58)
            lbl.next_to(col, RIGHT, buff=0.08)
            h_labels.append(lbl)

            arr = _vertical_arrow(
                encoder, col,
                buff=0.06, tip_length=0.10,
                color=self.COLOR_H, stroke_width=2.0,
            )
            enc_to_h_arrows.append(arr)

        # Yellow glow ring around h_2
        h2_col = h_cols[self.MASK_INDEX]
        h_glow = SurroundingRectangle(
            h2_col, color=self.COLOR_HIGHLIGHT,
            buff=0.06, stroke_width=2.5,
        )

        self.play(
            *[FadeIn(c) for c in h_cols],
            *[FadeIn(l) for l in h_labels],
            *[Create(a) for a in enc_to_h_arrows],
            FadeIn(h_glow),
            run_time=0.7,
        )
        self.wait(0.3)

        # ===================== h_2 -> W_vocab -> y_2 -> "sat" =====================
        # Clean vertical chain below h_2
        x_mask = self._x_for(self.MASK_INDEX)

        # W_vocab box
        w_vocab = LabeledBox(
            label=r"W_{\mathrm{vocab}}",
            width=self.W_VOCAB_W, height=self.W_VOCAB_H,
            color=self.COLOR_W, label_scale=0.50, fill_opacity=0.20,
        ).move_to([x_mask, self.Y_W, 0.0])

        arr_h2_to_w = _vertical_arrow(
            h2_col, w_vocab,
            buff=0.08, tip_length=0.12,
            color=self.COLOR_W, stroke_width=2.5,
        )

        # y_2 softmax tensor (directly below W_vocab)
        argmax_idx = 1
        y2 = TensorColumn(
            dim=5, cell_size=self.CELL,
            color=self.COLOR_W, fill_opacity=0.30,
            highlight_index=argmax_idx,
            highlight_color="#F5C518", highlight_opacity=0.95,
        ).move_to([x_mask, self.Y_Y, 0.0])
        y2_lbl = (
            MathTex(r"y_2").scale(0.58)
            .next_to(y2, LEFT, buff=0.10)
        )

        arr_w_to_y = _vertical_arrow(
            w_vocab, y2,
            buff=0.08, tip_length=0.12,
            color=self.COLOR_W, stroke_width=2.5,
        )

        # Predicted token "sat" (to the right of y_2)
        pred_token = (
            Tex(rf"\textit{{``sat''}}", color=self.COLOR_HIGHLIGHT)
            .scale(0.65)
            .move_to([x_mask + 2.40, self.Y_Y, 0.0])
        )
        arr_y_to_pred = _horizontal_arrow(
            y2, pred_token,
            buff=0.10, tip_length=0.12,
            color=self.COLOR_HIGHLIGHT, stroke_width=2.5,
        )

        self.play(
            FadeIn(w_vocab), Create(arr_h2_to_w),
            run_time=0.45,
        )
        self.play(
            FadeIn(y2), FadeIn(y2_lbl),
            Create(arr_w_to_y),
            run_time=0.50,
        )
        self.play(
            FadeIn(pred_token), Create(arr_y_to_pred),
            run_time=0.45,
        )

        # Caption
        caption = (
            MathTex(
                r"\arg\max_v y_2[v] = \text{``sat''} \;\checkmark"
            )
            .scale(0.52)
            .move_to([0.0, self.Y_PRED, 0.0])
        )
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(1.0)
