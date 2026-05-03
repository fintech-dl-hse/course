"""Full encoder-decoder forward pass -- translation (V15).

Encoder (left) processes source tokens in parallel; decoder (right) processes
target tokens sequentially with cross-attention to encoder's K/V.

Layout: encoder and decoder as two vertical pipelines side by side.
Cross-attention bridge connects encoder K/V to decoder via clean horizontal
arrows at a visible y-level.

Color convention:
- BLUE -- embeddings, encoder hidden, decoder hidden d_t.
- ORANGE -- W_K and K stack.
- GREEN -- W_V and V stack.
- TEAL -- cross-attn sublayer, predicted token.
- PURPLE -- causal self-attn sublayer.
- YELLOW -- softmax output.
- GREY_B -- neutral blocks (encoder/decoder/W_vocab).
"""
from __future__ import annotations

from typing import Any

from manim import (
    Arrow,
    BLUE,
    Create,
    DOWN,
    FadeIn,
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
    defaults: dict[str, Any] = {"buff": 0.08, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


def _vertical_arrow(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    defaults: dict[str, Any] = {"buff": 0.06, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[1] >= b.get_center()[1]:
        start, end = a.get_bottom(), b.get_top()
    else:
        start, end = a.get_top(), b.get_bottom()
    return Arrow(start=start, end=end, **defaults)


class EncoderDecoderForward(Scene):
    """V15: full encoder-decoder forward pass for translation EN -> FR."""

    TENSOR_DIM = 4
    CELL = 0.22

    # --- Vertical layout ---
    Y_TITLE = 3.75
    Y_TOK = 3.30
    Y_EMB = 2.55
    Y_PE_LBL = 2.00

    # Encoder/decoder blocks
    Y_BLOCK_TOP = 1.55
    Y_BLOCK_BOT = -0.15
    Y_BLOCK_MID = (Y_BLOCK_TOP + Y_BLOCK_BOT) / 2.0

    # Encoder outputs
    Y_HIDDEN = -0.85
    Y_KV = -2.10              # K/V stacks (more room below e labels)

    # Decoder output
    Y_DT = -0.90
    Y_VOCAB = -1.90
    Y_PRED = -2.80
    Y_CAPTION = -3.65

    # --- Horizontal layout ---
    X_ENC = -3.50              # encoder center
    X_DEC = 3.50               # decoder center

    # Source/target tokens
    SRC_TOKENS = ("the", "cat", "sat")
    TGT_TOKENS = ("<bos>", "le", "chat")
    PRED_TOKEN = "assis"

    X_EMB_SPACING = 0.60
    X_TOK_SPACING = 1.00

    # K/V stacks
    X_KV_INNER_SPACING = 0.30
    X_K_CENTER_OFFSET = -0.90
    X_V_CENTER_OFFSET = 0.90

    # Block dims
    ENC_BLOCK_W = 2.80
    ENC_BLOCK_H = 1.75
    DEC_BLOCK_W = 2.80
    DEC_BLOCK_H = 1.75
    SUBLAYER_W = 2.40
    SUBLAYER_H = 0.50

    VOCAB_BOX_W = 0.90
    VOCAB_BOX_H = 0.45
    WPROJ_W = 0.80
    WPROJ_H = 0.38

    # ----------------------------- helpers -----------------------------
    def _strip_xs(self, n: int, center_x: float, spacing: float) -> list[float]:
        first_x = center_x - (n - 1) * spacing / 2.0
        return [first_x + j * spacing for j in range(n)]

    def _build_emb_row(
        self, center_x: float, n: int, color: str,
        fill_opacity: float = 0.40,
    ) -> list[TensorColumn]:
        cols: list[TensorColumn] = []
        xs = self._strip_xs(n, center_x, self.X_EMB_SPACING)
        for cx in xs:
            col = TensorColumn(
                dim=self.TENSOR_DIM, cell_size=self.CELL,
                color=color, fill_opacity=fill_opacity,
            ).move_to([cx, self.Y_EMB, 0.0])
            cols.append(col)
        return cols

    def _build_e_row(self) -> list[TensorColumn]:
        cols: list[TensorColumn] = []
        xs = self._strip_xs(3, self.X_ENC, self.X_EMB_SPACING)
        for cx in xs:
            col = TensorColumn(
                dim=self.TENSOR_DIM, cell_size=self.CELL,
                color=BLUE, fill_opacity=0.50,
            ).move_to([cx, self.Y_HIDDEN, 0.0])
            cols.append(col)
        return cols

    def _build_kv_stack(
        self, center_x_abs: float, color: str,
    ) -> list[TensorColumn]:
        cols: list[TensorColumn] = []
        xs = self._strip_xs(3, center_x_abs, self.X_KV_INNER_SPACING)
        for cx in xs:
            col = TensorColumn(
                dim=self.TENSOR_DIM, cell_size=self.CELL,
                color=color, fill_opacity=0.55,
            ).move_to([cx, self.Y_KV, 0.0])
            cols.append(col)
        return cols

    def _build_dec_block(self) -> tuple[LabeledBox, LabeledBox, LabeledBox]:
        outer = LabeledBox(
            label=r"\text{Decoder Block} \times N",
            width=self.DEC_BLOCK_W, height=self.DEC_BLOCK_H,
            color=GREY_B, label_scale=0.52, fill_opacity=0.10,
        ).move_to([self.X_DEC, self.Y_BLOCK_MID, 0.0])
        outer.label_tex.move_to(
            [self.X_DEC, self.Y_BLOCK_TOP - 0.16, 0.0]
        )
        causal = LabeledBox(
            label=r"\text{Causal Self-Attn}",
            width=self.SUBLAYER_W, height=self.SUBLAYER_H,
            color=PURPLE, label_scale=0.48, fill_opacity=0.22,
        ).move_to([self.X_DEC, self.Y_BLOCK_MID + 0.35, 0.0])
        cross = LabeledBox(
            label=r"\text{Cross-Attn}",
            width=self.SUBLAYER_W, height=self.SUBLAYER_H,
            color=TEAL, label_scale=0.48, fill_opacity=0.22,
        ).move_to([self.X_DEC, self.Y_BLOCK_MID - 0.35, 0.0])
        return outer, causal, cross

    def _build_enc_block(self) -> LabeledBox:
        return LabeledBox(
            label=r"\text{Encoder Block} \times N",
            width=self.ENC_BLOCK_W, height=self.ENC_BLOCK_H,
            color=GREY_B, label_scale=0.52, fill_opacity=0.10,
        ).move_to([self.X_ENC, self.Y_BLOCK_MID, 0.0])

    # ----------------------------- main -----------------------------
    def construct(self) -> None:
        # ===================== Title =====================
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

        # ===================== Source tokens (left) =====================
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

        src_side_lbl = (
            MathTex(r"\text{EN}")
            .scale(0.55)
            .move_to([self.X_ENC - 2.20, self.Y_TOK, 0.0])
            .set_color(GREY_B)
        )

        self.play(
            *[FadeIn(t) for t in src_tok_mobs],
            FadeIn(src_side_lbl),
            run_time=0.4,
        )

        # ===================== Source embeddings + PE =====================
        src_emb_cols = self._build_emb_row(self.X_ENC, len(self.SRC_TOKENS), BLUE)
        src_pe_lbl = (
            MathTex(r"+\,PE_i")
            .scale(0.50)
            .next_to(src_emb_cols[-1], RIGHT, buff=0.12)
        )

        src_tok_to_emb: list[Arrow] = []
        for j in range(len(self.SRC_TOKENS)):
            arr = _vertical_arrow(
                src_tok_mobs[j], src_emb_cols[j],
                buff=0.12, tip_length=0.10, color=BLUE, stroke_width=2.0,
            )
            src_tok_to_emb.append(arr)

        self.play(
            *[FadeIn(c) for c in src_emb_cols],
            *[Create(a) for a in src_tok_to_emb],
            FadeIn(src_pe_lbl),
            run_time=0.45,
        )

        # ===================== Encoder block =====================
        enc_block = self._build_enc_block()
        arr_emb_to_enc = _vertical_arrow(
            src_emb_cols[1], enc_block,
            buff=0.10, tip_length=0.12, color=GREY_B, stroke_width=2.5,
        )
        self.play(FadeIn(enc_block), Create(arr_emb_to_enc), run_time=0.45)

        # ===================== Encoder hidden e_1..e_3 =====================
        e_cols = self._build_e_row()
        # Per-column labels to the right of each column
        e_labels: list[MathTex] = []
        for j in range(3):
            lbl = (
                MathTex(rf"e_{j + 1}")
                .scale(0.52)
                .next_to(e_cols[j], RIGHT, buff=0.06)
            )
            e_labels.append(lbl)

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

        # ===================== K, V projections =====================
        x_wk = self.X_ENC + self.X_K_CENTER_OFFSET
        x_wv = self.X_ENC + self.X_V_CENTER_OFFSET
        y_proj = (self.Y_HIDDEN + self.Y_KV) / 2 - 0.05
        w_k = LabeledBox(
            label=r"W_K", width=self.WPROJ_W, height=self.WPROJ_H,
            color=ORANGE, label_scale=0.50, fill_opacity=0.20,
        ).move_to([x_wk, y_proj, 0.0])
        w_v = LabeledBox(
            label=r"W_V", width=self.WPROJ_W, height=self.WPROJ_H,
            color=GREEN, label_scale=0.50, fill_opacity=0.20,
        ).move_to([x_wv, y_proj, 0.0])

        k_cols = self._build_kv_stack(x_wk, ORANGE)
        v_cols = self._build_kv_stack(x_wv, GREEN)
        k_label = (
            MathTex(r"K").scale(0.58)
            .next_to(k_cols[-1], RIGHT, buff=0.10)
        )
        v_label = (
            MathTex(r"V").scale(0.58)
            .next_to(v_cols[-1], RIGHT, buff=0.10)
        )

        arr_e_to_wk = _vertical_arrow(
            e_cols[0], w_k, buff=0.08, tip_length=0.10,
            color=ORANGE, stroke_width=2.0,
        )
        arr_e_to_wv = _vertical_arrow(
            e_cols[2], w_v, buff=0.08, tip_length=0.10,
            color=GREEN, stroke_width=2.0,
        )
        arr_wk_to_k = _vertical_arrow(
            w_k, k_cols[1], buff=0.06, tip_length=0.10,
            color=ORANGE, stroke_width=2.0,
        )
        arr_wv_to_v = _vertical_arrow(
            w_v, v_cols[1], buff=0.06, tip_length=0.10,
            color=GREEN, stroke_width=2.0,
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

        # ===================== Target tokens (right) =====================
        tgt_xs = self._strip_xs(
            len(self.TGT_TOKENS), self.X_DEC, self.X_TOK_SPACING
        )
        tgt_tok_mobs: list[Tex] = []
        for j, tok in enumerate(self.TGT_TOKENS):
            if tok.startswith("<"):
                inner = tok.strip("<>")
                t = (
                    Tex(rf"\textless\textit{{{inner}}}\textgreater")
                    .scale(0.50)
                    .move_to([tgt_xs[j], self.Y_TOK, 0.0])
                )
            else:
                t = (
                    Tex(rf"\textit{{``{tok}''}}")
                    .scale(0.55)
                    .move_to([tgt_xs[j], self.Y_TOK, 0.0])
                )
            tgt_tok_mobs.append(t)

        tgt_side_lbl = (
            MathTex(r"\text{FR}")
            .scale(0.55)
            .move_to([self.X_DEC + 2.20, self.Y_TOK, 0.0])
            .set_color(GREY_B)
        )

        self.play(
            *[FadeIn(t) for t in tgt_tok_mobs],
            FadeIn(tgt_side_lbl),
            run_time=0.4,
        )

        # ===================== Target embeddings + PE =====================
        tgt_emb_cols = self._build_emb_row(self.X_DEC, len(self.TGT_TOKENS), BLUE)
        tgt_pe_lbl = (
            MathTex(r"+\,PE_i")
            .scale(0.50)
            .next_to(tgt_emb_cols[-1], RIGHT, buff=0.12)
        )
        tgt_tok_to_emb: list[Arrow] = []
        for j in range(len(self.TGT_TOKENS)):
            arr = _vertical_arrow(
                tgt_tok_mobs[j], tgt_emb_cols[j],
                buff=0.12, tip_length=0.10, color=BLUE, stroke_width=2.0,
            )
            tgt_tok_to_emb.append(arr)
        self.play(
            *[FadeIn(c) for c in tgt_emb_cols],
            *[Create(a) for a in tgt_tok_to_emb],
            FadeIn(tgt_pe_lbl),
            run_time=0.45,
        )

        # ===================== Decoder block + sublayers =====================
        dec_outer, causal_sub, cross_sub = self._build_dec_block()
        arr_temb_to_dec = _vertical_arrow(
            tgt_emb_cols[1], dec_outer,
            buff=0.10, tip_length=0.12, color=GREY_B, stroke_width=2.5,
        )
        self.play(FadeIn(dec_outer), Create(arr_temb_to_dec), run_time=0.45)
        self.play(FadeIn(causal_sub), run_time=0.25)
        self.play(FadeIn(cross_sub), run_time=0.25)
        self.wait(0.15)

        # ===================== Cross-attention bridge =====================
        # Clean horizontal arrows from K and V stacks to the cross-attn sublayer.
        # Route at the Y level of cross_sub for maximum clarity.
        cross_y = float(cross_sub.get_center()[1])
        dec_left_x = float(dec_outer.get_left()[0]) - 0.05

        # K bridge: horizontal arrow from K stack right edge to decoder left
        k_right_x = float(k_cols[-1].get_right()[0]) + 0.05
        k_bridge = Arrow(
            start=[k_right_x, cross_y - 0.12, 0.0],
            end=[dec_left_x, cross_y - 0.12, 0.0],
            buff=0.0, stroke_width=2.5, tip_length=0.12, color=ORANGE,
        )
        k_bridge_lbl = (
            MathTex(r"K").scale(0.45)
            .move_to([(k_right_x + dec_left_x) / 2, cross_y - 0.12 + 0.22, 0.0])
            .set_color(ORANGE)
        )

        # V bridge: horizontal arrow from V stack right edge to decoder left
        v_right_x = float(v_cols[-1].get_right()[0]) + 0.05
        v_bridge = Arrow(
            start=[v_right_x, cross_y + 0.12, 0.0],
            end=[dec_left_x, cross_y + 0.12, 0.0],
            buff=0.0, stroke_width=2.5, tip_length=0.12, color=GREEN,
        )
        v_bridge_lbl = (
            MathTex(r"V").scale(0.45)
            .move_to([(v_right_x + dec_left_x) / 2, cross_y + 0.12 + 0.22, 0.0])
            .set_color(GREEN)
        )

        self.play(
            Create(k_bridge), Create(v_bridge),
            FadeIn(k_bridge_lbl), FadeIn(v_bridge_lbl),
            run_time=0.7,
        )
        self.wait(0.2)

        # ===================== Decoder hidden d_t =====================
        d_t = TensorColumn(
            dim=self.TENSOR_DIM, cell_size=self.CELL,
            color=BLUE, fill_opacity=0.55,
        ).move_to([self.X_DEC, self.Y_DT, 0.0])
        d_t_lbl = (
            MathTex(r"d_t").scale(0.58)
            .next_to(d_t, LEFT, buff=0.10)
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

        # ===================== Vocab + softmax + prediction =====================
        vocab_box = LabeledBox(
            label=r"W_{\text{vocab}}",
            width=self.VOCAB_BOX_W, height=self.VOCAB_BOX_H,
            color=GREY_B, label_scale=0.48, fill_opacity=0.12,
        ).move_to([self.X_DEC, self.Y_VOCAB, 0.0])

        arr_dt_to_vbox = _vertical_arrow(
            d_t, vocab_box,
            buff=0.08, tip_length=0.12, color=GREY_B, stroke_width=2.0,
        )

        softmax_tensor = TensorColumn(
            dim=5, cell_size=0.20,
            color=YELLOW, fill_opacity=0.25,
            highlight_index=2, highlight_color="#F5C518",
            highlight_opacity=0.90,
        ).move_to([self.X_DEC + 1.60, self.Y_VOCAB, 0.0])
        softmax_lbl = (
            MathTex(r"\mathrm{softmax}")
            .scale(0.48)
            .next_to(softmax_tensor, UP, buff=0.08)
        )

        arr_vbox_to_softmax = _horizontal_arrow(
            vocab_box, softmax_tensor,
            buff=0.08, tip_length=0.12, color=YELLOW, stroke_width=2.0,
        )

        pred_tok = (
            Tex(rf"\textit{{``{self.PRED_TOKEN}''}}")
            .scale(0.58)
            .move_to([self.X_DEC + 1.60, self.Y_PRED, 0.0])
            .set_color(TEAL)
        )
        arr_softmax_to_pred = _vertical_arrow(
            softmax_tensor, pred_tok,
            buff=0.08, tip_length=0.12, color=TEAL, stroke_width=2.0,
        )

        self.play(
            FadeIn(vocab_box), Create(arr_dt_to_vbox),
            run_time=0.40,
        )
        self.play(
            FadeIn(softmax_tensor), FadeIn(softmax_lbl),
            Create(arr_vbox_to_softmax),
            run_time=0.45,
        )
        self.play(
            FadeIn(pred_tok), Create(arr_softmax_to_pred),
            run_time=0.40,
        )
        self.wait(0.4)

        # ===================== Caption =====================
        caption = (
            MathTex(
                r"\text{Encoder: parallel over source.\ "
                r"Decoder: sequential over target.}"
            )
            .scale(0.55)
            .move_to([0.0, self.Y_CAPTION, 0.0])
        )
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(1.0)
