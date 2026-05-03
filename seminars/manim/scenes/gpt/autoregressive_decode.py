"""GPT autoregressive decode + KV-cache scene for seminar 09 (V13).

Redesigned layout with clear vertical flow:
- Token ribbon at top
- Current token embedding feeds DOWN into SelfAttn block
- Q branches left, attends to K cache (left side)
- h_t emerges right, feeds into vocab projection
- V cache shown alongside K cache
- KV cache grows visibly each step

Color convention (compatible with V06):
- TEAL — Q (query of current token).
- ORANGE — K cache.
- GREEN — V cache.
- BLUE — embedding + hidden state h_t.
- YELLOW — softmax output; argmax cell highlighted.
"""
from __future__ import annotations

from typing import Any

from manim import (
    Arrow,
    BLUE,
    Create,
    DOWN,
    DashedLine,
    FadeIn,
    FadeOut,
    GREEN,
    GREY_B,
    LEFT,
    MathTex,
    ORANGE,
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
    """Arrow always attaching to right/left edges."""
    defaults: dict[str, Any] = {"buff": 0.08, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


def _vertical_arrow(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    """Arrow always attaching to top/bottom edges."""
    defaults: dict[str, Any] = {"buff": 0.06, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[1] >= b.get_center()[1]:
        start, end = a.get_bottom(), b.get_top()
    else:
        start, end = a.get_top(), b.get_bottom()
    return Arrow(start=start, end=end, **defaults)


class AutoregressiveDecode(Scene):
    """V13: GPT greedy decoding with KV-cache (3 generated tokens)."""

    TENSOR_DIM = 4
    KV_CELL = 0.22
    QH_CELL = 0.26
    EMB_CELL = 0.24
    VOCAB_DIM = 5
    VOCAB_CELL = 0.22

    # --- Vertical layout (top-to-bottom for this scene since it's
    # sequential generation — each step feeds forward) ---
    Y_TITLE = 3.65
    Y_TOKENS = 3.00           # ribbon of generated tokens
    Y_EMB = 2.10              # embedding of current last token
    Y_BLOCK = 0.60            # SelfAttn box center
    Y_Q = 0.60                # Q tensor (same row as block, left of it)
    Y_HT = -0.80              # h_t output below block
    Y_KV = -0.80              # KV cache row (same height as h_t, flanking)
    Y_VOCAB = -2.20           # vocab projection row
    Y_CAPTION = -3.65

    # --- Horizontal layout ---
    X_BLOCK = 0.0             # SelfAttn block center
    X_EMB = 0.0               # embedding centered
    X_Q = -2.50               # Q tensor left of block
    X_HT = 0.0                # h_t centered under block

    # KV cache: K and V side-by-side, left of center
    X_KCACHE_BASE = -6.20     # leftmost K column
    X_VCACHE_BASE = -3.70     # leftmost V column (right of K)
    KV_COL_SPACING = 0.36

    # Vocab projection chain (horizontal, right of h_t)
    X_VOCAB_BOX = 2.40
    X_VOCAB_TENSOR = 4.50
    X_PRED_TOKEN = 6.10

    # Token ribbon
    X_TOKEN_BASE = -4.50
    TOKEN_SPACING = 1.30

    # Dims
    SELFATTN_W = 1.80
    SELFATTN_H = 0.90
    VOCAB_BOX_W = 1.10
    VOCAB_BOX_H = 0.55

    # Generation config
    PROMPT_TOKENS = ["the", "cat"]
    GEN_TOKENS = ["sat", "on", "the"]
    ARGMAX_INDICES = [2, 0, 4]

    # ----------------------------- helpers -----------------------------
    def _kv_column_x(self, base_x: float, idx: int) -> float:
        return base_x + idx * self.KV_COL_SPACING

    def _make_cache_column(
        self, base_x: float, idx: int, color: str,
        fill_opacity: float = 0.55,
    ) -> TensorColumn:
        return TensorColumn(
            dim=self.TENSOR_DIM,
            cell_size=self.KV_CELL,
            color=color,
            fill_opacity=fill_opacity,
        ).move_to([self._kv_column_x(base_x, idx), self.Y_KV, 0.0])

    def _make_vocab_tensor(self, argmax_idx: int) -> TensorColumn:
        return TensorColumn(
            dim=self.VOCAB_DIM,
            cell_size=self.VOCAB_CELL,
            color=YELLOW,
            fill_opacity=0.20,
            highlight_index=argmax_idx,
            highlight_color="#F5C518",
            highlight_opacity=0.90,
        ).move_to([self.X_VOCAB_TENSOR, self.Y_VOCAB, 0.0])

    # ----------------------------- main -----------------------------
    def construct(self) -> None:
        # ===================== Title =====================
        title = (
            MathTex(r"\text{Autoregressive decode: greedy + KV cache}")
            .scale(0.58)
            .move_to([0.0, self.Y_TITLE, 0.0])
        )
        self.play(Write(title), run_time=0.6)
        self.wait(0.15)

        # ===================== Prompt tokens =====================
        ribbon: list[Tex] = []
        for j, tok in enumerate(self.PROMPT_TOKENS):
            tx = self.X_TOKEN_BASE + j * self.TOKEN_SPACING
            t = (
                Tex(rf"\textit{{``{tok}''}}")
                .scale(0.55)
                .move_to([tx, self.Y_TOKENS, 0.0])
            )
            ribbon.append(t)
        self.play(*[FadeIn(t) for t in ribbon], run_time=0.45)
        self.wait(0.15)

        # ===================== SelfAttn block (persistent) =====================
        selfattn = LabeledBox(
            label=r"\mathrm{Self\text{-}Attn}",
            width=self.SELFATTN_W,
            height=self.SELFATTN_H,
            color=BLUE,
            label_scale=0.58,
            fill_opacity=0.12,
        ).move_to([self.X_BLOCK, self.Y_BLOCK, 0.0])

        # Cache labels
        k_label = (
            MathTex(r"K\text{ cache}")
            .scale(0.55)
            .move_to([self.X_KCACHE_BASE + 0.50, self.Y_KV + 0.85, 0.0])
            .set_color(ORANGE)
        )
        v_label = (
            MathTex(r"V\text{ cache}")
            .scale(0.55)
            .move_to([self.X_VCACHE_BASE + 0.50, self.Y_KV + 0.85, 0.0])
            .set_color(GREEN)
        )

        self.play(
            FadeIn(selfattn), FadeIn(k_label), FadeIn(v_label),
            run_time=0.45,
        )
        self.wait(0.15)

        # ===================== Initial KV cache (from prompt) =====================
        k_cache: list[TensorColumn] = []
        v_cache: list[TensorColumn] = []
        for idx in range(len(self.PROMPT_TOKENS)):
            k_cache.append(self._make_cache_column(self.X_KCACHE_BASE, idx, ORANGE))
            v_cache.append(self._make_cache_column(self.X_VCACHE_BASE, idx, GREEN))
        self.play(
            *[FadeIn(c) for c in k_cache + v_cache],
            run_time=0.5,
        )
        self.wait(0.25)

        # ===================== Decode steps =====================
        for step_idx, gen_token in enumerate(self.GEN_TOKENS):
            t_now = len(self.PROMPT_TOKENS) + step_idx
            last_tok_text = (
                self.PROMPT_TOKENS[-1] if step_idx == 0
                else self.GEN_TOKENS[step_idx - 1]
            )

            # Step label
            step_lbl = (
                MathTex(rf"\text{{step }}t = {t_now}")
                .scale(0.55)
                .move_to([5.50, self.Y_TITLE - 0.50, 0.0])
                .set_color(GREY_B)
            )

            # --- Embedding (above SelfAttn, centered) ---
            emb = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.EMB_CELL,
                color=BLUE,
                fill_opacity=0.40,
            ).move_to([self.X_EMB, self.Y_EMB, 0.0])
            emb_lbl = (
                MathTex(rf"e(\text{{{last_tok_text}}})")
                .scale(0.55)
                .next_to(emb, RIGHT, buff=0.12)
            )

            # Arrow: emb → SelfAttn (vertical, top-to-bottom)
            arr_emb_to_sa = _vertical_arrow(
                emb, selfattn, buff=0.08, tip_length=0.13, color=BLUE,
                stroke_width=2.5,
            )

            # --- Q tensor (left of SelfAttn, same row) ---
            q_col = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.QH_CELL,
                color=TEAL,
                fill_opacity=0.55,
            ).move_to([self.X_Q, self.Y_Q, 0.0])
            q_lbl = (
                MathTex(r"q_t")
                .scale(0.55)
                .next_to(q_col, UP, buff=0.08)
            )

            # --- h_t (below SelfAttn, centered) ---
            h_col = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.QH_CELL,
                color=BLUE,
                fill_opacity=0.55,
            ).move_to([self.X_HT, self.Y_HT, 0.0])
            h_lbl = (
                MathTex(rf"h_{{{t_now}}}")
                .scale(0.58)
                .next_to(h_col, LEFT, buff=0.10)
            )
            arr_sa_to_h = _vertical_arrow(
                selfattn, h_col, buff=0.08, tip_length=0.13, color=BLUE,
                stroke_width=2.5,
            )

            # --- Attend arrows: Q → each K column (fan-out pattern) ---
            attend_arrows: list[Arrow] = []
            for k_col_cache in k_cache:
                a = arrow_between(
                    q_col, k_col_cache,
                    buff=0.06, tip_length=0.10, color=ORANGE,
                    stroke_width=1.8,
                )
                attend_arrows.append(a)

            # --- Vocab projection ---
            vocab_box = LabeledBox(
                label=r"W_{\text{vocab}}",
                width=self.VOCAB_BOX_W,
                height=self.VOCAB_BOX_H,
                color=GREY_B,
                label_scale=0.52,
                fill_opacity=0.10,
            ).move_to([self.X_VOCAB_BOX, self.Y_VOCAB, 0.0])

            vocab_tensor = self._make_vocab_tensor(self.ARGMAX_INDICES[step_idx])
            vocab_lbl = (
                MathTex(r"\mathrm{softmax}")
                .scale(0.50)
                .next_to(vocab_tensor, UP, buff=0.08)
            )

            arr_h_to_vbox = _vertical_arrow(
                h_col, vocab_box,
                buff=0.10, tip_length=0.12, color=GREY_B,
                stroke_width=2.0,
            )
            arr_vbox_to_vtensor = _horizontal_arrow(
                vocab_box, vocab_tensor,
                buff=0.08, tip_length=0.12, color=YELLOW,
                stroke_width=2.0,
            )

            # Predicted token
            pred_tok = (
                Tex(rf"\textit{{``{gen_token}''}}")
                .scale(0.58)
                .move_to([self.X_PRED_TOKEN, self.Y_VOCAB, 0.0])
                .set_color(TEAL)
            )
            arr_vtensor_to_pred = _horizontal_arrow(
                vocab_tensor, pred_tok,
                buff=0.10, tip_length=0.12, color=TEAL,
                stroke_width=2.0,
            )

            # --- Animate ---
            self.play(
                FadeIn(step_lbl),
                FadeIn(emb), FadeIn(emb_lbl),
                Create(arr_emb_to_sa),
                run_time=0.40,
            )
            self.play(
                FadeIn(q_col), FadeIn(q_lbl),
                run_time=0.25,
            )
            self.play(
                *[Create(a) for a in attend_arrows],
                run_time=0.45,
            )
            self.play(
                FadeIn(h_col), FadeIn(h_lbl),
                Create(arr_sa_to_h),
                run_time=0.35,
            )
            self.play(
                FadeIn(vocab_box),
                Create(arr_h_to_vbox),
                run_time=0.30,
            )
            self.play(
                FadeIn(vocab_tensor), FadeIn(vocab_lbl),
                Create(arr_vbox_to_vtensor),
                run_time=0.35,
            )
            self.play(
                FadeIn(pred_tok),
                Create(arr_vtensor_to_pred),
                run_time=0.30,
            )
            self.wait(0.3)

            # --- Append predicted token to ribbon ---
            new_tok_x = self.X_TOKEN_BASE + t_now * self.TOKEN_SPACING
            new_tok = (
                Tex(rf"\textit{{``{gen_token}''}}")
                .scale(0.55)
                .move_to([new_tok_x, self.Y_TOKENS, 0.0])
                .set_color(TEAL)
            )
            self.play(FadeIn(new_tok), run_time=0.25)
            ribbon.append(new_tok)

            # --- Grow KV cache ---
            new_k = self._make_cache_column(
                self.X_KCACHE_BASE, len(k_cache), ORANGE, fill_opacity=0.70
            )
            new_v = self._make_cache_column(
                self.X_VCACHE_BASE, len(v_cache), GREEN, fill_opacity=0.70
            )
            self.play(FadeIn(new_k), FadeIn(new_v), run_time=0.35)
            k_cache.append(new_k)
            v_cache.append(new_v)
            self.play(
                new_k.animate.set_fill(ORANGE, opacity=0.55),
                new_v.animate.set_fill(GREEN, opacity=0.55),
                run_time=0.20,
            )

            # --- Cleanup transient objects ---
            cleanup = [
                FadeOut(step_lbl),
                FadeOut(emb), FadeOut(emb_lbl),
                FadeOut(arr_emb_to_sa),
                FadeOut(q_col), FadeOut(q_lbl),
                *[FadeOut(a) for a in attend_arrows],
                FadeOut(h_col), FadeOut(h_lbl),
                FadeOut(arr_sa_to_h),
                FadeOut(vocab_box), FadeOut(arr_h_to_vbox),
                FadeOut(vocab_tensor), FadeOut(vocab_lbl),
                FadeOut(arr_vbox_to_vtensor),
                FadeOut(pred_tok), FadeOut(arr_vtensor_to_pred),
                new_tok.animate.set_color(WHITE),
            ]
            self.play(*cleanup, run_time=0.40)
            for m in (
                step_lbl, emb, emb_lbl, arr_emb_to_sa,
                q_col, q_lbl,
                h_col, h_lbl, arr_sa_to_h,
                vocab_box, arr_h_to_vbox,
                vocab_tensor, vocab_lbl, arr_vbox_to_vtensor,
                pred_tok, arr_vtensor_to_pred,
                *attend_arrows,
            ):
                self.remove(m)

        self.wait(0.4)

        # ===================== Caption =====================
        caption = (
            MathTex(
                r"\text{KV cache grows by 1 column per step;\ "
                r"Q computed only for the new token}"
            )
            .scale(0.55)
            .move_to([0.0, self.Y_CAPTION, 0.0])
        )
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(1.0)
