"""GPT autoregressive decode + KV-cache scene for seminar 09 (V13).

Сцена показывает пошаговую авторегрессионную генерацию GPT-модели c
KV-кэшем. На каждом шаге t:

1. Текущий последний токен превращается в эмбеддинг и проходит через
   collapsed-«Self-Attn»-блок (внутренности не раскрываются — V08/V12 их
   уже показали).
2. Для нового токена считается *только один* столбец Q (a не Q для всей
   последовательности, как в обычной forward-pass).
3. Этот Q «обращается» к накопленным K/V из KV-кэша (горизонтальные
   стопки столбцов слева и справа). Кэш растёт на 1 столбец за шаг.
4. Выход блока h_t проектируется в распределение по словарю; argmax
   выбирает следующий токен («sat», «on», «the»).
5. Новый токен дописывается к ленте сверху. Соответствующие новые K и V
   добавляются в кэш справа (видимое расширение стопки).

Цветовая конвенция совместима с V06 (``scaled_dot_product.py``):
- ``TEAL`` — Q (query текущего токена).
- ``ORANGE`` — K-кэш.
- ``GREEN`` — V-кэш.
- ``BLUE`` — embedding текущего входа и hidden state h_t.
- ``YELLOW`` — softmax-выход по словарю; argmax-ячейка подсвечена.

Используются примитивы ``shared.neural`` (TensorColumn, LabeledBox,
arrow_between). Локальный ``_horizontal_arrow`` (как в rnn_forward.py:45-58)
нужен, чтобы стрелки от Q к Self-Attn-боксу не пересекали другие узлы.
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
    """Стрелка, всегда прилипающая к правому/левому краю объектов.

    Локальная копия (см. ``rnn_forward.py:45-58``). Гарантирует горизон-
    тальную привязку даже если |dy| > |dx| (например, когда Q сидит
    выше блока Self-Attn).
    """
    defaults: dict[str, Any] = {"buff": 0.08, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


class AutoregressiveDecode(Scene):
    """V13: GPT greedy decoding with KV-cache (3 generated tokens)."""

    # ---- Layout constants — 720p (-qm) frame ±(7.11, 4.0) ----
    TENSOR_DIM = 4
    KV_CELL = 0.20            # K/V cache cell size (small — many columns fit)
    QH_CELL = 0.24            # Q and h_t tensor cell size
    EMB_CELL = 0.22           # input embedding cell size
    VOCAB_DIM = 5             # softmax tensor over toy vocab
    VOCAB_CELL = 0.24

    # Vertical anchors (top → bottom).
    Y_TITLE = 3.65
    Y_TOKENS = 2.95           # the ribbon of generated/prompt tokens
    Y_EMB = 2.05              # embedding of the current "last" token
    Y_BLOCK = 0.55            # collapsed Self-Attn box, h_t, Q  — middle row
    Y_KV = -1.55              # K and V cache row centers
    Y_VOCAB = -2.85           # softmax-over-vocab tensor + predicted token
    Y_CAPTION = -3.70

    # Horizontal layout.
    # Self-Attn block dead center; Q a bit to its left; h_t to its right.
    X_SELFATTN = 0.0
    X_Q = -2.10
    X_H = 2.10
    X_VOCAB_BOX = 3.85        # vocab projection LabeledBox
    X_VOCAB_TENSOR = 5.40     # softmax tensor over vocab

    # KV cache anchors. Cache columns grow rightward; place K cache on
    # the left half and V cache on the right half so they can each grow
    # without colliding.
    X_KCACHE_BASE = -5.95     # x of column 0 of K cache
    X_VCACHE_BASE = 0.55      # x of column 0 of V cache
    KV_COL_SPACING = 0.40     # horizontal spacing between adjacent cache columns

    # Token ribbon.
    X_TOKEN_BASE = -5.50
    TOKEN_SPACING = 1.40

    # Sublayer box dims.
    SELFATTN_W = 1.55
    SELFATTN_H = 1.10
    VOCAB_BOX_W = 1.10
    VOCAB_BOX_H = 0.70

    # The toy generation: 2-token prompt + 3 generated tokens.
    PROMPT_TOKENS = ["the", "cat"]
    GEN_TOKENS = ["sat", "on", "the"]
    # argmax index in the vocab softmax for each generated step (just for
    # visual distinction — the actual token is the GEN_TOKENS string).
    ARGMAX_INDICES = [2, 0, 4]

    # ----------------------------- helpers -----------------------------
    def _kv_column_x(self, base_x: float, idx: int) -> float:
        """X of the idx-th column in a cache that starts at ``base_x``."""
        return base_x + idx * self.KV_COL_SPACING

    def _make_cache_column(
        self,
        base_x: float,
        idx: int,
        color: str,
        fill_opacity: float = 0.55,
    ) -> TensorColumn:
        """Build one TensorColumn at slot ``idx`` of a cache."""
        return TensorColumn(
            dim=self.TENSOR_DIM,
            cell_size=self.KV_CELL,
            color=color,
            fill_opacity=fill_opacity,
        ).move_to([self._kv_column_x(base_x, idx), self.Y_KV, 0.0])

    def _make_vocab_tensor(
        self,
        argmax_idx: int,
    ) -> TensorColumn:
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
        # ===================== Phase 0: title =====================
        title = (
            MathTex(
                r"\text{Autoregressive decode: greedy + KV cache}"
            )
            .scale(0.55)
            .move_to([0.0, self.Y_TITLE, 0.0])
        )
        self.play(Write(title), run_time=0.6)
        self.wait(0.15)

        # ===================== Phase 1: prompt tokens =====================
        # Place the 2 prompt tokens on the ribbon at the leftmost positions.
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

        # ===================== Phase 2: Self-Attn block + cache labels =====================
        selfattn = LabeledBox(
            label=r"\mathrm{SelfAttn}",
            width=self.SELFATTN_W,
            height=self.SELFATTN_H,
            color=BLUE,
            label_scale=0.45,
            fill_opacity=0.12,
        ).move_to([self.X_SELFATTN, self.Y_BLOCK, 0.0])

        # Cache headings (K / V), placed ABOVE the cache row so cache columns
        # are unobstructed.
        k_label = (
            MathTex(r"K\ \mathrm{cache}")
            .scale(0.50)
            .move_to([self.X_KCACHE_BASE + 0.6, self.Y_KV + 0.85, 0.0])
        )
        v_label = (
            MathTex(r"V\ \mathrm{cache}")
            .scale(0.50)
            .move_to([self.X_VCACHE_BASE + 0.6, self.Y_KV + 0.85, 0.0])
        )

        self.play(
            FadeIn(selfattn), FadeIn(k_label), FadeIn(v_label),
            run_time=0.45,
        )
        self.wait(0.15)

        # ===================== Phase 3: initial cache (2 columns from prompt) =====================
        # After processing the 2-token prompt, K and V each have 2 columns.
        # We don't animate the prompt's forward pass (V08 already does that);
        # we just FadeIn 2 cache columns each side.
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

        # ===================== Phase 4: decode steps 1..3 =====================
        # We re-use the same on-stage Q/h/vocab mobjects across steps: at the
        # end of each step we FadeOut Q, h_t, the vocab tensor + softmax box,
        # the predicted-token ghost text, and the per-step arrows. Cache and
        # ribbon persist (they grow over the scene).
        for step_idx, gen_token in enumerate(self.GEN_TOKENS):
            t_now = len(self.PROMPT_TOKENS) + step_idx  # 0-based "current" position
            # Last input token feeding this step (last token currently on the
            # ribbon — either the last prompt token or the previous gen token).
            last_tok_text = (
                self.PROMPT_TOKENS[-1] if step_idx == 0 else self.GEN_TOKENS[step_idx - 1]
            )

            # --- Step header (a small annotation in the lower-left, optional)
            step_lbl = (
                MathTex(rf"t = {t_now}")
                .scale(0.50)
                .move_to([-6.30, self.Y_VOCAB + 0.30, 0.0])
            )

            # --- Embedding of the last token (input to Self-Attn for this step)
            emb_x = self.X_SELFATTN - 3.20
            emb = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.EMB_CELL,
                color=BLUE,
                fill_opacity=0.40,
            ).move_to([emb_x, self.Y_EMB, 0.0])
            emb_lbl = (
                MathTex(rf"e(\text{{{last_tok_text}}})")
                .scale(0.45)
                .next_to(emb, DOWN, buff=0.12)
            )

            # Arrow: emb → SelfAttn (forced horizontal so it doesn't drop
            # straight down through unrelated mobjects).
            arr_emb_to_sa = _horizontal_arrow(
                emb, selfattn, buff=0.08, tip_length=0.13, color=BLUE,
                stroke_width=2.5,
            )

            # --- Q tensor (single column, computed only for the new token)
            q_col = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.QH_CELL,
                color=TEAL,
                fill_opacity=0.55,
            ).move_to([self.X_Q, self.Y_BLOCK, 0.0])
            q_lbl = (
                MathTex(rf"q_{{{t_now}}}")
                .scale(0.50)
                .next_to(q_col, DOWN, buff=0.12)
            )

            # Arrow: SelfAttn → h_t (h_t = output hidden of the block).
            h_col = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.QH_CELL,
                color=BLUE,
                fill_opacity=0.55,
            ).move_to([self.X_H, self.Y_BLOCK, 0.0])
            h_lbl = (
                MathTex(rf"h_{{{t_now}}}")
                .scale(0.50)
                .next_to(h_col, DOWN, buff=0.12)
            )
            arr_sa_to_h = _horizontal_arrow(
                selfattn, h_col, buff=0.08, tip_length=0.13, color=BLUE,
                stroke_width=2.5,
            )

            # --- "attend" arrows: from q_col to each existing K column.
            #     Use horizontal_arrow for K cache (q sits to the right of K
            #     cache → forced left/right attachment is fine). We render
            #     ALL of them at once. They'll be FadeOut at end of step.
            attend_arrows: list[Arrow] = []
            for k_col in k_cache:
                a = _horizontal_arrow(
                    q_col, k_col,
                    buff=0.06, tip_length=0.10, color=ORANGE,
                    stroke_width=1.8,
                )
                attend_arrows.append(a)

            # --- Vocab projection LabeledBox + softmax tensor + predicted token
            vocab_box = LabeledBox(
                label=r"W_{vocab}",
                width=self.VOCAB_BOX_W,
                height=self.VOCAB_BOX_H,
                color=GREY_B,
                label_scale=0.40,
                fill_opacity=0.10,
            ).move_to([self.X_VOCAB_BOX, self.Y_VOCAB, 0.0])

            vocab_tensor = self._make_vocab_tensor(self.ARGMAX_INDICES[step_idx])
            vocab_lbl = (
                MathTex(r"\mathrm{softmax}(h_t W_{vocab})")
                .scale(0.40)
                .next_to(vocab_tensor, DOWN, buff=0.10)
            )

            # Arrow chain: h_col → vocab_box → vocab_tensor.
            arr_h_to_vbox = _horizontal_arrow(
                h_col, vocab_box,
                buff=0.10, tip_length=0.12, color=GREY_B,
                stroke_width=2.0,
            )
            arr_vbox_to_vtensor = _horizontal_arrow(
                vocab_box, vocab_tensor,
                buff=0.08, tip_length=0.12, color=YELLOW,
                stroke_width=2.0,
            )

            # h_col → vocab path goes through space below the SelfAttn block;
            # vocab box sits at Y_VOCAB. Use a straight horizontal arrow but
            # we need to FIRST move h_col's y down? No — keep h at Y_BLOCK,
            # use horizontal arrow which will choose right/left. Since
            # |dx| < |dy| (X distance ~1.75, Y distance ~3.4), horizontal_arrow
            # would go top/bottom by default; force left/right with our local
            # helper above.

            # --- Animate the step.
            self.play(
                FadeIn(step_lbl),
                FadeIn(emb), FadeIn(emb_lbl),
                Create(arr_emb_to_sa),
                run_time=0.45,
            )
            # Q is computed inside the block; show it as appearing TO THE LEFT
            # of the block (visualization choice — we'd otherwise have to
            # cross the block frame). Then attend to K cache.
            self.play(
                FadeIn(q_col), FadeIn(q_lbl),
                run_time=0.30,
            )
            self.play(
                *[Create(a) for a in attend_arrows],
                run_time=0.50,
            )
            # h_t emerges from the block.
            self.play(
                FadeIn(h_col), FadeIn(h_lbl),
                Create(arr_sa_to_h),
                run_time=0.40,
            )
            # h_t → vocab projection → softmax → argmax.
            self.play(
                FadeIn(vocab_box),
                Create(arr_h_to_vbox),
                run_time=0.30,
            )
            self.play(
                FadeIn(vocab_tensor),
                FadeIn(vocab_lbl),
                Create(arr_vbox_to_vtensor),
                run_time=0.40,
            )

            # --- Append the predicted token to the ribbon.
            new_tok_x = self.X_TOKEN_BASE + t_now * self.TOKEN_SPACING
            new_tok = (
                Tex(rf"\textit{{``{gen_token}''}}")
                .scale(0.55)
                .move_to([new_tok_x, self.Y_TOKENS, 0.0])
                .set_color(TEAL)
            )
            self.play(FadeIn(new_tok), run_time=0.30)
            ribbon.append(new_tok)

            # --- Grow the KV cache: append a new K column and a new V column.
            new_k = self._make_cache_column(
                self.X_KCACHE_BASE, len(k_cache), ORANGE, fill_opacity=0.65
            )
            new_v = self._make_cache_column(
                self.X_VCACHE_BASE, len(v_cache), GREEN, fill_opacity=0.65
            )
            self.play(FadeIn(new_k), FadeIn(new_v), run_time=0.40)
            k_cache.append(new_k)
            v_cache.append(new_v)
            # Recolor the just-added column back to standard opacity after
            # the "highlight" of being newly added.
            self.play(
                new_k.animate.set_fill(ORANGE, opacity=0.55),
                new_v.animate.set_fill(GREEN, opacity=0.55),
                run_time=0.25,
            )

            # --- Cleanup transient mobjects for this step (cache + ribbon
            #     persist; the new TEAL token cools to WHITE).
            cleanup = [
                FadeOut(step_lbl),
                FadeOut(emb), FadeOut(emb_lbl),
                FadeOut(arr_emb_to_sa),
                FadeOut(q_col), FadeOut(q_lbl),
                *[FadeOut(a) for a in attend_arrows],
                FadeOut(h_col), FadeOut(h_lbl),
                FadeOut(arr_sa_to_h),
                FadeOut(vocab_box),
                FadeOut(arr_h_to_vbox),
                FadeOut(vocab_tensor),
                FadeOut(vocab_lbl),
                FadeOut(arr_vbox_to_vtensor),
                new_tok.animate.set_color(WHITE),
            ]
            self.play(*cleanup, run_time=0.45)
            # Manually remove transient mobjects from the scene tree so they
            # don't pile up and confuse later lint passes.
            for m in (
                step_lbl, emb, emb_lbl, arr_emb_to_sa,
                q_col, q_lbl,
                h_col, h_lbl, arr_sa_to_h,
                vocab_box, arr_h_to_vbox,
                vocab_tensor, vocab_lbl, arr_vbox_to_vtensor,
                *attend_arrows,
            ):
                self.remove(m)

        self.wait(0.4)

        # ===================== Phase 5: caption =====================
        caption = (
            MathTex(
                r"\text{KV cache grows by 1 column / step;\ Q computed only "
                r"for the new token}"
            )
            .scale(0.45)
            .move_to([0.0, self.Y_CAPTION, 0.0])
        )
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(1.0)
