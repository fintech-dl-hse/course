"""Seq2seq bottleneck scene for seminar 09 (V04 of the curriculum catalog).

Мотивирует attention: показывает, почему один фиксированный вектор
``h_T`` энкодера не способен переносить достаточно информации о всём
исходном предложении в декодер.

1. Сверху — формула ``\\text{seq2seq: encoder} \\to h_T \\to \\text{decoder}``
   и подзаголовок-проблема.
2. На верхней половине кадра — лента из 5 source-токенов
   (``the / quick / brown / fox / jumps``).
3. Энкодер: для каждого токена строится скрытое состояние ``h_t``
   (TensorColumn, dim=4) через recurrence ``h_{t-1} \\to h_t`` с
   зелёной ``W_{hh}``-коробкой.
4. Визуальный пуант: после прохода всех 5 шагов состояния
   ``h_1..h_4`` затухают (низкая прозрачность), яркой остаётся только
   ``h_5`` — единственное, что передаётся в декодер.
5. Узкая (тонкая) стрелка от ``h_5`` к контекст-вектору в центре
   кадра, противопоставленная толстым стрелкам recurrence — визуальная
   метафора bottleneck'а.
6. Декодер: 4 target-шага справа, каждый рисует свой выход и тянет
   пунктирную линию обратно к одному и тому же ``h_5`` — отсюда видно,
   что весь target опирается на одну компрессию.
7. Снизу — итоговая подпись: ``bottleneck: |h_T| фиксирован, длинные
   последовательности теряют информацию``.

Сцена использует общие примитивы ``shared.neural`` (TensorColumn,
LabeledBox, arrow_between) и стилистически совместима с ``RNNForward``
и ``EmbeddingLookup``: те же ячейки TensorColumn (``cell_size=0.27``),
тот же подход «токены сверху → тензорный поток ниже».
"""
from __future__ import annotations

from typing import Any

from manim import (
    Arrow,
    BLUE,
    Create,
    DashedLine,
    DOWN,
    FadeIn,
    GREEN,
    LEFT,
    MathTex,
    RIGHT,
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
    """Стрелка, всегда прилипающая к правому/левому краю объектов."""
    defaults: dict[str, Any] = {"buff": 0.08, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    if a.get_center()[0] <= b.get_center()[0]:
        start, end = a.get_right(), b.get_left()
    else:
        start, end = a.get_left(), b.get_right()
    return Arrow(start=start, end=end, **defaults)


class Seq2SeqBottleneck(Scene):
    """V04: source → encoder RNN → h_T → decoder; визуализация bottleneck'а."""

    # ---- Layout constants — tuned for 720p (-qm) frame ±(7.11, 4.0) ----
    CELL_SIZE = 0.27
    TENSOR_DIM = 4

    # Vertical anchors.
    Y_TITLE = 3.55
    Y_SRC_TOKEN = 2.40       # source token strip
    Y_ENC = 1.05             # encoder hidden row h_1..h_5
    Y_CONTEXT = -0.45        # context vector (the bottleneck output)
    Y_DEC = -1.85            # decoder hidden row s_1..s_4
    Y_TGT_TOKEN = -3.10      # decoder predicted-token strip
    Y_CAPTION = -3.75

    # Encoder horizontal layout: 5 timesteps on the left ~60% of frame.
    # X_ENC_BASE is the x of h_1; h_0 sits one X_ENC_SPACING to the left of
    # it, so X_ENC_BASE must be at least X_ENC_SPACING + cell_size/2 +
    # margin away from the left frame edge (-7.11).
    X_ENC_BASE = -5.30
    X_ENC_SPACING = 1.30

    # Decoder horizontal layout: 4 timesteps on the right.
    X_DEC_BASE = 2.20
    X_DEC_SPACING = 1.40

    # W_hh / W_ss recurrence boxes.
    BOX_W = 0.55
    BOX_H = 0.34

    # Faded opacity for the discarded h_1..h_{T-1} columns.
    FADED_OPACITY = 0.18

    # Source / target tokens.
    SRC_TOKENS = ["the", "quick", "brown", "fox", "jumps"]
    # 4 target tokens (a hypothetical translation/paraphrase).
    TGT_TOKENS = ["der", "schnelle", "fuchs", "springt"]

    def construct(self) -> None:
        # ---------------- Title ----------------
        eq_top = MathTex(
            r"\text{seq2seq: encoder}",
            r"\to",
            r"h_T",
            r"\to",
            r"\text{decoder}",
        ).scale(0.62)
        eq_top[2].set_color(YELLOW)
        eq_bot = MathTex(
            r"\text{problem: only } h_T \text{ carries source info}",
        ).scale(0.55)
        title = (
            VGroup(eq_top, eq_bot)
            .arrange(DOWN, buff=0.14)
            .move_to([0.0, self.Y_TITLE, 0.0])
        )
        self.play(Write(title))
        self.wait(0.2)

        # ---------------- Source token strip ----------------
        src_strip: list[Tex] = []
        for i, tok in enumerate(self.SRC_TOKENS):
            x_pos = self.X_ENC_BASE + i * self.X_ENC_SPACING
            t = (
                Tex(rf"\textit{{``{tok}''}}")
                .scale(0.55)
                .move_to([x_pos, self.Y_SRC_TOKEN, 0.0])
            )
            src_strip.append(t)
        src_group = VGroup(*src_strip)
        self.play(FadeIn(src_group), run_time=0.5)
        self.wait(0.2)

        # ---------------- Initial encoder hidden state h_0 ----------------
        # Sits to the left of h_1 with a small W_hh slot between them so the
        # recurrence chain reads the same way as in RNNForward.
        h0_x = self.X_ENC_BASE - self.X_ENC_SPACING
        h_prev = TensorColumn(
            dim=self.TENSOR_DIM,
            cell_size=self.CELL_SIZE,
            label_scale=0.45,
        ).move_to([h0_x, self.Y_ENC, 0.0])
        h_prev_lbl = (
            MathTex(r"h_0")
            .scale(0.5)
            .next_to(h_prev, DOWN, buff=0.12)
        )
        self.play(FadeIn(h_prev), FadeIn(h_prev_lbl), run_time=0.4)

        # ---------------- Encoder rollout: 5 timesteps ----------------
        h_columns: list[TensorColumn] = []
        h_labels: list[MathTex] = []
        w_hh_boxes: list[LabeledBox] = []
        rec_arrows_in: list[Arrow] = []   # h_{t-1} -> w_hh
        rec_arrows_out: list[Arrow] = []  # w_hh -> h_t
        src_arrows: list[Arrow] = []      # token -> h_t

        for i, tok in enumerate(self.SRC_TOKENS):
            t = i + 1
            x_pos = self.X_ENC_BASE + i * self.X_ENC_SPACING

            # h_t TensorColumn.
            h_t = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL_SIZE,
            ).move_to([x_pos, self.Y_ENC, 0.0])
            h_lbl = (
                MathTex(rf"h_{t}")
                .scale(0.5)
                .next_to(h_t, DOWN, buff=0.10)
            )

            # Recurrence box W_hh between prev_h (right edge) and h_t (left).
            w_hh_x = (h_prev.get_right()[0] + h_t.get_left()[0]) / 2.0
            w_hh = LabeledBox(
                label="W_{hh}",
                width=self.BOX_W,
                height=self.BOX_H,
                label_scale=0.40,
                color=GREEN,
            ).move_to([w_hh_x, self.Y_ENC, 0.0])

            # Two horizontal arrows: prev -> w_hh -> h_t (FAT stroke to
            # contrast with the thin bottleneck arrow later).
            a_in = _horizontal_arrow(
                h_prev, w_hh, buff=0.06, tip_length=0.12, stroke_width=4
            )
            a_out = _horizontal_arrow(
                w_hh, h_t, buff=0.06, tip_length=0.12, stroke_width=4
            )

            # Source token -> h_t (vertical, into the column from above).
            src_to_h = arrow_between(
                src_strip[i], h_t, buff=0.10, tip_length=0.10, stroke_width=3
            )

            self.play(
                FadeIn(w_hh),
                Create(a_in),
                Create(a_out),
                FadeIn(h_t),
                FadeIn(h_lbl),
                Create(src_to_h),
                run_time=0.55,
            )

            h_columns.append(h_t)
            h_labels.append(h_lbl)
            w_hh_boxes.append(w_hh)
            rec_arrows_in.append(a_in)
            rec_arrows_out.append(a_out)
            src_arrows.append(src_to_h)
            h_prev = h_t

        self.wait(0.4)

        # ---------------- Visual punchline: fade h_1..h_4 ----------------
        # Highlight h_5 in YELLOW so the eye lands on the only state that
        # makes it to the decoder.
        fade_anims: list[Any] = []
        last_idx = len(h_columns) - 1
        for i, col in enumerate(h_columns):
            if i == last_idx:
                continue
            for cell in col.cells:
                fade_anims.append(cell.animate.set_fill(BLUE, opacity=0.06))
                fade_anims.append(cell.animate.set_stroke(opacity=0.25))
            fade_anims.append(h_labels[i].animate.set_opacity(0.30))
            fade_anims.append(rec_arrows_in[i].animate.set_opacity(0.25))
            fade_anims.append(rec_arrows_out[i].animate.set_opacity(0.25))
            fade_anims.append(w_hh_boxes[i].animate.set_opacity(0.25))
            fade_anims.append(src_arrows[i].animate.set_opacity(0.25))
            fade_anims.append(src_strip[i].animate.set_opacity(0.35))

        # Highlight h_5 (yellow stroke + brighter fill).
        h_T = h_columns[last_idx]
        for cell in h_T.cells:
            fade_anims.append(cell.animate.set_stroke(YELLOW, width=3))
            fade_anims.append(cell.animate.set_fill(YELLOW, opacity=0.45))
        fade_anims.append(h_labels[last_idx].animate.set_color(YELLOW))

        self.play(*fade_anims, run_time=0.9)
        self.wait(0.5)

        # ---------------- Bottleneck: thin arrow h_5 -> context vector ----------------
        context_vec = TensorColumn(
            dim=self.TENSOR_DIM,
            cell_size=self.CELL_SIZE,
            color=YELLOW,
            fill_opacity=0.45,
        ).move_to([self.X_DEC_BASE - 0.55, self.Y_CONTEXT, 0.0])
        context_lbl = (
            MathTex(r"c = h_T")
            .scale(0.45)
            .next_to(context_vec, DOWN, buff=0.10)
        )

        # Use a deliberately THIN stroke for the bottleneck arrow to
        # visually contrast with the fat recurrence arrows (stroke 4).
        bottleneck = Arrow(
            start=h_T.get_bottom() + 0.02 * DOWN,
            end=context_vec.get_top() + 0.02 * UP,
            buff=0.10,
            tip_length=0.14,
            stroke_width=1.6,
            color=YELLOW,
        )
        bottleneck_lbl = (
            MathTex(r"\text{bottleneck}")
            .scale(0.42)
            .next_to(bottleneck, RIGHT, buff=0.10)
        )

        self.play(
            Create(bottleneck),
            FadeIn(bottleneck_lbl),
            run_time=0.5,
        )
        self.play(
            FadeIn(context_vec),
            FadeIn(context_lbl),
            run_time=0.4,
        )
        self.wait(0.3)

        # ---------------- Decoder: 4 timesteps reading from c = h_T ----------------
        # Decoder hidden row at y=Y_DEC; predicted token strip at y=Y_TGT_TOKEN.
        # Each decoder step gets a faint dotted line back to the context vec
        # — driving home that all 4 outputs draw their only source-side info
        # from the same compressed h_T.
        s_columns: list[TensorColumn] = []
        s_labels: list[MathTex] = []
        s_rec_in: list[Arrow] = []
        s_rec_out: list[Arrow] = []
        w_ss_boxes: list[LabeledBox] = []
        tgt_tokens: list[Tex] = []
        tgt_arrows: list[Arrow] = []
        ctx_dotted: list[DashedLine] = []

        # Decoder s_0 to the left of s_1, mirroring encoder h_0.
        s0_x = self.X_DEC_BASE - self.X_DEC_SPACING * 0.85
        s_prev = TensorColumn(
            dim=self.TENSOR_DIM,
            cell_size=self.CELL_SIZE,
            color=BLUE,
            fill_opacity=0.25,
        ).move_to([s0_x, self.Y_DEC, 0.0])
        s_prev_lbl = (
            MathTex(r"s_0")
            .scale(0.45)
            .next_to(s_prev, DOWN, buff=0.10)
        )
        self.play(FadeIn(s_prev), FadeIn(s_prev_lbl), run_time=0.35)

        for i, tok in enumerate(self.TGT_TOKENS):
            t = i + 1
            x_pos = self.X_DEC_BASE + i * self.X_DEC_SPACING

            s_t = TensorColumn(
                dim=self.TENSOR_DIM,
                cell_size=self.CELL_SIZE,
                color=BLUE,
                fill_opacity=0.30,
            ).move_to([x_pos, self.Y_DEC, 0.0])
            s_lbl = (
                MathTex(rf"s_{t}")
                .scale(0.45)
                .next_to(s_t, DOWN, buff=0.10)
            )

            w_ss_x = (s_prev.get_right()[0] + s_t.get_left()[0]) / 2.0
            w_ss = LabeledBox(
                label="W_{ss}",
                width=self.BOX_W,
                height=self.BOX_H,
                label_scale=0.40,
                color=GREEN,
            ).move_to([w_ss_x, self.Y_DEC, 0.0])

            a_in = _horizontal_arrow(
                s_prev, w_ss, buff=0.06, tip_length=0.12, stroke_width=3
            )
            a_out = _horizontal_arrow(
                w_ss, s_t, buff=0.06, tip_length=0.12, stroke_width=3
            )

            # Predicted target token below s_t.
            tgt = (
                Tex(rf"\textit{{``{tok}''}}")
                .scale(0.58)
                .move_to([x_pos, self.Y_TGT_TOKEN, 0.0])
            )
            s_to_tgt = arrow_between(
                s_t, tgt, buff=0.10, tip_length=0.10, stroke_width=3
            )

            # Faint dotted line from context_vec to s_t — "all of you
            # depend on me alone".
            dot = DashedLine(
                start=context_vec.get_right() + 0.02 * RIGHT,
                end=s_t.get_top() + 0.02 * UP,
                stroke_width=1.5,
                color=YELLOW,
                dash_length=0.10,
            ).set_opacity(0.55)

            self.play(
                FadeIn(w_ss),
                Create(a_in),
                Create(a_out),
                FadeIn(s_t),
                FadeIn(s_lbl),
                Create(dot),
                Create(s_to_tgt),
                FadeIn(tgt),
                run_time=0.55,
            )

            s_columns.append(s_t)
            s_labels.append(s_lbl)
            s_rec_in.append(a_in)
            s_rec_out.append(a_out)
            w_ss_boxes.append(w_ss)
            tgt_tokens.append(tgt)
            tgt_arrows.append(s_to_tgt)
            ctx_dotted.append(dot)
            s_prev = s_t

        self.wait(0.5)

        # ---------------- Final caption ----------------
        caption = (
            MathTex(
                r"\text{bottleneck: } |h_T| \text{ fixed; long sequences lose info}"
            )
            .scale(0.50)
            .move_to([0.0, self.Y_CAPTION, 0.0])
        )
        self.play(FadeIn(caption))
        self.wait(0.8)
