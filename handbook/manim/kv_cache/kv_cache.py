from __future__ import annotations
import copy
from collections import defaultdict
from manim_imports_ext import *
from _2024.transformers.helpers import *
from _2024.transformers.embedding import break_into_words
from _2024.transformers.embedding import break_into_tokens
from _2024.transformers.embedding import get_piece_rectangles


# Что такое KV Cache?

# Мы знаем, что трансформер - это модель авторегрессивной генерации.
# Это значит, что:
#   - За один шаг мы генерируем обычно один токен
#   - Каждый новый токен зависит от всех предыдущих
# Формула P(x_t | x_1, x_2, ..., x_{t-1})

# Постановка проблемы:
# Визуализация квадратичной сложности Attention:
#   Каждый токен считает похожести Key, Query с каждым другим токеном
#   Поэтому количество вычислений растет квадратично

# Это значит, что чем больше длинна последовательности, тем больше вычислений

# Решение:
# Стандартная практика сэкономить вычисления - это закэшировать результат.

# Блиц:
# Сложность вычислений без KV-Cache: O(n^2). Какая сложность вычислений с KV-Cache?
#     - O(n)
# Можно ли использовать KV-Cache во время обучения?
#     - На практике KV-Cache используется только во время инференса
# Сколько памяти занимает KV-Cache?
#     - TODO посчитать график для 3B, 8B модели

class CommonFixture:

    def create_word_rect(self, word_mob, text_height=None, text_y=None):

        rect = SurroundingRectangle(word_mob)

        if text_height is None:
            text_height = word_mob.get_height()

        if text_y is None:
            text_y = word_mob.get_y()

        rect.set_height(text_height + SMALL_BUFF, stretch=True)
        rect.set_width(word_mob.get_width() + SMALL_BUFF, stretch=True)
        rect.set_y(text_y)
        rect.set_stroke(GREY, 1)
        rect.set_fill(GREY, 0.25)
        return rect

    def generation_step_mob(self, generation_step):
        return Text(
            f"Generation step: {generation_step}",
            font_size=30,
            alignment='LEFT',
            fill_color=GREEN_A,
        )


    def render_attention_sequence(self, *,
                                  header_text,
                                  header_color,
                                  sentence_text,
                                  prefix_words,
                                  use_kv_cache):
        """Shared rendering for the attention computations scenes.

        Parameters
        ----------
        header_text : str
            Title to show in the upper-left corner.
        header_color : Manim color
            Color for the title.
        sentence_text : str
            The sentence to display and iterate over.
        prefix_words : int
            Number of initial words visible at start.
        use_kv_cache : bool
            If True, only draw arrows from past tokens to current (O(n));
            if False, draw all pairwise arrows up to current (O(n^2)).
        """

        # Header
        header = Text(
            header_text,
            font_size=30,
            alignment='LEFT',
            fill_color=header_color,
        )
        header.to_corner(LEFT + UP).fix_in_frame()

        # Sentence
        text = sentence_text
        text_mob = Text(text, font_size=30, fill_color=BLUE_A)
        text_mob.align_to(header, DOWN + LEFT)
        text_mob.shift(text_mob.get_height() * 12 * DOWN)

        display_characters = sum(len(word) for word in text.split(" ")[:prefix_words])

        full_text_height = text_mob.get_height()
        full_text_y = text_mob.get_y()

        # Split words and precreate rects for prefix
        processed_letters = 0
        words = text.split(" ")
        words_mobs = []
        all_rects = []
        for word_i, word in enumerate(words):
            word_len = len(word)
            word_mob = text_mob[processed_letters:processed_letters + word_len]
            words_mobs.append(word_mob)
            processed_letters += word_len

            if word_i < prefix_words:
                rect = self.create_word_rect(word_mob, text_height=full_text_height, text_y=full_text_y)
                all_rects.append(rect)

        # Initial draws
        self.add(text_mob[:display_characters])
        self.add(header)
        self.wait()

        # Iterate words and draw arrows
        generation_step = 1
        arrow_counts = defaultdict(int)
        for word_i in range(prefix_words, len(words_mobs)):
            step_mob = self.generation_step_mob(generation_step)
            step_mob.align_to(header, DOWN + LEFT)
            step_mob.shift(step_mob.get_height() * 3 * DOWN)

            word_mob = words_mobs[word_i]
            rect = self.create_word_rect(word_mob, text_height=full_text_height, text_y=full_text_y)

            arrows = []
            if use_kv_cache:
                # Only from past to current
                word_to_arrow = word_i
                for word_from_arrow in range(word_to_arrow):
                    key = (word_from_arrow, word_to_arrow)
                    current_count = arrow_counts[key] + 1
                    t = min(max((current_count - 1) / 4.0, 0.0), 1.0)
                    color = interpolate_color(BLUE_B, RED_E, t)
                    arrow = Arrow(
                        words_mobs[word_from_arrow].get_top(), words_mobs[word_to_arrow].get_top(),
                        path_arc=-150 * DEGREES, buff=0.1, stroke_color=color,
                        thickness=1.5,
                        fill_opacity=0.9,
                        fill_color=color,
                    )
                    arrows.append(arrow)
                    arrow_counts[key] = current_count
            else:
                # All pairwise up to current word
                for word_to_arrow in range(word_i + 1):
                    for word_from_arrow in range(word_to_arrow):
                        key = (word_from_arrow, word_to_arrow)
                        current_count = arrow_counts[key] + 1
                        t = min(max((current_count - 1) / 4.0, 0.0), 1.0)
                        color = interpolate_color(BLUE_B, RED_E, t)
                        arrow = Arrow(
                            words_mobs[word_from_arrow].get_top(), words_mobs[word_to_arrow].get_top(),
                            path_arc=-150 * DEGREES, buff=0.1, stroke_color=color,
                            thickness=1.5,
                            fill_opacity=0.9,
                            fill_color=color,
                        )
                        arrows.append(arrow)
                        arrow_counts[key] = current_count

            self.add(rect, step_mob)
            self.play(
                *[GrowArrow(arrow) for arrow in arrows],
                Write(word_mob, stroke_color=BLUE_B)
            )
            self.wait()

            step_mob.clear()
            generation_step += 1

        self.wait()


class TransformerAutoregressiveGeneration(InteractiveScene, CommonFixture):

    machine_name = "Transformer"
    machine_phi = 10 * DEGREES
    machine_theta = 12 * DEGREES

    def get_transformer_drawing(self):
        self.camera.light_source.move_to([-5, 5, 10])
        self.frame.set_field_of_view(20 * DEGREES)
        blocks = VGroup(
            VPrism(3, 2, 0.2)
            for n in range(4)
        )
        blocks.set_fill(PURPLE_E, 1)
        blocks.set_stroke(width=0)
        blocks.set_shading(0.25, 0.5, 0.2)
        blocks.arrange(OUT)
        blocks.move_to(ORIGIN, OUT)
        blocks.rotate(self.machine_phi, RIGHT, about_edge=OUT)
        blocks.rotate(self.machine_theta, UP, about_edge=OUT)

        blocks.deactivate_depth_test()
        for block in blocks:
            block.sort(lambda p: p[2])

        word = Text(self.machine_name, alignment="LEFT")
        word.next_to(blocks[-1], UP)
        word.shift(0.1 * UP + 0.4 * LEFT)
        word.move_to(blocks[-1])
        word.set_backstroke(BLACK, 5)
        out_arrow = Vector(
            0.5 * RIGHT, stroke_width=10,
            max_tip_length_to_length_ratio=0.5,
            max_width_to_length_ratio=12
        )
        out_arrow.next_to(blocks[-1], RIGHT, buff=SMALL_BUFF)
        out_arrow.set_opacity(0)

        result = VGroup(blocks, word, out_arrow)
        return result


    def construct(self):
        # Add sentence
        recap_mob = Text(
            "Autoregressive generation.",
            font_size=30,
            alignment='LEFT',
        )
        recap_mob.to_corner(LEFT + UP).fix_in_frame()

        machine = self.get_transformer_drawing()
        machine.center()
        machine.shift(RIGHT * 0.2)

        text = "The cat sat on the mat."
        text_mob = Text(text, font_size=30, fill_color=BLUE_A)
        text_mob.align_to(machine, DOWN)
        text_mob.shift(text_mob.get_height() * 3 * DOWN)

        generated_text = "The cat sat on the mat."
        generated_text_mob = Text(text, font_size=30, fill_color=BLUE_A)
        generated_text_mob.align_to(machine, UP)
        generated_text_mob.shift(generated_text_mob.get_height() * 3 * UP)


        prefix_words = 1
        display_characters = sum(len(word) for word in text.split(" ")[:prefix_words])

        full_text_height = text_mob.get_height()
        full_text_y = text_mob.get_y()

        # Create word rects
        processed_letters = 0
        words = text.split(" ")
        all_rects = []
        words_mobs = []
        generated_words_mobs = []
        for word_i, word in enumerate(words):
            word_len = len(word)
            word_mob = text_mob[processed_letters:processed_letters + word_len]
            words_mobs.append(word_mob)

            generated_words_mobs.append(generated_text_mob[processed_letters:processed_letters + word_len])

            processed_letters += word_len


            if word_i >= prefix_words:
                continue

            rect = self.create_word_rect(word_mob, text_height=full_text_height, text_y=full_text_y)
            all_rects.append(rect)


        self.add(machine)
        self.add(VGroup(*all_rects[:prefix_words]))
        self.add(text_mob[:display_characters])
        self.add(recap_mob)

        generation_step = 1

        prefix_words_mobs = copy.deepcopy(words_mobs[:prefix_words])


        for word_i in range(prefix_words, len(words_mobs)):
            generation_step_mob = self.generation_step_mob(generation_step)
            generation_step_mob.align_to(recap_mob, DOWN + LEFT)
            generation_step_mob.shift(generation_step_mob.get_height() * 2 * DOWN)

            word_mob = words_mobs[word_i]
            rect = self.create_word_rect(word_mob, text_height=full_text_height, text_y=full_text_y)

            generated_word = generated_words_mobs[word_i]

            self.add(rect, generation_step_mob)

            prefix_words_mobs_group = VGroup(*copy.deepcopy(prefix_words_mobs))

            blocks = machine[0]
            self.play(
                FlashAround(generation_step_mob[str(generation_step)], run_time=0.5),
                Transform(prefix_words_mobs_group, generated_word, run_time=0.5),
                LaggedStart(
                    (
                        block.animate.set_color(
                            block.get_color() if block is blocks[-1] else TEAL
                        ).set_anim_args(rate_func=there_and_back)
                        for block in blocks
                    ),
                    lag_ratio=0.1,
                    run_time=0.5,
                ),
                Animation(machine[1:]),
            )
            self.wait(0.1)

            self.play(Transform(generated_word, word_mob, run_time=0.5))

            self.add(word_mob)
            generation_step_mob.clear()
            generation_step += 1

            prefix_words_mobs.append(word_mob)

            generated_word.clear()
            prefix_words_mobs_group.clear()

        self.wait()


class SequenceAttentionComputationsNoKVCache(InteractiveScene, CommonFixture):

    def construct(self):
        self.render_attention_sequence(
            header_text="No KV Cache",
            header_color=RED_E,
            sentence_text="The cat sat on the mat.",
            prefix_words=1,
            use_kv_cache=False,
        )


class SequenceAttentionComputationsWithKVCache(InteractiveScene, CommonFixture):

    def construct(self):
        self.render_attention_sequence(
            header_text="With KV Cache",
            header_color=GREEN_E,
            sentence_text="The cat sat on the mat.",
            prefix_words=1,
            use_kv_cache=True,
        )
