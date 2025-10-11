from __future__ import annotations

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
# Можно ли использовать KV-Cache во время обучения?
#     - На практике KV-Cache используется только во время инференса
# Сколько памяти занимает KV-Cache?
#     - TODO посчитать график для 3B, 8B модели
#


class KVCache(InteractiveScene):

    def construct(self):
        # Add sentence
        text = "The cat sat on the mat."
        text_mob = Text(text, font_size=30).move_to(2 * UP)

        prefix_words = 2
        display_characters = sum(len(word) for word in text.split(" ")[:prefix_words])

        self.play(Write(text_mob[:display_characters], stroke_color=BLUE_B))
        self.wait()

        # Create word rects

        # TODO add animation update func? Or explicitly draw boxes on next tokens generation.
        processed_letters = 0
        words = text.split(" ")
        all_rects = []
        for word in words:
            word_len = len(word)
            word_mob = text_mob[processed_letters:processed_letters + word_len]
            rect = SurroundingRectangle(word_mob)
            processed_letters += word_len

            rect.set_height(text_mob.get_height() + SMALL_BUFF, stretch=True)
            rect.set_width(word_mob.get_width() + (SMALL_BUFF / 5), stretch=True)
            rect.match_y(text_mob)
            rect.set_stroke(GREY, 1)
            rect.set_fill(GREY, 0.25)
            all_rects.append(rect)

        self.add(VGroup(*all_rects[:prefix_words]))
        self.wait()

        # Adjectives updating noun
        adjs = ["fluffy", "blue", "verdant"]
        nouns = ["creature", "forest"]
        others = ["a", "roamed", "the"]
        adj_mobs, noun_mobs, other_mobs = [
            VGroup(word2mob[substr] for substr in group)
            for group in [adjs, nouns, others]
        ]
        adj_rects, noun_rects, other_rects = [
            VGroup(word2rect[substr] for substr in group)
            for group in [adjs, nouns, others]
        ]
        adj_rects.set_submobject_colors_by_gradient(BLUE_C, BLUE_D, GREEN)
        noun_rects.set_color(GREY_BROWN).set_stroke(width=3)
        kw = dict()
        adj_arrows = VGroup(
            Arrow(
                adj_mobs[i].get_top(), noun_mobs[j].get_top(),
                path_arc=-150 * DEGREES, buff=0.1, stroke_color=GREY_B
            )
            for i, j in [(0, 0), (1, 0), (2, 1)]
        )

        self.play(
            LaggedStartMap(DrawBorderThenFill, adj_rects),
            Animation(adj_mobs),
        )
        self.wait()
        self.play(
            LaggedStartMap(DrawBorderThenFill, noun_rects),
            Animation(noun_mobs),
            LaggedStartMap(ShowCreation, adj_arrows, lag_ratio=0.2, run_time=1.5),
        )

        kw = dict(time_width=2, max_stroke_width=10, lag_ratio=0.2, path_arc=150 * DEGREES)
        self.play(
            ContextAnimation(noun_mobs[0], adj_mobs[:2], strengths=[1, 1], **kw),
            ContextAnimation(noun_mobs[1], adj_mobs[2:], strengths=[1], **kw),
        )
        self.wait()
