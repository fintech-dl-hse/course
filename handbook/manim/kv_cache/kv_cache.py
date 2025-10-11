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

        # Create word rects
        processed_letters = 0
        words = text.split(" ")
        all_rects = []
        words_mobs = []
        for word_i, word in enumerate(words):
            word_len = len(word)
            word_mob = text_mob[processed_letters:processed_letters + word_len]
            words_mobs.append(word_mob)
            processed_letters += word_len

            if word_i >= prefix_words:
                continue

            rect = SurroundingRectangle(word_mob)
            rect.set_height(text_mob.get_height() + SMALL_BUFF, stretch=True)
            rect.set_width(word_mob.get_width() + (SMALL_BUFF / 5), stretch=True)
            rect.match_y(text_mob)
            rect.set_stroke(GREY, 1)
            rect.set_fill(GREY, 0.25)
            all_rects.append(rect)

        self.add(VGroup(*all_rects[:prefix_words]))
        self.play(Write(text_mob[:display_characters], stroke_color=BLUE_B))
        self.wait()

        for word_i in range(prefix_words, len(words_mobs)):
            word_mob = words_mobs[word_i]
            rect = SurroundingRectangle(word_mob)
            rect.set_height(word_mob.get_height() + SMALL_BUFF, stretch=True)
            rect.set_width(word_mob.get_width() + (SMALL_BUFF / 2), stretch=True)
            rect.match_y(word_mob)
            rect.set_stroke(GREY, 1)
            rect.set_fill(GREY, 0.25)

            adj_arrows = VGroup(
                Arrow(
                    words_mobs[i].get_top(), word_mob.get_top(),
                    path_arc=-150 * DEGREES, buff=0.1, stroke_color=GREY_B,
                    thickness=1.0
                )
                for i in range(word_i)
            )

            self.add(rect, adj_arrows)
            self.play(Write(word_mob, stroke_color=BLUE_B))
            self.wait()

            adj_arrows.clear()

        self.wait()
