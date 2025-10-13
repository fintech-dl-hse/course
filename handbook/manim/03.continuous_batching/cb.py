from manimlib import *

# 🎬 Continuous Batching Visualization
# Tested with manimGL

class ContinuousBatching(Scene):
    def construct(self):
        # СЦЕНА 1: Пояснение обычного batching
        title = Text("Batch с паддингом", font="Roboto", color=YELLOW).scale(0.8)
        title.to_edge(UP)
        self.play(Write(title))

        # Матрица с паддингом
        tokens = [
            ["Hello", "world", "[PAD]", "[PAD]", "[PAD]"],
            ["The", "weather", "is", "nice", "today"]
        ]

        grid = VGroup()
        colors = [BLUE, GREEN, GREY, GREY, GREY]

        for row_idx, row in enumerate(tokens):
            row_group = VGroup()
            for i, token in enumerate(row):
                color = colors[i] if "[PAD]" not in token else GREY
                box = Rectangle(height=0.6, width=1, color=WHITE)
                text = Text(token, font="Roboto", color=color, size=0.25)
                group = VGroup(box, text).arrange(DOWN, buff=0.05)
                row_group.add(group)
            row_group.arrange(RIGHT, buff=0.1)
            grid.add(row_group)

        grid.arrange(DOWN, buff=0.2)
        grid.shift(DOWN*0.5)
        self.play(LaggedStart(*[FadeIn(r) for r in grid], lag_ratio=0.1))
        self.wait(1)

        # Подсветка паддингов
        pad_rects = [r for r in grid[0][2:] if isinstance(r, VGroup)]
        pad_rects += [r for r in grid[1] if "[PAD]" in r[1].text]
        self.play(*[r[1].animate.set_color(RED) for r in pad_rects])
        self.play(Indicate(grid[0][2:], color=RED))
        self.wait(1)

        note1 = Text("Много пустых токенов → неэффективно", font="Roboto", color=RED).scale(0.6)
        note1.next_to(grid, DOWN)
        self.play(Write(note1))
        self.wait(1.5)

        # СЦЕНА 2: Переход к continuous batching
        self.play(FadeOut(note1), FadeOut(title))
        self.wait(0.5)

        title2 = Text("Continuous batching", font="Roboto", color=YELLOW).scale(0.8)
        title2.to_edge(UP)
        self.play(Write(title2))

        # "Расплющиваем" батч
        flattened_tokens = ["Hello", "world", "The", "weather", "is", "nice", "today"]
        colors2 = [BLUE, BLUE, GREEN, GREEN, GREEN, GREEN, GREEN]

        flat_group = VGroup()
        for token, color in zip(flattened_tokens, colors2):
            box = Rectangle(height=0.6, width=1, color=WHITE)
            text = Text(token, font="Roboto", color=color, size=0.25)
            group = VGroup(box, text).arrange(DOWN, buff=0.05)
            flat_group.add(group)

        flat_group.arrange(RIGHT, buff=0.1)
        flat_group.move_to(ORIGIN)

        self.play(
            grid.animate.arrange(RIGHT, buff=0.05).scale(0.8).move_to(ORIGIN),
            run_time=1.5
        )
        self.play(Transform(grid, flat_group), run_time=2)
        self.wait(1)

        # Покажем tensor с длинами
        lengths = Text("lengths = [2, 5]", font="Roboto", color=WHITE).scale(0.6)
        lengths.next_to(flat_group, DOWN)
        self.play(Write(lengths))
        self.wait(1.5)

        # Добавим визуальные границы между сэмплами
        line = Line(
            start=flat_group[1].get_right() + 0.1*RIGHT,
            end=flat_group[1].get_right() + 0.1*RIGHT + 1.5*UP,
            color=YELLOW
        )
        self.play(ShowCreation(line))
        self.wait(1)

        # СЦЕНА 3: Вывод
        note2 = Text("Без паддингов → быстрее и экономнее", font="Roboto", color=GREEN).scale(0.6)
        note2.next_to(lengths, DOWN)
        self.play(Write(note2))
        self.wait(2)

        # Завершение
        self.play(FadeOut(grid), FadeOut(lengths), FadeOut(line), FadeOut(note2))
        outro = Text("Continuous batching = эффективность ⚡️", font="Roboto", color=YELLOW).scale(0.9)
        self.play(Write(outro))
        self.wait(2)



