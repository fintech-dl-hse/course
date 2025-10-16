from __future__ import annotations
import copy
from collections import defaultdict
import json
import math
from pathlib import Path
from manim_imports_ext import *
from _2024.transformers.helpers import *
from _2024.transformers.embedding import break_into_words
from _2024.transformers.embedding import break_into_tokens
from _2024.transformers.embedding import get_piece_rectangles

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
        header.shift(0.1 * RIGHT)

        # Sentence
        text = sentence_text
        text_mob = Text(text, font_size=30, fill_color=YELLOW_D)
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

        # Bars axis and per-token recomputation bars/labels
        num_words = len(words_mobs)
        # Upper bound for recomputations per token without KV cache is (num_words - 1)
        bars_axes = Axes(
            x_range=[0, num_words, 1],
            y_range=[0, max(1, num_words), 1],
            width=text_mob.get_width(),
            height=2.5,
            x_axis_config={"include_tip": False, "include_ticks": False},
            y_axis_config={"include_tip": False, "include_ticks": True},
        )
        bars_axes.next_to(text_mob, DOWN, buff=1.0)

        bar_width_units = 0.4
        bars = VGroup()
        # Precompute alignment helpers
        baseline_y = bars_axes.c2p(0, 0)[1]
        word_centers_x = [mob.get_center()[0] for mob in words_mobs]
        for i in range(num_words):
            # Start with ~0 height to avoid zero-height rectangle issues
            init_height_units = 1e-3
            init_t = 0.0  # start color like arrows at lowest count
            init_color = interpolate_color(BLUE_B, RED_E, init_t)
            rect = Rectangle(
                width=bars_axes.get_x_axis().get_unit_size() * bar_width_units,
                height=bars_axes.get_y_axis().get_unit_size() * init_height_units,
                stroke_width=1,
                stroke_color=init_color,
                fill_color=init_color,
                fill_opacity=0.85,
            )
            rect.move_to(np.array([word_centers_x[i], baseline_y, 0.0]), DOWN)
            bars.add(rect)

        # Numeric recomputation counters under each word
        recompute_counts = [0 for _ in range(num_words)]

        for i in range(prefix_words):
            recompute_counts[i] = 1

        # X-axis labels with words, positioned under the axis and aligned with words
        x_word_labels = VGroup()
        for i, word in enumerate(words):
            wlbl = Text(word, font_size=22, fill_color=GREY_B)
            # Place at word center x, slightly below axis baseline
            wlbl.move_to([word_centers_x[i], baseline_y - 0.35, 0])
            x_word_labels.add(wlbl)

        # Y-axis title
        y_title = Text("Recomputations", font_size=24, fill_color=GREY_B)
        y_title.rotate(PI / 2)
        y_title.next_to(bars_axes.get_y_axis(), LEFT, buff=0.1)

        # Initial draws
        self.add(VGroup(*all_rects[:prefix_words]))
        self.add(text_mob[:display_characters])
        self.add(header)
        self.add(bars_axes, bars, x_word_labels, y_title)
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
                recompute_counts[word_i] += 1

                for word_from_arrow in range(word_to_arrow):
                    key = (word_from_arrow, word_to_arrow)
                    current_count = arrow_counts[key] + 1
                    t = min(max((current_count - 1) / 4.0, 0.0), 1.0)
                    color = interpolate_color(BLUE_B, RED_E, t)
                    arrow = Arrow(
                        words_mobs[word_from_arrow].get_top(), words_mobs[word_to_arrow].get_top(),
                        path_arc=-150 * DEGREES, buff=0.1, stroke_color=color,
                        thickness=1.5,
                        fill_opacity=1.0,
                        fill_color=color,
                    )
                    arrows.append(arrow)
                    arrow_counts[key] = current_count
            else:
                # All pairwise up to current word
                for word_to_arrow in range(word_i + 1):
                    recompute_counts[word_to_arrow] += 1

                    for word_from_arrow in range(word_to_arrow):
                        key = (word_from_arrow, word_to_arrow)
                        current_count = arrow_counts[key] + 1
                        t = min(max((current_count - 1) / 4.0, 0.0), 1.0)
                        color = interpolate_color(BLUE_B, RED_E, t)
                        arrow = Arrow(
                            words_mobs[word_from_arrow].get_top(), words_mobs[word_to_arrow].get_top(),
                            path_arc=-150 * DEGREES, buff=0.1, stroke_color=color,
                            thickness=1.5,
                            fill_opacity=1.0,
                            fill_color=color,
                        )
                        arrows.append(arrow)
                        arrow_counts[key] = current_count

            self.add(rect, step_mob)
            # Prepare bar and counter updates (no-KV case increments past tokens)
            bar_anims = []
            label_anims = []

            for j in range(word_i+1):
                new_h_units = recompute_counts[j]
                # Match arrow color schedule based on count
                t = min(max((new_h_units - 1) / 4.0, 0.0), 1.0)
                new_color = interpolate_color(BLUE_B, RED_E, t)
                # New rectangle with updated height, anchored at y=0
                new_rect = Rectangle(
                    width=bars_axes.get_x_axis().get_unit_size() * bar_width_units,
                    height=bars_axes.get_y_axis().get_unit_size() * max(new_h_units, 1e-3),
                    stroke_width=1,
                    stroke_color=new_color,
                    fill_color=new_color,
                    fill_opacity=0.85,
                )
                new_rect.move_to(np.array([word_centers_x[j], baseline_y, 0.0]), DOWN)
                bar_anims.append(Transform(bars[j], new_rect))


            self.play(
                *[GrowArrow(arrow, run_time=0.2) for arrow in arrows],
                Write(word_mob),
                *bar_anims,
                *label_anims,
            )
            # self.wait()

            if not use_kv_cache:
                self.play(*[FadeOut(arrow, run_time=0.2) for arrow in arrows])
                # self.wait()

            step_mob.clear()
            generation_step += 1

        self.wait(0.1)

        # Add memory complexity below
        mem_title = Text("Memory Complexity", font_size=25)
        mem_title.move_to(bars_axes, UP)
        mem_title.set_color(GREEN_C)
        mem_title.shift(0.4 * LEFT)
        # mem_title.next_to(title, DOWN, buff=0.4)

        tex_kwargs = {
            't2c': {
                'n': BLUE_C,
                'k': TEAL_C,
            }
        }

        if use_kv_cache:
            mem_formula = Tex(r"O(n)", font_size=30, **tex_kwargs)
        else:
            mem_formula = Tex(r"O(1)", font_size=30, **tex_kwargs)
        mem_formula.next_to(mem_title, DOWN, buff=0.2)
        mem_formula.align_to(mem_title, LEFT)
        mem_formula.shift(0.5 * RIGHT)


        title = Text("Computation Complexity", font_size=25)
        title.set_color(GREEN_C)
        title.next_to(mem_formula, DOWN, buff=0.4)
        title.align_to(mem_title, LEFT)

        # Transform recomputation axis and bars into complexity formula
        if use_kv_cache:
            formula = Tex(
                r"O(n)\ \text{as}\ \sum_{k=1}^{n} 1",
                font_size=30,
                **tex_kwargs,
            )
        else:
            formula = Tex(
                'O(n^2)\ \\text{as}',
                font_size=30,
                **tex_kwargs,
            )
        formula.next_to(title, DOWN, buff=0.2)
        formula.align_to(mem_title, LEFT)
        formula.shift(0.5 * RIGHT)

        target_objects = [formula, mem_formula]

        if not use_kv_cache:
            formula_extended = Tex(
                '\sum_{k=1}^{n} k\ =\ \\frac{(n-1)n}{2}\ =\ \\frac{n^2 - n}{2}',
                font_size=20,
                **tex_kwargs,
            )
            formula_extended.next_to(formula, DOWN, buff=0.2)
            formula_extended.align_to(formula, LEFT)
            formula_extended.shift(0.3 * RIGHT)

            target_objects.append(formula_extended)

        self.play(
            ReplacementTransform(VGroup(bars_axes, bars, x_word_labels, y_title), VGroup(*target_objects)),
            Write(VGroup(title, mem_title)),
            run_time=2.0,
        )

        self.wait(0.1)



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
        blocks.set_fill(MAROON_E, 1)
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

        formula = Tex(
            'x_t \sim P_{\\theta}(x_t | x_1, ..., x_{t-1})',
            font_size=24,
            t2c={
                'x_t': GREEN_E,
                'x_1, ..., x_{t-1}': YELLOW_D,
            },
            # stroke_width=1.1,
            fill_border_width=1.0
        )
        formula.next_to(word, DOWN, buff=0.2)
        formula.set_backstroke(BLACK, 3)
        # formula.align_to(word, LEFT)

        out_arrow = Vector(
            0.5 * RIGHT, stroke_width=10,
            max_tip_length_to_length_ratio=0.5,
            max_width_to_length_ratio=12
        )
        out_arrow.next_to(blocks[-1], RIGHT, buff=SMALL_BUFF)
        out_arrow.set_opacity(0)

        result = VGroup(blocks, word, out_arrow, formula)
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
        generated_text_mob = Text(text, font_size=30, fill_color=GREEN_E)
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
            header_color=GREEN_E,
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


class KVCacheSizeVsSequenceLength(InteractiveScene):

    def load_coeff(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        a = int(payload["a_bytes_per_token"])  # bytes per token
        b = int(payload.get("b_bytes", 0))
        meta = payload.get("meta", {})
        return a, b, meta

    def construct(self):
        # Locate JSON coefficient files next to this script
        base_dir = Path(__file__).resolve().parent
        files = [
            (base_dir / "kv_cache_coeffs_llama3.1-7B.json", TEAL),
            (base_dir / "kv_cache_coeffs_llama3.2-3B.json", MAROON_A),
        ]

        models = []
        max_tokens = 0
        counts_union = set()
        for path, color in files:
            if not path.exists():
                raise ValueError(f"Model file not found: {path}")

            a, b, meta = self.load_coeff(path)
            display_name = meta.get("model", path.stem)
            counts = meta.get("counts", [1_000_000])
            for c in counts:
                if isinstance(c, (int, float)):
                    counts_union.add(int(c))
            max_tokens = max(max_tokens, max(counts))
            models.append({
                "name": display_name,
                "a": a,
                "b": b,
                "color": color,
            })

        # Compute y range in GB with headroom
        max_bytes = max(m["a"] * max_tokens + m["b"] for m in models)
        max_gb = max_bytes / 1e9
        y_max = max(1.0, max_gb * 1.1)

        # Header
        header = Text("KV-Cache size vs sequence length", font_size=20)
        header.to_corner(LEFT + UP).fix_in_frame()

        # Axes
        axes = Axes(
            x_range=[0, max_tokens, max(1, max_tokens // 5)],
            y_range=[0, y_max, max(0.5, y_max / 5)],
            width=2,
            height=2,
            x_axis_config={"include_tip": False, "include_ticks": True},
            y_axis_config={"include_tip": False, "include_ticks": True},
            # ticks
        )
        axes.center().shift(0.5 * DOWN)

        x_label = Text("Sequence length (tokens)", font_size=16)
        x_label.next_to(axes, DOWN, buff=0.3)
        y_label = Text("KV-Cache size (GB)", font_size=16)
        y_label.rotate(PI / 2)
        y_label.next_to(axes.get_y_axis(), LEFT, buff=0.3)

        self.add(header)
        self.add(axes, x_label, y_label)

        # Explicit ticks and labels
        def abbrev(n: int) -> str:
            if n >= 1_000_000:
                return f"{n // 1_000_000}M"
            if n >= 1_000:
                return f"{n // 1_000}k"
            return str(n)

        # X ticks at ~5 uniform intervals (tokens)
        x_step = max(1, int(math.ceil(max_tokens / 5.0)))
        x_vals = [k * x_step for k in range(0, int(math.floor(max_tokens / x_step)) + 1)]

        x_ticks = VGroup()
        for xv in x_vals:
            p = axes.c2p(xv, 0)
            tick = Line(p + 0.05 * UP, p + 0.05 * DOWN, stroke_color=GREY_B, stroke_width=2)
            label = Text(abbrev(int(xv)), font_size=12)
            label.next_to(tick, DOWN, buff=0.05)
            x_ticks.add(VGroup(tick, label))

        # Y ticks at ~5 intervals (GB)
        y_step = max(1.0, math.ceil(y_max / 5.0))
        y_vals = [k * y_step for k in range(0, int(math.floor(y_max / y_step)) + 1)]

        y_ticks = VGroup()
        for yv in y_vals:
            p = axes.c2p(0, yv)
            tick = Line(p + 0.05 * LEFT, p + 0.05 * RIGHT, stroke_color=GREY_B, stroke_width=2)
            if yv > 0:
                label = Text(f"{int(yv)}", font_size=12)
                label.next_to(tick, LEFT, buff=0.05)
                y_ticks.add(VGroup(tick, label))
            else:
                y_ticks.add(tick)

        self.add(x_ticks, y_ticks)

        # Plot lines (linear: size_bytes = a * n + b). Convert to GB for plotting
        lines = VGroup()
        for m in models:
            a = m["a"]
            b = m["b"]
            start = axes.c2p(0, b / 1e9)
            end = axes.c2p(max_tokens, (a * max_tokens + b) / 1e9)
            line = Line(start, end, stroke_color=m["color"], stroke_width=6)
            lines.add(line)

        self.play(*[ShowCreation(line) for line in lines])

        # Legend
        legend_items = VGroup()
        for i, m in enumerate(models):
            swatch = Line(ORIGIN, 0.2 * RIGHT, stroke_color=m["color"], stroke_width=8)
            label = Text(m["name"], font_size=14)
            item = VGroup(swatch, label)
            item.arrange(RIGHT, buff=0.1)
            legend_items.add(item)

        legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        legend_bg = SurroundingRectangle(legend_items, buff=0.1)
        legend_bg.set_stroke(GREY_B, 1)
        legend_bg.set_fill(BLACK, 0.2)
        legend = VGroup(legend_bg, legend_items)
        legend.to_corner(RIGHT + UP).shift(0.5 * LEFT + 0.5 * DOWN)
        legend.fix_in_frame()

        self.add(legend)
        self.wait(0.5)
