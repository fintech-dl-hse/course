from manimlib import *
import numpy as np

class PrefillDecode(Scene):
    def create_transformer_stack(self, num_layers=6, width=4.5, height=5.0):
        layer_height = height / num_layers
        layers = []
        for i in range(num_layers):
            rect = Rectangle(
                width=width,
                height=layer_height * 0.85,
                stroke_color=WHITE,
                fill_color=BLUE_E,
                fill_opacity=0.15,
            )
            layers.append(rect)
        stack = VGroup(*layers).arrange(DOWN, buff=layer_height * 0.15)
        title = Text("Transformer", color=WHITE).scale(0.5)
        title.next_to(stack, UP, buff=0.25)
        group = VGroup(title, stack)
        return group

    def create_tokens_row(self, token_texts, color=YELLOW_E):
        tokens = []
        for t in token_texts:
            box = RoundedRectangle(width=0.7, height=0.5, corner_radius=0.08, fill_color=color, fill_opacity=0.25, stroke_color=color)
            label = Text(t).scale(0.35)
            label.move_to(box.get_center())
            g = VGroup(box, label)
            tokens.append(g)
        row = VGroup(*tokens).arrange(RIGHT, buff=0.15)
        return row

    def create_kv_cache(self, num_rows=4, num_cols=6):
        cells = []
        for r in range(num_rows):
            row_cells = []
            for c in range(num_cols):
                cell = Square(side_length=0.22, stroke_color=TEAL_E, fill_color=TEAL_E, fill_opacity=0.2)
                row_cells.append(cell)
            row = VGroup(*row_cells).arrange(RIGHT, buff=0.05)
            cells.append(row)
        grid = VGroup(*cells).arrange(DOWN, buff=0.05)
        title = Text("KV Cache", color=TEAL_A).scale(0.4)
        title.next_to(grid, UP, buff=0.15)
        return VGroup(title, grid)

    def create_resource_bars(self, compute_level=0.9, memory_level=0.4, width=4.5):
        # Base bars
        base_compute = Rectangle(width=width, height=0.25, stroke_color=GREY_BROWN, fill_color=GREY_BROWN, fill_opacity=0.2)
        base_memory = base_compute.copy()
        # Fill bars
        compute_fill = Rectangle(width=width * compute_level, height=0.25, stroke_color=GREEN_E, fill_color=GREEN_E, fill_opacity=0.6)
        memory_fill = Rectangle(width=width * memory_level, height=0.25, stroke_color=BLUE_E, fill_color=BLUE_E, fill_opacity=0.6)
        # Overlay fill on top of base and left-align
        compute_fill.move_to(base_compute)
        compute_fill.align_to(base_compute, LEFT)
        memory_fill.move_to(base_memory)
        memory_fill.align_to(base_memory, LEFT)
        # Labels
        compute_label = Text("Compute", color=GREEN_E).scale(0.4)
        memory_label = Text("Memory", color=BLUE_E).scale(0.4)
        # Layout
        compute_bar = VGroup(base_compute, compute_fill)
        memory_bar = VGroup(base_memory, memory_fill)
        compute_group = VGroup(compute_label, compute_bar).arrange(LEFT, buff=0.25)
        memory_group = VGroup(memory_label, memory_bar).arrange(LEFT, buff=0.25)
        bars = VGroup(compute_group, memory_group).arrange(DOWN, buff=0.25)
        return bars, compute_fill, memory_fill

    def create_logits_bar(self, num_candidates=6, select_index=2):
        bars = []
        heights = np.linspace(0.2, 1.0, num_candidates)
        np.random.seed(1)
        np.random.shuffle(heights)
        for i, h in enumerate(heights):
            color = YELLOW_E if i == select_index else GREY_B
            rect = Rectangle(width=0.25, height=h, stroke_color=color, fill_color=color, fill_opacity=0.8)
            bars.append(rect)
        group = VGroup(*bars).arrange(RIGHT, buff=0.08)
        title = Text("Logits / Probabilities", color=YELLOW_E).scale(0.35)
        title.next_to(group, UP, buff=0.15)
        return VGroup(title, group)

    def construct(self):
        title = Text("LLM Inference: Prefill vs Decode").scale(0.8)
        subtitle = Text("KV cache reuse and resource bottlenecks").scale(0.45).set_color(GREY_B)
        subtitle.next_to(title, DOWN, buff=0.15)
        header = VGroup(title, subtitle).to_edge(UP)
        self.play(FadeIn(header, shift=DOWN))
        self.wait(0.5)

        # Core diagram elements
        transformer = self.create_transformer_stack(num_layers=8)
        transformer.move_to(ORIGIN)
        self.play(FadeIn(transformer, shift=RIGHT))

        # Prefill section
        prefill_label = Text("Prefill (compute-bound)", color=GREEN_E).scale(0.5)
        prefill_label.to_edge(LEFT).shift(UP * 2.0)

        bars_prefill, compute_fill_prefill, memory_fill_prefill = self.create_resource_bars(compute_level=0.95, memory_level=0.45)
        bars_prefill.next_to(prefill_label, DOWN, buff=0.3).to_edge(LEFT, buff=0.75)

        prompt_tokens = ["You", "are", "an", "LLM", ":", "Explain"]
        prompt_row = self.create_tokens_row(prompt_tokens, color=YELLOW_E)
        prompt_row.to_edge(LEFT, buff=0.75)
        prompt_row.shift(DOWN * 2.2)

        self.play(
            FadeIn(prefill_label, shift=DOWN),
            FadeIn(bars_prefill, shift=DOWN),
            LaggedStart(*[FadeIn(tok, lag_ratio=0.1) for tok in prompt_row], lag_ratio=0.07),
        )

        # Animate prompt flowing into transformer (all tokens at once)
        token_arrows = VGroup()
        for tok in prompt_row:
            arr = Arrow(tok.get_top() + UP * 0.1, transformer[1].get_bottom() + DOWN * 0.2, color=YELLOW_E, buff=0.1, stroke_width=2.5)
            token_arrows.add(arr)
        self.play(LaggedStart(*[GrowArrow(a) for a in token_arrows], lag_ratio=0.05))

        # KV cache builds during prefill
        kv_cache = self.create_kv_cache(num_rows=5, num_cols=len(prompt_tokens))
        kv_cache.next_to(transformer, RIGHT, buff=1.0)
        self.play(FadeIn(kv_cache, shift=RIGHT))

        # Emphasize compute-bound nature
        compute_glow = SurroundingRectangle(compute_fill_prefill, color=GREEN_E, buff=0.05)
        self.play(ShowCreation(compute_glow))
        self.wait(0.5)

        # Sampling next token from logits
        logits = self.create_logits_bar(num_candidates=7, select_index=3)
        logits.next_to(kv_cache, RIGHT, buff=1.0)
        self.play(FadeIn(logits, shift=RIGHT))

        sampled = RoundedRectangle(width=0.7, height=0.5, corner_radius=0.08, fill_color=YELLOW_E, fill_opacity=0.4, stroke_color=YELLOW_E)
        sampled_label = Text("<BOS>").scale(0.35)
        sampled_group = VGroup(sampled, sampled_label)
        sampled_group.next_to(logits, DOWN, buff=0.5)
        self.play(GrowFromCenter(sampled_group))
        self.wait(0.75)

        # Transition to Decode
        decode_label = Text("Decode (memory-bound)", color=BLUE_E).scale(0.5)
        decode_label.to_edge(LEFT).shift(UP * 2.0)
        bars_decode, compute_fill_decode, memory_fill_decode = self.create_resource_bars(compute_level=0.35, memory_level=0.95)
        bars_decode.next_to(decode_label, DOWN, buff=0.3).to_edge(LEFT, buff=0.75)

        self.play(
            FadeOut(prefill_label, shift=UP),
            FadeOut(bars_prefill, shift=UP),
            FadeOut(token_arrows),
            FadeOut(logits),
        )
        self.play(FadeIn(decode_label, shift=DOWN), FadeIn(bars_decode, shift=DOWN))

        # Only latest token goes through transformer; KV cache reused
        latest_tok = RoundedRectangle(width=0.7, height=0.5, corner_radius=0.08, fill_color=PURPLE_E, fill_opacity=0.35, stroke_color=PURPLE_E)
        latest_lab = Text("tₙ").scale(0.35)
        latest = VGroup(latest_tok, latest_lab)
        latest.next_to(prompt_row, RIGHT, buff=0.6)
        latest.shift(RIGHT * 0.5)

        self.play(FadeIn(latest, shift=UP))
        arr_decode = Arrow(latest.get_top() + UP * 0.1, transformer[1].get_bottom() + DOWN * 0.2, color=PURPLE_E, buff=0.1, stroke_width=2.5)
        self.play(GrowArrow(arr_decode))

        # Highlight KV cache reuse
        kv_highlight = SurroundingRectangle(kv_cache, color=TEAL_E, buff=0.1)
        self.play(ShowCreation(kv_highlight), Indicate(kv_cache))

        # Generate a few tokens iteratively (decode steps)
        generated = VGroup(sampled_group.copy())
        generated.arrange(RIGHT, buff=0.15)
        generated.next_to(transformer, DOWN, buff=1.0)
        self.play(FadeIn(generated))

        for i in range(1, 4):
            step_tok = RoundedRectangle(width=0.7, height=0.5, corner_radius=0.08, fill_color=PURPLE_E, fill_opacity=0.35, stroke_color=PURPLE_E)
            step_lab = Text(f"t{i}").scale(0.35)
            step = VGroup(step_tok, step_lab)
            step.move_to(latest)
            arrow_step = Arrow(step.get_top() + UP * 0.1, transformer[1].get_bottom() + DOWN * 0.2, color=PURPLE_E, buff=0.1, stroke_width=2.5)

            next_out = RoundedRectangle(width=0.7, height=0.5, corner_radius=0.08, fill_color=YELLOW_E, fill_opacity=0.35, stroke_color=YELLOW_E)
            next_lab = Text(f"y{i}").scale(0.35)
            next_tok = VGroup(next_out, next_lab)
            next_tok.next_to(generated, RIGHT, buff=0.15)

            self.play(FadeIn(step, shift=UP))
            self.play(GrowArrow(arrow_step))
            self.play(Indicate(kv_cache))
            self.play(GrowFromCenter(next_tok))
            generated.add(next_tok)

        # Emphasize memory-bound nature
        memory_glow = SurroundingRectangle(memory_fill_decode, color=BLUE_E, buff=0.05)
        self.play(ShowCreation(memory_glow))
        self.wait(1.0)

        footer = Text("Prefill: process all prompt tokens (build KV)\nDecode: reuse KV, process last token only").scale(0.4)
        footer.set_color(GREY_B)
        footer.to_edge(DOWN)
        self.play(FadeIn(footer, shift=UP))
        self.wait(1.5)