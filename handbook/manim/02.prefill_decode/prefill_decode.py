from __future__ import annotations
import copy
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from manim_imports_ext import *
from manimlib.scene.interactive_scene import InteractiveScene

from manimlib.constants import FRAME_WIDTH, GREY_A, GREY_C, GREEN_A, GREEN_C, BLUE_A, BLUE_C, DOWN, LEFT, RIGHT, UP
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.animation.creation import Write
from _2024.transformers.helpers import *
from _2024.transformers.embedding import break_into_words
from _2024.transformers.embedding import break_into_tokens
from _2024.transformers.embedding import get_piece_rectangles

# Import helper from kv_cache module
sys.path.insert(0, str(Path(__file__).parent.parent / "01.kv_cache"))
from kv_cache import create_boxed_tokens_objects


def create_prompt_section(tag_text: str, content_text: str, tag_color: str, content_color: str, font_size: int = 18):
    """Create a prompt section with a tag and content."""
    prompt = Text(f"{tag_text} {content_text}", font_size=font_size)
    prompt.set_color(content_color)
    prompt[:len(tag_text)].set_color(tag_color)
    return prompt


def add_boxed_tokens(
    scene,
    text_str: str,
    text_mob: Text,
    box_color: str,
    animate: bool = False,
    run_time: float = 0.3,
):
    """Add boxed tokens to the scene, optionally animating them."""
    token_data = create_boxed_tokens_objects(
        text_str,
        text_mob,
        box_color=box_color,
        prefix_tokens=0,
        box_opacity=0.1,
        box_stroke_width=1,
    )
    token_mobs = token_data['token_mobs']
    create_box_func = token_data['create_box_func']

    if animate:
        # Animate tokens appearing one by one
        for token_mob in token_mobs:
            box = create_box_func(token_mob, color=box_color)
            scene.play(
                FadeIn(token_mob),
                FadeIn(box),
                run_time=run_time,
            )
    else:
        # Add all boxes at once
        for token_mob in token_mobs:
            box = create_box_func(token_mob, color=box_color)
            scene.add(box)


class IntroMeme(InteractiveScene):
    def construct(self):
        # add image from videos/kv-cache-cover.jpg
        image = ImageMobject("videos/prefill_decode/cover.jpg")
        image.set_width(FRAME_WIDTH)
        image.center()
        self.add(image)

        self.wait(6.5)

class PrefillDecodeExample(InteractiveScene):
    def construct(self):
        FONT_SIZE = 18

        # Define prompt sections
        sections = [
            {
                'tag': '<system>',
                'content': "You are a helpful assistant.",
                'tag_color': GREY_C,
                'content_color': GREY_A,
                'box_color': None,
            },
            {
                'tag': '<user>',
                'content': "What is the capital of Russia?",
                'tag_color': GREEN_C,
                'content_color': GREEN_A,
                'box_color': GREEN_C,
            },
            {
                'tag': '<assistant>',
                'content': "The capital of Russia is Moscow.",
                'tag_color': BLUE_C,
                'content_color': BLUE_A,
                'box_color': BLUE_C,
                'animate': True,  # Only assistant response is animated
            },
        ]

        # Create prompt mobjects
        prompt_mobs = []
        content_mobs = []
        content_texts = []

        for i, section in enumerate(sections):
            tag_text = section['tag']
            content_text = section['content']

            if i < 2:  # System and user: tag + content together
                prompt = create_prompt_section(
                    tag_text, content_text,
                    section['tag_color'], section['content_color'], FONT_SIZE
                )
                prompt_mob = prompt[:len(tag_text)]
                prompt_mobs.append(prompt_mob)
                content_mob = prompt[len(tag_text):]
                content_mobs.append(content_mob)
                content_texts.append(content_text)
            else:  # Assistant: tag and content separate
                tag_mob = Text(f"{tag_text} ", font_size=FONT_SIZE)
                tag_mob.set_color(section['tag_color'])
                prompt_mobs.append(tag_mob)
                prompt_mob = tag_mob

                content_mob = Text(content_text, font_size=FONT_SIZE)
                content_mob.set_color(section['content_color'])
                content_mobs.append(content_mob)
                content_texts.append(content_text)

            if i > 0:
                # Mobs Positioning
                prompt_mob.next_to(prompt_mobs[i-1], DOWN, buff=0.1)
                prompt_mob.align_to(prompt_mobs[0], LEFT)

            content_mob.next_to(prompt_mobs[i], RIGHT, buff=0.1)

        # Center everything
        all_mobs = prompt_mobs + content_mobs
        prompt_group = VGroup(*all_mobs)
        prompt_group.shift(1 * DOWN)
        prompt_group.shift(-prompt_group.get_center()[0] * RIGHT)

        # Display sections with boxes
        for i, section in enumerate(sections):
            should_animate = section.get('animate', False)

            # Add prompt mobject(s)
            if i < 2:
                self.add(prompt_mobs[i])
                self.add(content_mobs[i])
            else:
                self.add(prompt_mobs[i])

            # Add boxed tokens
            add_boxed_tokens(
                self,
                content_texts[i],
                content_mobs[i],
                section['box_color'],
                animate=should_animate,
                run_time=0.3,
            )

            # Wait between sections (except before last)
            # if i < len(sections) - 1:
            #     self.wait(0.5)

        self.wait(5)




