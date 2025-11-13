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

        # Write text for system prompt, user request
        system_prompt_text = '<system>'
        system_content_str = "You are a helpful assistant."
        system_prompt = Text(
            f"{system_prompt_text} {system_content_str}",
            font_size=FONT_SIZE,
        )
        system_prompt.set_color(GREY_A)
        system_prompt[:len(system_prompt_text)].set_color(GREY_C)

        user_prompt_text = '<user>'
        user_request_prompt = Text(
            f"{user_prompt_text} ",
            font_size=FONT_SIZE,
        )
        user_request_prompt.set_color(GREEN_A)
        user_request_prompt[:len(user_prompt_text)].set_color(GREEN_C)
        user_request_text_str = "What is the capital of Russia?"
        user_request_text = Text(
            user_request_text_str,
            font_size=FONT_SIZE,
        )
        user_request_text.set_color(GREEN_A)

        user_request_prompt.next_to(system_prompt, DOWN, buff=0.1)
        user_request_prompt.align_to(system_prompt, LEFT)
        user_request_text.align_to(user_request_prompt, DOWN)
        user_request_text.next_to(user_request_prompt, RIGHT, buff=0.1)

        assistant_prompt_text = '<assistant>'
        assistant_prompt = Text(
            f"{assistant_prompt_text} ",
            font_size=FONT_SIZE,
        )
        assistant_prompt.set_color(BLUE_C)
        assistant_prompt.next_to(user_request_prompt, DOWN, buff=0.1)
        assistant_prompt.align_to(user_request_prompt, LEFT)

        assistant_response_text_str = "The capital of Russia is Moscow."
        assistant_response_text = Text(
            assistant_response_text_str,
            font_size=FONT_SIZE,
        )
        assistant_response_text.set_color(BLUE_A)
        assistant_response_text.align_to(assistant_prompt, RIGHT)
        assistant_response_text.next_to(assistant_prompt, RIGHT, buff=0.1)

        prompt_group = VGroup(system_prompt, user_request_prompt, user_request_text, assistant_prompt, assistant_response_text)
        prompt_group.shift(1 * DOWN)
        prompt_group.shift( -prompt_group.get_center()[0] * RIGHT )

        # Extract content portion from system_prompt (after the tag) for box creation
        # system_prompt contains: "<system> You are a helpful assistant."
        tag_len = len(system_prompt_text) + 1  # +1 for space
        system_content_mob = system_prompt[tag_len:]  # This is a slice of the original Text

        # Use helper to create boxed tokens for system content
        # Note: The helper will extract tokens from the slice, which works correctly
        system_token_data = create_boxed_tokens_objects(
            system_content_str,
            system_content_mob,
            box_color=GREY_C,
            prefix_tokens=0,  # All tokens visible at once
        )
        system_token_mobs = system_token_data['token_mobs']
        system_create_box_func = system_token_data['create_box_func']

        # Use helper to create boxed tokens for user request
        user_token_data = create_boxed_tokens_objects(
            user_request_text_str,
            user_request_text,
            box_color=GREEN_C,
            prefix_tokens=0,  # All tokens visible at once
        )
        user_token_mobs = user_token_data['token_mobs']
        user_create_box_func = user_token_data['create_box_func']

        # Add system prompt with boxes
        self.add(system_prompt)
        for token_mob in system_token_mobs:
            box = system_create_box_func(token_mob, color=GREY_C)
            self.add(box)

        # Add user request with boxes
        self.add(user_request_prompt, user_request_text)
        for token_mob in user_token_mobs:
            box = user_create_box_func(token_mob, color=GREEN_C)
            self.add(box)

        self.wait(0.5)

        self.add(assistant_prompt)

        # Use helper to create boxed tokens for assistant response
        token_data = create_boxed_tokens_objects(
            assistant_response_text_str,
            assistant_response_text,
            box_color=BLUE_C,
            prefix_tokens=0,  # Start with no tokens visible
        )
        token_mobs = token_data['token_mobs']
        create_box_func = token_data['create_box_func']

        # Animate tokens appearing one by one with boxes
        for token_mob in token_mobs:
            box = create_box_func(token_mob, color=BLUE_C)
            self.play(
                FadeIn(token_mob),
                FadeIn(box),
                run_time=0.3,
            )

        self.wait(5)




