from __future__ import annotations
import copy
from collections import defaultdict
import json
import math
from pathlib import Path
from tkinter import RIGHT
from manim_imports_ext import *
from manimlib.scene.interactive_scene import InteractiveScene
from manimlib.mobject.svg.image_mobject import ImageMobject
from manimlib.mobject.svg.text_mobject import Text
from manimlib.constants import FRAME_WIDTH, GREY_A, GREY_C, GREEN_A, GREEN_C, BLUE_A, BLUE_C, DOWN, LEFT, RIGHT, UP
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.animation.creation import Write
from _2024.transformers.helpers import *
from _2024.transformers.embedding import break_into_words
from _2024.transformers.embedding import break_into_tokens
from _2024.transformers.embedding import get_piece_rectangles

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
        system_prompt = Text(
            f"{system_prompt_text} You are a helpful assistant.",
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
        user_request_text = Text(
            "What is the capital of Russia?",
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

        assistant_response_text = "The capital of Russia is Moscow."
        assistant_response_text = Text(
            assistant_response_text,
            font_size=FONT_SIZE,
        )
        assistant_response_text.set_color(BLUE_A)
        assistant_response_text.align_to(assistant_prompt, RIGHT)
        assistant_response_text.next_to(assistant_prompt, RIGHT, buff=0.1)

        prompt_group = VGroup(system_prompt, user_request_prompt, user_request_text, assistant_prompt, assistant_response_text)
        prompt_group.shift(1 * DOWN)
        prompt_group.shift( -prompt_group.get_center()[0] * RIGHT )


        self.add(system_prompt, user_request_prompt, user_request_text)
        self.wait(0.5)

        self.add(assistant_prompt)
        self.play(Write(assistant_response_text))

        self.wait(5)




