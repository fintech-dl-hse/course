from __future__ import annotations
import copy
from collections import defaultdict
import json
import math
from pathlib import Path
from tkinter import RIGHT
from manim_imports_ext import *
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

        # Write text for system prompt, user request
        system_prompt_text = '<system>'
        system_prompt = Text(
            f"{system_prompt_text} You are a helpful assistant.",
            font_size=20,
        )
        system_prompt.set_color(BLUE_A)
        system_prompt[:len(system_prompt_text)].set_color(BLUE_C)

        user_prompt_text = '<user>'
        user_request_prompt = Text(
            f"{user_prompt_text} ",
            font_size=20,
        )
        user_request_prompt.set_color(GREEN_A)
        user_request_prompt[:len(user_prompt_text)].set_color(GREEN_C)
        user_request_text = Text(
            "What is the capital of Russia?",
            font_size=20,
        )
        user_request_text.set_color(GREEN_A)

        user_request_prompt.next_to(system_prompt, DOWN, buff=0.1)
        user_request_prompt.align_to(system_prompt, LEFT)
        user_request_text.align_to(user_request_prompt, DOWN)
        user_request_text.next_to(user_request_prompt, RIGHT, buff=0.1)

        self.add(system_prompt, user_request_prompt)
        self.play(Write(user_request_text))

        self.wait(5)




