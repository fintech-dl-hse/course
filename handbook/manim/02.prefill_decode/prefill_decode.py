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

        self.wait(5.0)


