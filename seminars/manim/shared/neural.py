"""Reusable ManimCE primitives for neural-network scenes.

Общие примитивы для визуализации нейронных сетей (Neuron / LabeledBox / arrow_between).
Used across seminar scenes so layout and styling stay consistent.
"""
from __future__ import annotations

from typing import Any, Optional

from manim import (
    Arrow,
    Circle,
    MathTex,
    RoundedRectangle,
    VGroup,
    VMobject,
    BLUE,
    GREY_B,
    WHITE,
)


class Neuron(VGroup):
    """Labeled circle representing a neuron / activation node.

    Пример использования: `Neuron(label="x_1")` для входа, `Neuron(label="h_t")` для скрытого состояния.

    Args:
        label: Математическая метка (передаётся в MathTex). Пустая строка — без метки.
        radius: Радиус круга.
        color: Цвет окружности.
        fill_opacity: Прозрачность заливки (0..1).
        label_scale: Масштаб MathTex-метки (по умолчанию 0.7 для совместимости).
        **kwargs: Проброс в VGroup.
    """

    def __init__(
        self,
        label: str = "",
        radius: float = 0.4,
        color: str = BLUE,
        fill_opacity: float = 0.15,
        label_scale: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.circle = Circle(radius=radius, color=color).set_fill(color, opacity=fill_opacity)
        self.add(self.circle)
        if label:
            self.label_tex: Optional[MathTex] = MathTex(label).scale(label_scale)
            self.label_tex.move_to(self.circle.get_center())
            self.add(self.label_tex)
        else:
            self.label_tex = None


class LabeledBox(VGroup):
    """Rounded rectangle with a math label — used for weight matrices like W_{ih}, W_{hh}.

    Args:
        label: Математическая метка (MathTex).
        width: Ширина прямоугольника.
        height: Высота прямоугольника.
        corner_radius: Радиус закругления углов.
        color: Цвет обводки.
        fill_opacity: Прозрачность заливки (0..1).
        label_scale: Масштаб MathTex-метки (по умолчанию 0.6 для совместимости).
        **kwargs: Проброс в VGroup.
    """

    def __init__(
        self,
        label: str,
        width: float = 1.2,
        height: float = 0.7,
        corner_radius: float = 0.12,
        color: str = GREY_B,
        fill_opacity: float = 0.1,
        label_scale: float = 0.6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.box = RoundedRectangle(
            corner_radius=corner_radius,
            width=width,
            height=height,
            color=color,
        ).set_fill(color, opacity=fill_opacity)
        self.add(self.box)
        self.label_tex = MathTex(label).scale(label_scale)
        self.label_tex.move_to(self.box.get_center())
        self.add(self.label_tex)


def arrow_between(a: VMobject, b: VMobject, **kwargs: Any) -> Arrow:
    """Построить стрелку от края ``a`` к краю ``b``.

    Типизированный враппер над manim.Arrow, использующий координаты границ
    переданных объектов (а не центров), чтобы стрелки аккуратно прилипали
    к внешним контурам.

    Args:
        a: Объект-источник стрелки.
        b: Объект-приёмник стрелки.
        **kwargs: Проброс в Arrow (buff, stroke_width, color, и т.д.).

    Returns:
        Arrow из a в b.
    """
    defaults: dict[str, Any] = {"buff": 0.1, "stroke_width": 3, "color": WHITE}
    defaults.update(kwargs)
    dx = abs(b.get_center()[0] - a.get_center()[0])
    dy = abs(b.get_center()[1] - a.get_center()[1])
    if dy > dx:
        # Vertical relationship: attach top/bottom edges.
        start = a.get_top() if a.get_center()[1] <= b.get_center()[1] else a.get_bottom()
        end = b.get_bottom() if a.get_center()[1] <= b.get_center()[1] else b.get_top()
    else:
        # Horizontal relationship: attach left/right edges.
        start = a.get_right() if a.get_center()[0] <= b.get_center()[0] else a.get_left()
        end = b.get_left() if a.get_center()[0] <= b.get_center()[0] else b.get_right()
    return Arrow(start=start, end=end, **defaults)
