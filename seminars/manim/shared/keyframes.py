"""Keyframe recording mixin and decorator for ManimCE scenes.

Записывает временные метки ключевых кадров (play / wait) в JSON-файл-«спутник»
рядом с рендером. Используется для автоматической проверки анимаций.

Keyframe recording mixin and decorator for ManimCE scenes. Records timestamps
of key animation events (play / wait) into a sidecar JSON file. Used for
automated animation review pipelines.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _default_sidecar_dir() -> Path:
    """Return the default directory for keyframe sidecar files.

    Возвращает путь по умолчанию для JSON-файлов с ключевыми кадрами:
    ``seminars/manim/.out/keyframes/``.

    Override in tests by monkeypatching this function in the ``shared.keyframes``
    module namespace.
    """
    return Path(__file__).resolve().parent.parent / ".out" / "keyframes"


class KeyframeRecorder:
    """Mixin that records play/wait events to a sidecar JSON file.

    Миксин для записи событий ``play`` / ``wait`` в JSON-файл рядом с рендером.
    Должен стоять ПЕРЕД ``Scene`` в MRO:

    .. code-block:: python

        class MyScene(KeyframeRecorder, Scene):
            ...

    После завершения сцены создаётся файл
    ``seminars/manim/.out/keyframes/<ClassName>.json`` со схемой::

        {
          "scene": "<ClassName>",
          "events": [
            {"t_seconds": 1.2, "kind": "play_end", "animation": "Write"},
            {"t_seconds": 2.4, "kind": "wait_end", "animation": ""}
          ]
        }
    """

    def setup(self) -> None:  # type: ignore[override]
        """Инициализирует аккумулятор событий перед стартом сцены."""
        super().setup()  # type: ignore[misc]
        self._kf_events: list[dict[str, Any]] = []

    def play(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """Воспроизводит анимацию и записывает событие ``play_end``."""
        super().play(*args, **kwargs)  # type: ignore[misc]
        animation_name = args[0].__class__.__name__ if args else ""
        self._kf_events.append(
            {
                "t_seconds": float(self.renderer.time),  # type: ignore[attr-defined]
                "kind": "play_end",
                "animation": animation_name,
            }
        )

    def wait(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """Выдерживает паузу и записывает событие ``wait_end``."""
        super().wait(*args, **kwargs)  # type: ignore[misc]
        self._kf_events.append(
            {
                "t_seconds": float(self.renderer.time),  # type: ignore[attr-defined]
                "kind": "wait_end",
                "animation": "",
            }
        )

    def tear_down(self) -> None:  # type: ignore[override]
        """Записывает сайдкар-файл перед завершением сцены.

        ManimCE Scene defines the lifecycle hook as ``tear_down`` (snake_case),
        not ``tearDown``. See manim/scene/scene.py:268 which calls
        ``self.tear_down()`` during ``render()``.
        """
        scene_name = type(self).__name__
        sidecar_path = _default_sidecar_dir() / f"{scene_name}.json"
        try:
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_text(
                json.dumps(
                    {"scene": scene_name, "events": self._kf_events},
                    indent=2,
                )
            )
        except OSError as exc:
            print(f"[KeyframeRecorder] WARNING: could not write sidecar {sidecar_path}: {exc}")
        super().tear_down()  # type: ignore[misc]


def record_keyframes(cls: type) -> type:
    """Class decorator that mixes :class:`KeyframeRecorder` into *cls*.

    Декоратор-класс, добавляющий :class:`KeyframeRecorder` в MRO:

    .. code-block:: python

        @record_keyframes
        class MyScene(Scene):
            ...

    Сохраняет ``__name__`` и ``__qualname__`` исходного класса.
    """
    new_cls = type(cls.__name__, (KeyframeRecorder, cls), {})
    new_cls.__qualname__ = cls.__qualname__
    return new_cls
