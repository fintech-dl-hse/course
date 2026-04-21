"""Unit tests for the KeyframeRecorder mixin and record_keyframes decorator.

Тесты не требуют реального рендера Manim — используется FakeScene,
имитирующая только интерфейс, затрагиваемый миксином.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
SHARED_DIR = HERE.parent / "shared"
sys.path.insert(0, str(HERE.parent))

import shared.keyframes as keyframes_mod  # noqa: E402
from shared.keyframes import KeyframeRecorder, record_keyframes  # noqa: E402


# ---------------------------------------------------------------------------
# Fake infrastructure — no real Manim renderer required
# ---------------------------------------------------------------------------


class FakeRenderer:
    """Minimal renderer stub exposing only ``time``."""

    def __init__(self, time: float = 0.0) -> None:
        self.time = time


class FakeScene:
    """Minimal Scene stub — mimics only the surface touched by KeyframeRecorder."""

    def __init__(self) -> None:
        self.renderer = FakeRenderer()
        self._played: list[Any] = []
        self._waited: list[Any] = []

    def setup(self) -> None:
        pass

    def play(self, *args: Any, **kwargs: Any) -> None:
        self._played.append(args)

    def wait(self, *args: Any, **kwargs: Any) -> None:
        self._waited.append(args)

    def tear_down(self) -> None:
        pass


class FakeAnimation:
    """Stub animation whose class name should appear in events."""


class MixedScene(KeyframeRecorder, FakeScene):
    """Concrete scene combining the mixin with FakeScene."""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mixin_records_play_end() -> None:
    """play() appends a play_end event with the animation class name."""
    scene = MixedScene()
    scene.setup()
    scene.renderer.time = 1.5
    scene.play(FakeAnimation())
    assert len(scene._kf_events) == 1
    evt = scene._kf_events[0]
    assert evt["kind"] == "play_end"
    assert evt["animation"] == "FakeAnimation"
    assert evt["t_seconds"] == pytest.approx(1.5)


def test_mixin_records_wait_end() -> None:
    """wait() appends a wait_end event with an empty animation name."""
    scene = MixedScene()
    scene.setup()
    scene.renderer.time = 2.3
    scene.wait(0.5)
    assert len(scene._kf_events) == 1
    evt = scene._kf_events[0]
    assert evt["kind"] == "wait_end"
    assert evt["animation"] == ""
    assert evt["t_seconds"] == pytest.approx(2.3)


def test_sidecar_written_on_teardown(tmp_path: Path) -> None:
    """tear_down() writes a valid sidecar JSON with the correct schema."""
    with patch.object(keyframes_mod, "_default_sidecar_dir", return_value=tmp_path / "keyframes"):
        scene = MixedScene()
        scene.setup()

        scene.renderer.time = 1.0
        scene.play(FakeAnimation())

        scene.renderer.time = 2.0
        scene.wait(0.5)

        scene.tear_down()

    sidecar = tmp_path / "keyframes" / "MixedScene.json"
    assert sidecar.exists(), f"sidecar not found at {sidecar}"

    data = json.loads(sidecar.read_text())
    assert data["scene"] == "MixedScene"
    assert isinstance(data["events"], list)
    assert len(data["events"]) == 2

    for evt in data["events"]:
        assert "t_seconds" in evt
        assert "kind" in evt
        assert "animation" in evt

    assert data["events"][0]["kind"] == "play_end"
    assert data["events"][0]["animation"] == "FakeAnimation"
    assert data["events"][1]["kind"] == "wait_end"
    assert data["events"][1]["animation"] == ""


def test_decorator_preserves_class_name() -> None:
    """record_keyframes preserves __name__ and __qualname__ of the decorated class."""

    @record_keyframes
    class Original(FakeScene):
        pass

    assert Original.__name__ == "Original"
    assert "Original" in Original.__qualname__


def test_opt_out_is_noop() -> None:
    """A plain FakeScene without the mixin doesn't have _kf_events and doesn't blow up."""
    scene = FakeScene()
    scene.setup()
    scene.play(FakeAnimation())
    scene.wait(0.5)
    scene.tear_down()
    assert not hasattr(scene, "_kf_events")
