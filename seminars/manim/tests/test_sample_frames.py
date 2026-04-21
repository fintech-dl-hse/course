"""Unit tests for scripts/sample_frames.py (midpoint + endpoint sampler).

All ffprobe/ffmpeg invocations are monkeypatched so no real video decode occurs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import sample_frames  # noqa: E402


def _make_dummy_video(tmp_path: Path, name: str = "test_scene.mp4") -> Path:
    """Create a zero-byte dummy .mp4 so Path.exists() passes."""
    video = tmp_path / name
    video.write_bytes(b"")
    return video


def _patch_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    captured: list[float],
    captured_scales: list[int] | None = None,
) -> None:
    def _fake_extract(
        video: Path, t: float, out_path: Path, downscale: int = 2
    ) -> None:
        captured.append(t)
        if captured_scales is not None:
            captured_scales.append(downscale)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x89PNG\r\n")

    monkeypatch.setattr(sample_frames, "_extract_frame", _fake_extract)


def test_extracts_two_frames_at_midpoint_and_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duration=10s → frames at 5.0s (50%) and 9.95s (duration - default offset)."""
    video = _make_dummy_video(tmp_path)
    out_dir = tmp_path / "frames_out"
    captured: list[float] = []

    monkeypatch.setattr(sample_frames, "_probe_duration", lambda v: 10.0)
    _patch_ffmpeg(monkeypatch, captured)

    rc = sample_frames.main([str(video), "--out-dir", str(out_dir)])

    assert rc == 0
    assert captured[0] == pytest.approx(5.0)
    assert captured[1] == pytest.approx(9.95)
    assert (out_dir / "frame_0001.png").exists()
    assert (out_dir / "frame_0002.png").exists()

    index = json.loads((out_dir / "frames_index.json").read_text())
    assert index["source"] == "midpoint_endpoint"
    assert index["duration_seconds"] == 10.0
    assert index["end_offset_seconds"] == 0.05
    assert index["downscale"] == 2
    assert len(index["frames"]) == 2
    assert index["frames"][0]["role"] == "midpoint"
    assert index["frames"][0]["t_seconds"] == pytest.approx(5.0)
    assert index["frames"][0]["path"] == "frame_0001.png"
    assert index["frames"][1]["role"] == "endpoint"
    assert index["frames"][1]["t_seconds"] == pytest.approx(9.95)
    assert index["frames"][1]["path"] == "frame_0002.png"


def test_custom_end_offset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--end-offset 0.5 → endpoint frame is at duration - 0.5."""
    video = _make_dummy_video(tmp_path)
    out_dir = tmp_path / "frames_out"
    captured: list[float] = []

    monkeypatch.setattr(sample_frames, "_probe_duration", lambda v: 4.0)
    _patch_ffmpeg(monkeypatch, captured)

    rc = sample_frames.main(
        [str(video), "--out-dir", str(out_dir), "--end-offset", "0.5"]
    )

    assert rc == 0
    assert captured[0] == pytest.approx(2.0)
    assert captured[1] == pytest.approx(3.5)


def test_clears_existing_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stale frames from a previous run must be removed before new extraction."""
    video = _make_dummy_video(tmp_path)
    out_dir = tmp_path / "frames_out"
    out_dir.mkdir()
    stale = out_dir / "frame_0099.png"
    stale.write_bytes(b"stale")

    monkeypatch.setattr(sample_frames, "_probe_duration", lambda v: 2.0)
    _patch_ffmpeg(monkeypatch, [])

    rc = sample_frames.main([str(video), "--out-dir", str(out_dir)])
    assert rc == 0
    assert not stale.exists()
    assert (out_dir / "frame_0001.png").exists()
    assert (out_dir / "frame_0002.png").exists()


def test_endpoint_clamped_to_zero_for_short_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """duration < end-offset must not produce a negative seek."""
    video = _make_dummy_video(tmp_path)
    out_dir = tmp_path / "frames_out"
    captured: list[float] = []

    monkeypatch.setattr(sample_frames, "_probe_duration", lambda v: 0.02)
    _patch_ffmpeg(monkeypatch, captured)

    rc = sample_frames.main([str(video), "--out-dir", str(out_dir)])
    assert rc == 0
    assert captured[1] == pytest.approx(0.0)


def test_downscale_flag_is_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--downscale 4 is forwarded to _extract_frame and recorded in index."""
    video = _make_dummy_video(tmp_path)
    out_dir = tmp_path / "frames_out"
    captured: list[float] = []
    scales: list[int] = []

    monkeypatch.setattr(sample_frames, "_probe_duration", lambda v: 8.0)
    _patch_ffmpeg(monkeypatch, captured, scales)

    rc = sample_frames.main(
        [str(video), "--out-dir", str(out_dir), "--downscale", "4"]
    )
    assert rc == 0
    assert scales == [4, 4]

    index = json.loads((out_dir / "frames_index.json").read_text())
    assert index["downscale"] == 4


def test_missing_video_returns_error(tmp_path: Path) -> None:
    rc = sample_frames.main([str(tmp_path / "missing.mp4"), "--out-dir", str(tmp_path)])
    assert rc == 2


def test_zero_duration_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_dummy_video(tmp_path)
    monkeypatch.setattr(sample_frames, "_probe_duration", lambda v: 0.0)
    rc = sample_frames.main([str(video), "--out-dir", str(tmp_path / "out")])
    assert rc == 3
