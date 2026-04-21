"""Unit tests for scripts/sample_frames.py.

Tests use tmp_path fixture and monkeypatching to avoid real video decode.
All subprocess.run calls are mocked so no real ffmpeg invocation occurs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import sample_frames  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_video(tmp_path: Path) -> Path:
    """Create a zero-byte dummy .mp4 so Path.exists() passes."""
    video = tmp_path / "test_scene.mp4"
    video.write_bytes(b"")
    return video


def _make_sidecar(tmp_path: Path, scene: str, events: list[dict]) -> Path:
    """Write a keyframe sidecar JSON into tmp_path/.out/keyframes/."""
    sidecar_dir = tmp_path / ".out" / "keyframes"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar = sidecar_dir / f"{scene}.json"
    sidecar.write_text(json.dumps({"scene": scene, "events": events}))
    return sidecar


def _fake_ffmpeg_ok(*args: Any, **kwargs: Any) -> SimpleNamespace:
    """Fake subprocess.run that succeeds and creates the output file."""
    cmd: list[str] = args[0]
    # Find the output path — last argument in the ffmpeg command.
    out_path = Path(cmd[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"\x89PNG\r\n")  # minimal fake PNG header
    return SimpleNamespace(returncode=0, stderr="", stdout="")


# ---------------------------------------------------------------------------
# test_hook_path_selects_sidecar
# ---------------------------------------------------------------------------


def test_hook_path_selects_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sidecar present with valid events → frames_index.json has source='hook'."""
    video = _make_dummy_video(tmp_path)
    _make_sidecar(
        tmp_path,
        "test_scene",
        [
            {"t_seconds": 1.0, "kind": "play_end", "animation": "Write"},
            {"t_seconds": 2.5, "kind": "wait_end", "animation": ""},
        ],
    )

    out_dir = tmp_path / "frames_out"

    # Patch __file__ so that base resolves to tmp_path (where we put the sidecar).
    monkeypatch.setattr(sample_frames, "__file__", str(tmp_path / "scripts" / "sample_frames.py"))
    monkeypatch.setattr("subprocess.run", _fake_ffmpeg_ok)

    rc = sample_frames.main([str(video), "--out-dir", str(out_dir)])

    assert rc == 0
    index_path = out_dir / "frames_index.json"
    assert index_path.exists(), "frames_index.json not written"
    index = json.loads(index_path.read_text())
    assert index["source"] == "hook"
    assert index["scene"] == "test_scene"
    assert index["pre_hook_event_count"] == 2
    assert index["post_dedupe_event_count"] == 2
    assert len(index["keyframes"]) == 2
    assert index["keyframes"][0]["t_seconds"] == 1.0
    assert index["keyframes"][0]["kind"] == "play_end"
    assert index["keyframes"][1]["t_seconds"] == 2.5
    # pHash fields must NOT be present in hook mode.
    assert "phash_threshold" not in index
    assert "dropped" not in index


# ---------------------------------------------------------------------------
# test_fallback_path_when_sidecar_missing
# ---------------------------------------------------------------------------


def test_fallback_path_when_sidecar_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No sidecar → pHash branch; frames_index.json has source='phash_fallback'."""
    video = _make_dummy_video(tmp_path)
    out_dir = tmp_path / "frames_out"

    # Point base to tmp_path so no sidecar will be found there.
    monkeypatch.setattr(sample_frames, "__file__", str(tmp_path / "scripts" / "sample_frames.py"))

    # Mock _run_ffmpeg_extract to drop dummy PNGs into the tmp raw dir.
    def _fake_extract(v: Path, tmp_raw: Path) -> None:
        tmp_raw.mkdir(parents=True, exist_ok=True)
        for i in range(1, 4):
            (tmp_raw / f"frame_{i:04d}.png").write_bytes(b"\x89PNG\r\n")

    # Mock _dedupe to return all frames as kept.
    def _fake_dedupe(frames: list[Path], threshold: int):
        return list(frames), []

    monkeypatch.setattr(sample_frames, "_run_ffmpeg_extract", _fake_extract)
    monkeypatch.setattr(sample_frames, "_dedupe", _fake_dedupe)

    rc = sample_frames.main([str(video), "--out-dir", str(out_dir)])

    assert rc == 0
    index = json.loads((out_dir / "frames_index.json").read_text())
    assert index["source"] == "phash_fallback"
    assert "pre_dedupe_count" in index
    assert "post_dedupe_count" in index
    assert "phash_threshold" in index
    assert "dropped" in index


# ---------------------------------------------------------------------------
# test_force_phash_overrides_sidecar
# ---------------------------------------------------------------------------


def test_force_phash_overrides_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--force-phash flag bypasses hook path even when sidecar exists."""
    video = _make_dummy_video(tmp_path)
    _make_sidecar(
        tmp_path,
        "test_scene",
        [{"t_seconds": 1.0, "kind": "play_end", "animation": "Write"}],
    )
    out_dir = tmp_path / "frames_out"

    monkeypatch.setattr(sample_frames, "__file__", str(tmp_path / "scripts" / "sample_frames.py"))

    def _fake_extract(v: Path, tmp_raw: Path) -> None:
        tmp_raw.mkdir(parents=True, exist_ok=True)
        (tmp_raw / "frame_0001.png").write_bytes(b"\x89PNG\r\n")

    def _fake_dedupe(frames: list[Path], threshold: int):
        return list(frames), []

    monkeypatch.setattr(sample_frames, "_run_ffmpeg_extract", _fake_extract)
    monkeypatch.setattr(sample_frames, "_dedupe", _fake_dedupe)

    rc = sample_frames.main([str(video), "--out-dir", str(out_dir), "--force-phash"])

    assert rc == 0
    index = json.loads((out_dir / "frames_index.json").read_text())
    assert index["source"] == "phash_fallback", (
        "--force-phash must route to phash_fallback, got: " + index["source"]
    )


# ---------------------------------------------------------------------------
# test_hook_skips_zero_timestamp
# ---------------------------------------------------------------------------


def test_hook_skips_zero_timestamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sidecar with only t_seconds=0 events → falls through to pHash path."""
    video = _make_dummy_video(tmp_path)
    _make_sidecar(
        tmp_path,
        "test_scene",
        [
            {"t_seconds": 0, "kind": "play_end", "animation": "Write"},
            {"t_seconds": 0.0, "kind": "wait_end", "animation": ""},
        ],
    )
    out_dir = tmp_path / "frames_out"

    monkeypatch.setattr(sample_frames, "__file__", str(tmp_path / "scripts" / "sample_frames.py"))

    def _fake_extract(v: Path, tmp_raw: Path) -> None:
        tmp_raw.mkdir(parents=True, exist_ok=True)
        (tmp_raw / "frame_0001.png").write_bytes(b"\x89PNG\r\n")

    def _fake_dedupe(frames: list[Path], threshold: int):
        return list(frames), []

    monkeypatch.setattr(sample_frames, "_run_ffmpeg_extract", _fake_extract)
    monkeypatch.setattr(sample_frames, "_dedupe", _fake_dedupe)

    rc = sample_frames.main([str(video), "--out-dir", str(out_dir)])

    assert rc == 0
    index = json.loads((out_dir / "frames_index.json").read_text())
    assert index["source"] == "phash_fallback", (
        "all-zero timestamps must fall through to phash_fallback, got: " + index["source"]
    )


# ---------------------------------------------------------------------------
# test_hook_dedupes_identical_timestamps
# ---------------------------------------------------------------------------


def test_hook_collapses_tight_clusters_by_min_dt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Events spaced closer than --hook-min-dt collapse to the first of each cluster."""
    video = _make_dummy_video(tmp_path)
    # Cluster at [1.0, 1.3, 1.6] then gap to [3.0, 3.35], then [5.0]. With
    # min_dt=0.75 we expect: 1.0 kept, 1.3/1.6 dropped (within 0.75 of last kept),
    # 3.0 kept (gap 2.0 ≥ 0.75), 3.35 dropped, 5.0 kept.
    _make_sidecar(
        tmp_path,
        "test_scene",
        [
            {"t_seconds": 1.0, "kind": "play_end", "animation": "A"},
            {"t_seconds": 1.3, "kind": "play_end", "animation": "B"},
            {"t_seconds": 1.6, "kind": "play_end", "animation": "C"},
            {"t_seconds": 3.0, "kind": "play_end", "animation": "D"},
            {"t_seconds": 3.35, "kind": "play_end", "animation": "E"},
            {"t_seconds": 5.0, "kind": "wait_end", "animation": ""},
        ],
    )
    out_dir = tmp_path / "frames_out"

    monkeypatch.setattr(sample_frames, "__file__", str(tmp_path / "scripts" / "sample_frames.py"))
    monkeypatch.setattr("subprocess.run", _fake_ffmpeg_ok)

    rc = sample_frames.main([str(video), "--out-dir", str(out_dir), "--hook-min-dt", "0.75"])

    assert rc == 0
    index = json.loads((out_dir / "frames_index.json").read_text())
    assert index["source"] == "hook"
    timestamps = [kf["t_seconds"] for kf in index["keyframes"]]
    assert timestamps == [1.0, 3.0, 5.0], f"expected cluster collapse, got {timestamps}"
    assert index["pre_hook_event_count"] == 6
    assert index["post_dedupe_event_count"] == 3


def test_hook_min_dt_zero_preserves_distinct_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--hook-min-dt=0 disables cluster collapse but still dedupes identical timestamps."""
    video = _make_dummy_video(tmp_path)
    _make_sidecar(
        tmp_path,
        "test_scene",
        [
            {"t_seconds": 1.0, "kind": "play_end", "animation": "A"},
            {"t_seconds": 1.1, "kind": "play_end", "animation": "B"},  # close, but kept at min_dt=0
            {"t_seconds": 1.1, "kind": "wait_end", "animation": ""},    # identical → dropped
        ],
    )
    out_dir = tmp_path / "frames_out"

    monkeypatch.setattr(sample_frames, "__file__", str(tmp_path / "scripts" / "sample_frames.py"))
    monkeypatch.setattr("subprocess.run", _fake_ffmpeg_ok)

    rc = sample_frames.main([str(video), "--out-dir", str(out_dir), "--hook-min-dt", "0"])

    assert rc == 0
    index = json.loads((out_dir / "frames_index.json").read_text())
    timestamps = [kf["t_seconds"] for kf in index["keyframes"]]
    assert timestamps == [1.0, 1.1]


def test_hook_dedupes_identical_timestamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sidecar with 5 events but 3 share a timestamp → only 3 distinct kept."""
    video = _make_dummy_video(tmp_path)
    _make_sidecar(
        tmp_path,
        "test_scene",
        [
            {"t_seconds": 1.0, "kind": "play_end", "animation": "A"},
            {"t_seconds": 2.0, "kind": "play_end", "animation": "B"},
            {"t_seconds": 2.0, "kind": "wait_end", "animation": ""},   # dup
            {"t_seconds": 2.0, "kind": "play_end", "animation": "C"},  # dup
            {"t_seconds": 3.0, "kind": "wait_end", "animation": ""},
        ],
    )
    out_dir = tmp_path / "frames_out"

    monkeypatch.setattr(sample_frames, "__file__", str(tmp_path / "scripts" / "sample_frames.py"))
    monkeypatch.setattr("subprocess.run", _fake_ffmpeg_ok)

    rc = sample_frames.main([str(video), "--out-dir", str(out_dir)])

    assert rc == 0
    index = json.loads((out_dir / "frames_index.json").read_text())
    assert index["source"] == "hook"
    assert index["pre_hook_event_count"] == 5
    assert index["post_dedupe_event_count"] == 3, (
        f"expected 3 distinct timestamps, got {index['post_dedupe_event_count']}"
    )
    timestamps = [kf["t_seconds"] for kf in index["keyframes"]]
    assert timestamps == [1.0, 2.0, 3.0]
