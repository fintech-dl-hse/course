"""Unit tests for the render_loop helper.

These tests drive `scripts/render_loop.py` by importing it and calling
`main(argv)` directly — this avoids a subprocess fork (faster) while still
using the script's real CLI parser.

Tests use synthetic `loop_state.json` and `critic_output.json` files in a
tmp dir, plus the `--_test-skip-render` and `--_test-video-hash` hooks that
render_loop exposes to bypass the real `make render` / ffmpeg invocation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import render_loop  # noqa: E402 — path injection above


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_state(tmp_path: Path, iterations: list[dict]) -> Path:
    path = tmp_path / "loop_state.json"
    path.write_text(json.dumps({"iterations": iterations}, indent=2))
    return path


def _write_critic(tmp_path: Path, name: str, obj: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(obj))
    return path


def _invoke(state_path: Path, critic_path: Path, *, max_iters: int = 5,
            test_video_hash: str = "a" * 64,
            test_repo_root: Path | None = None) -> int:
    """Call render_loop.main with the given state + critic JSON path."""
    argv = [
        "--scene", "TestScene",
        "--max-iters", str(max_iters),
        "--prev-state", str(state_path),
        "--critic-output", str(critic_path),
        "--_test-skip-render",
        "--_test-video-hash", test_video_hash,
    ]
    if test_repo_root is not None:
        argv += ["--_test-repo-root", str(test_repo_root)]
    return render_loop.main(argv)


HASH_A = "a" * 64
HASH_B = "b" * 64


# ---------------------------------------------------------------------------
# Tests — names are load-bearing for AC #11 grep.
# ---------------------------------------------------------------------------


def test_exit_code_on_approved(tmp_path: Path) -> None:
    """Iteration 1 with critic approved=true AND no high issues → exit 0."""
    state_path = _write_state(tmp_path, [])
    critic_path = _write_critic(
        tmp_path,
        "critic.json",
        {"approved": True, "video_hash": HASH_A, "issues": []},
    )
    code = _invoke(state_path, critic_path, test_video_hash=HASH_A,
                   test_repo_root=tmp_path)
    assert code == 0, f"expected 0 (approved), got {code}"


def test_exit_code_on_hash_stuck_from_iter_2(tmp_path: Path) -> None:
    """Iter 2 with same hash as iter 1 AND non-empty issues → exit 3."""
    # Pre-populate iter 1 with same hash.
    state_path = _write_state(tmp_path, [
        {"iteration": 1, "video_hash": HASH_A, "issues_count": 1,
         "approved": False, "malformed": False, "timestamp": 0.0},
    ])
    critic_path = _write_critic(
        tmp_path,
        "critic.json",
        {
            "approved": False,
            "video_hash": HASH_A,
            "issues": [{
                "frame": "00:01",
                "severity": "med",
                "category": "overlap",
                "description": "still overlapping",
                "suggested_fix": "shift W_ih",
            }],
        },
    )
    code = _invoke(state_path, critic_path, test_video_hash=HASH_A,
                   test_repo_root=tmp_path)
    assert code == 3, f"expected 3 (stuck), got {code}"


def test_exit_code_on_max_iters(tmp_path: Path) -> None:
    """Iteration == max_iters with issues → exit 4."""
    # Fill iterations 1..4 so the next one is iter 5 (== max_iters=5).
    prior = [
        {"iteration": i, "video_hash": f"{i:064x}", "issues_count": 1,
         "approved": False, "malformed": False, "timestamp": 0.0}
        for i in range(1, 5)
    ]
    state_path = _write_state(tmp_path, prior)
    critic_path = _write_critic(
        tmp_path,
        "critic.json",
        {
            "approved": False,
            "video_hash": HASH_B,
            "issues": [{
                "frame": "00:02",
                "severity": "med",
                "category": "offscreen",
                "description": "edge content",
                "suggested_fix": "shrink layout",
            }],
        },
    )
    code = _invoke(state_path, critic_path, max_iters=5,
                   test_video_hash=HASH_B, test_repo_root=tmp_path)
    assert code == 4, f"expected 4 (max-iters), got {code}"


def test_exit_code_on_new_hash_continues(tmp_path: Path) -> None:
    """Iter 2 with NEW hash vs iter 1 AND issues → exit 1 (continue)."""
    state_path = _write_state(tmp_path, [
        {"iteration": 1, "video_hash": HASH_A, "issues_count": 1,
         "approved": False, "malformed": False, "timestamp": 0.0},
    ])
    critic_path = _write_critic(
        tmp_path,
        "critic.json",
        {
            "approved": False,
            "video_hash": HASH_B,
            "issues": [{
                "frame": "00:04",
                "severity": "med",
                "category": "text-clip",
                "description": "clipped",
                "suggested_fix": "resize font",
            }],
        },
    )
    code = _invoke(state_path, critic_path, test_video_hash=HASH_B,
                   test_repo_root=tmp_path)
    assert code == 1, f"expected 1 (continue), got {code}"


def test_hash_stuck_suppressed_on_iter_1(tmp_path: Path) -> None:
    """Iteration 1 MUST NOT fire hash-stuck even with issues."""
    state_path = _write_state(tmp_path, [])
    critic_path = _write_critic(
        tmp_path,
        "critic.json",
        {
            "approved": False,
            "video_hash": HASH_A,
            "issues": [{
                "frame": "00:00",
                "severity": "med",
                "category": "overlap",
                "description": "first-iter issue",
                "suggested_fix": "nudge",
            }],
        },
    )
    code = _invoke(state_path, critic_path, test_video_hash=HASH_A,
                   test_repo_root=tmp_path)
    assert code == 1, f"expected 1 (continue), got {code} (iter-1 must not be stuck)"


def test_malformed_critic_yields_empty_issues(tmp_path: Path) -> None:
    """Malformed critic output → issues_count=0 this iteration; does NOT reuse prior.

    With empty issues and a new-enough hash, the verdict must be `continue`
    (exit 1) — not stuck, not approved.
    """
    # Prior iteration had real issues; we must NOT inherit them.
    state_path = _write_state(tmp_path, [
        {"iteration": 1, "video_hash": HASH_A, "issues_count": 2,
         "approved": False, "malformed": False, "timestamp": 0.0},
    ])
    # Intentionally malformed: missing required fields.
    malformed_path = tmp_path / "critic.json"
    malformed_path.write_text('{"approved": "maybe"}')  # wrong type + missing keys

    code = _invoke(state_path, malformed_path, test_video_hash=HASH_B,
                   test_repo_root=tmp_path)
    assert code == 1, f"expected 1 (continue on new hash, empty issues), got {code}"

    # Confirm the appended record has issues_count=0 and malformed=True.
    state = json.loads(state_path.read_text())
    last = state["iterations"][-1]
    assert last["iteration"] == 2
    assert last["issues_count"] == 0, f"issues must not be inherited from prior: {last}"
    assert last["malformed"] is True
    assert last["approved"] is False
