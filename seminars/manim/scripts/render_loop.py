#!/usr/bin/env python3
"""Per-iteration helper for the ralph-driven render/critique/fix loop.

The ralph LEADER owns orchestration (it is the only layer allowed to spawn
subagents). This script is a deterministic per-iteration step:

Invocation modes
----------------
1. Render mode   — called without --critic-output (or with empty value):
   runs `make render` for the scene, samples + dedupes frames, hashes the
   video, writes pending state to `.out/loop_state.json.new`, exits 2
   (critic-needed). The leader then spawns the manim-frame-critic subagent
   and re-invokes this helper with --critic-output.

2. Verdict mode  — called with --critic-output <path>:
   validates the critic JSON, appends an iteration record to
   `.out/loop_state.json`, computes and returns the stop verdict.

Exit codes
----------
    0 — approved (critic JSON has approved=true AND no 'high' severity)
    1 — continue (render new iteration with fix)
    2 — critic-needed (after render+sample+hash, before critic invoked)
    3 — stuck (hash unchanged iter N vs N-1 with non-empty issues, iter>=2)
    4 — max-iters reached

First-iteration rule: hash-stuck is suppressed on iteration 1.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

MANIM_ROOT = Path(__file__).resolve().parent.parent  # seminars/manim/

EXIT_APPROVED = 0
EXIT_CONTINUE = 1
EXIT_CRITIC_NEEDED = 2
EXIT_STUCK = 3
EXIT_MAX_ITERS = 4


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _load_state(path: Path) -> dict[str, Any]:
    """Read loop state, returning a fresh skeleton when absent or corrupt."""
    if not path.exists():
        return {"iterations": []}
    try:
        obj = json.loads(path.read_text())
        if not isinstance(obj, dict) or "iterations" not in obj:
            return {"iterations": []}
        return obj
    except json.JSONDecodeError:
        return {"iterations": []}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Git-diff guard
# ---------------------------------------------------------------------------


def _git_diff_guard(repo_root: Path) -> Optional[str]:
    """Return None if clean outside the allowed scope, else an error string.

    Allowed paths for fix-step executor edits:
        - seminars/manim/**
        - .claude/agents/manim-frame-critic.md
    """
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        # git not available: skip the guard
        return None
    if proc.returncode != 0:
        return None  # non-fatal; not a git repo
    offenders: list[str] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        # porcelain format: "XY path"
        path = line[3:].strip()
        if path.startswith("seminars/manim/"):
            continue
        if path == ".claude/agents/manim-frame-critic.md":
            continue
        # Git porcelain collapses untracked dirs to their root (e.g. ".claude/agents/"
        # when only the critic file lives there). Accept that bare-dir form too,
        # as long as the sole intended addition is the critic file.
        if path == ".claude/agents/":
            continue
        offenders.append(path)
    if offenders:
        return "git-diff guard: changes outside allowed scope: " + ", ".join(offenders)
    return None


# ---------------------------------------------------------------------------
# Render + sample + hash (render mode)
# ---------------------------------------------------------------------------


def _run_render(scene: str) -> None:
    """Invoke `make render SCENE=<scene>` from seminars/manim/."""
    proc = subprocess.run(
        ["make", "render", f"SCENE={scene}"],
        cwd=str(MANIM_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"make render SCENE={scene} failed with exit {proc.returncode}")


def _run_sample(scene: str) -> None:
    video = MANIM_ROOT / ".out" / f"{scene}.mp4"
    proc = subprocess.run(
        [sys.executable, str(MANIM_ROOT / "scripts" / "sample_frames.py"), str(video)],
        cwd=str(MANIM_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"sample_frames.py failed with exit {proc.returncode}")


def _hash_video_path(video: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with video.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Critic validation (verdict mode)
# ---------------------------------------------------------------------------


def _validate_critic(critic_path: Path) -> tuple[bool, dict[str, Any]]:
    """Run validate_critic.py as subprocess; return (is_valid, parsed_obj_or_empty).

    On malformed output (validator rejects OR JSON parse fails) returns
    (False, {}) so caller treats this iteration's issues as empty. The
    validator's stdout/stderr is echoed so ralph operators keep the
    {ok:false, error:...} payload for debugging.
    """
    validator = MANIM_ROOT / "scripts" / "validate_critic.py"
    proc = subprocess.run(
        [sys.executable, str(validator), str(critic_path)],
        cwd=str(MANIM_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        if proc.stdout:
            print(f"[render_loop] validator stdout: {proc.stdout.strip()}", file=sys.stderr)
        if proc.stderr:
            print(f"[render_loop] validator stderr: {proc.stderr.strip()}", file=sys.stderr)
        return False, {}
    try:
        return True, json.loads(critic_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False, {}


# ---------------------------------------------------------------------------
# Verdict computation
# ---------------------------------------------------------------------------


def compute_verdict(
    state: dict[str, Any],
    current: dict[str, Any],
    max_iters: int,
) -> int:
    """Given the state (with current appended) and the current iteration,
    compute the stop-verdict exit code.
    """
    iteration: int = current["iteration"]
    approved: bool = bool(current.get("approved", False))
    issues_count: int = int(current.get("issues_count", 0))
    malformed: bool = bool(current.get("malformed", False))
    video_hash: str = str(current.get("video_hash", ""))

    # Approved rule: only counts if critic JSON is well-formed (malformed always continues).
    if approved and not malformed:
        # Also enforce: approved=true forbidden if high issues; caller already zeroed those.
        return EXIT_APPROVED

    # Hash-stuck: only from iteration >= 2.
    if iteration >= 2:
        prev_iter = None
        for it in reversed(state.get("iterations", [])[:-1]):
            prev_iter = it
            break
        if (
            prev_iter is not None
            and prev_iter.get("video_hash") == video_hash
            and issues_count > 0
        ):
            return EXIT_STUCK

    # Max-iters hit?
    if iteration >= max_iters:
        return EXIT_MAX_ITERS

    return EXIT_CONTINUE


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _compute_next_iteration(state: dict[str, Any]) -> int:
    iters = state.get("iterations", [])
    if not iters:
        return 1
    return int(iters[-1].get("iteration", 0)) + 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Per-iteration render/critic loop helper.")
    parser.add_argument("--scene", required=True, help="Scene class name (e.g. RNNUnroll)")
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument(
        "--prev-state",
        type=Path,
        default=MANIM_ROOT / ".out" / "loop_state.json",
        help="Path to loop_state.json (read+append)",
    )
    parser.add_argument(
        "--critic-output",
        type=str,
        default="",
        help="Path to critic JSON. If empty, runs render+sample+hash and exits 2.",
    )
    # Test hooks — let unit tests bypass real make/ffmpeg invocations.
    parser.add_argument("--_test-skip-render", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_test-video-hash", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_test-repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    state_path: Path = args.prev_state
    state = _load_state(state_path)

    # ------------------------------------------------------------------
    # Render mode (no critic output yet): render, sample, hash, write new state, exit 2.
    # ------------------------------------------------------------------
    if not args.critic_output:
        video_path = MANIM_ROOT / ".out" / f"{args.scene}.mp4"
        if not args._test_skip_render:
            _run_render(args.scene)
            _run_sample(args.scene)
            video_hash = _hash_video_path(video_path)
        else:
            video_hash = args._test_video_hash or "0" * 64

        pending = {
            "iteration": _compute_next_iteration(state),
            "video_hash": video_hash,
            "frames_dir": str(
                (MANIM_ROOT / ".out" / "frames" / args.scene).resolve()
            ),
            "timestamp": time.time(),
            "critic_pending": True,
        }
        pending_path = state_path.with_suffix(state_path.suffix + ".new")
        pending_path.parent.mkdir(parents=True, exist_ok=True)

        # git-diff guard before writing the pending marker: on offense, we
        # must not leave a stale .new file for a later verdict-mode call.
        repo_root = args._test_repo_root or MANIM_ROOT.parent.parent
        guard_err = _git_diff_guard(Path(repo_root))
        if guard_err is not None:
            pending_path.unlink(missing_ok=True)
            print(f"[render_loop] ABORT: {guard_err}", file=sys.stderr)
            return 10  # protected-path abort
        pending_path.write_text(json.dumps(pending, indent=2))
        return EXIT_CRITIC_NEEDED

    # ------------------------------------------------------------------
    # Verdict mode: validate critic JSON, append iteration, compute verdict.
    # ------------------------------------------------------------------
    critic_path = Path(args.critic_output)
    valid, parsed = _validate_critic(critic_path)

    if not valid:
        # Malformed critic output → empty issues THIS iteration only; do NOT reuse prior.
        approved = False
        issues_count = 0
        video_hash = args._test_video_hash or ""
        # Attempt to recover video_hash from .out/loop_state.json.new if present.
        pending_path = state_path.with_suffix(state_path.suffix + ".new")
        if not video_hash and pending_path.exists():
            try:
                pending_obj = json.loads(pending_path.read_text())
                video_hash = str(pending_obj.get("video_hash", ""))
            except (OSError, json.JSONDecodeError):
                pass
        malformed = True
    else:
        approved = bool(parsed.get("approved", False))
        issues_list = parsed.get("issues", []) or []
        # approved=true is forbidden if any high severity issue present.
        if approved and any(i.get("severity") == "high" for i in issues_list):
            approved = False
        issues_count = len(issues_list)
        video_hash = str(parsed.get("video_hash", "")) or (args._test_video_hash or "")
        malformed = False

    iteration = _compute_next_iteration(state)
    current = {
        "iteration": iteration,
        "video_hash": video_hash,
        "issues_count": issues_count,
        "approved": approved,
        "malformed": malformed,
        "timestamp": time.time(),
    }
    state.setdefault("iterations", []).append(current)
    _save_state(state_path, state)

    verdict = compute_verdict(state, current, args.max_iters)

    # git-diff guard: abort on any non-approved exit if changes leak outside allowed scope.
    if verdict != EXIT_APPROVED:
        repo_root = args._test_repo_root or MANIM_ROOT.parent.parent
        guard_err = _git_diff_guard(Path(repo_root))
        if guard_err is not None:
            print(f"[render_loop] ABORT: {guard_err}", file=sys.stderr)
            return 10

    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
