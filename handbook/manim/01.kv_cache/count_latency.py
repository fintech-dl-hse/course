#!/usr/bin/env python3
"""
Count per-token generation latency for large prefix lengths with and without KV-Cache.

Notes
- We measure the attention work per layer for a single generated token at prefix length n:
  S = Q @ K  (shape [h, n]) and O = softmax(S) @ V (shape [h, d]).
  This captures the part affected by KV-Cache. MLP and other ops are omitted.
- Total per-token attention time is (per-layer time) * num_layers.
- Without KV-Cache, recomputation implies an O(n^2) cost. We estimate that time by fitting
  a linear model for the measured per-layer KV time versus n, then summing across t=1..n.
  Specifically, if time_per_layer_kv(n) ≈ a * n + b, then time_per_layer_no_kv(n)
  ≈ a * n(n+1)/2 + b * n.

The script performs 10 trials for stability and writes a JSON report next to this file.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple


def try_import_torch():
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        return torch, F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to run this benchmark.") from exc


def pick_device(user_device: str) -> str:
    torch, _F = try_import_torch()
    if user_device:
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def map_dtype(dtype_str: str, device: str):
    torch, _F = try_import_torch()
    s = (dtype_str or "").lower()
    if device == "cpu" and s in ("float16", "fp16", "bfloat16", "bf16"):
        # Fallback to fp32 on CPU for stability/perf
        return torch.float32
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float32", "fp32", ""):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def synchronize(device: str) -> None:
    torch, _F = try_import_torch()
    if device == "cuda":
        torch.cuda.synchronize()


def load_default_dims_from_coeffs(base_dir: Path) -> Tuple[int, int, int]:
    """Try to load dims from kv_cache_coeffs_llama3.2-3B.json; fallback to sensible defaults.

    Returns (hidden_size, num_layers, num_attention_heads).
    """
    default_dims = (3072, 28, 24)  # Llama 3.2 3B-like
    coeffs_path = base_dir / "kv_cache_coeffs_llama3.2-3B.json"
    try:
        with open(coeffs_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        meta = payload.get("meta", {})
        hidden_size = int(meta.get("hidden_size", default_dims[0]))
        num_layers = int(meta.get("num_layers", default_dims[1]))
        num_heads = int(meta.get("num_attention_heads", default_dims[2]))
        return hidden_size, num_layers, num_heads
    except Exception:
        return default_dims


def measure_attention_per_layer_ms(
    *,
    n_tokens: int,
    hidden_size: int,
    num_attention_heads: int,
    device: str,
    torch_dtype,
    warmup: int,
    iters: int,
) -> List[float]:
    """Measure per-layer attention time (ms) for a single generated token at prefix length n.

    Performs iters timed runs (after warmup) of two GEMMs + softmax using shapes:
      Q: [h, d], K: [d, n], scores: [h, n], V: [n, d], out: [h, d]
    Returns a list of length iters with per-layer times in ms.
    """
    torch, F = try_import_torch()
    h = int(num_attention_heads)
    d = int(hidden_size // num_attention_heads)

    # Allocate once, reuse across runs
    gen = torch.Generator(device=device)
    gen.manual_seed(42)

    Q = torch.randn((h, d), dtype=torch_dtype, device=device, generator=gen)
    K = torch.randn((d, n_tokens), dtype=torch_dtype, device=device, generator=gen)
    V = torch.randn((n_tokens, d), dtype=torch_dtype, device=device, generator=gen)

    # Warmup
    for _ in range(max(0, warmup)):
        synchronize(device)
        _scores = Q @ K  # [h, n]
        _probs = F.softmax(_scores, dim=-1)
        _out = _probs @ V  # [h, d]

    # Timed runs
    times_ms: List[float] = []
    for _ in range(iters):
        synchronize(device)
        t0 = time.perf_counter()
        _scores = Q @ K
        _probs = F.softmax(_scores, dim=-1)
        _out = _probs @ V
        synchronize(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)
    return times_ms


def fit_linear(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Least squares fit y ≈ a*x + b. Returns (a, b)."""
    n = float(len(xs))
    if n < 2:
        return 0.0, ys[0] if ys else 0.0
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        return 0.0, sy / n
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return a, b


def estimate_no_kv_time_ms_from_fit(n_tokens: int, a: float, b: float) -> float:
    """Given per-layer KV time model y ≈ a*n + b (in ms), estimate per-layer no-KV time:
    sum_{t=1..n} (a*t + b) = a * n*(n+1)/2 + b * n (in ms).
    """
    n = float(n_tokens)
    return a * n * (n + 1.0) / 2.0 + b * n


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure per-token generation latency with/without KV-Cache.")
    parser.add_argument("--device", type=str, default="", help="cuda|cpu (auto if empty)")
    parser.add_argument("--dtype", type=str, default="float16", help="float16|bfloat16|float32")
    parser.add_argument("--hidden-size", type=int, default=None, help="Model hidden size (d_model)")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--num-attention-heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--prefix-lengths", type=int, nargs="*", default=[10_000, 100_000, 250_000], help="Prefix lengths to test")
    parser.add_argument("--trials", type=int, default=10, help="Trials per measurement")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs before timing (per measurement)")
    parser.add_argument("--calib-lengths", type=int, nargs="*", default=[4096, 8192, 16384], help="Prefix lengths to calibrate linear KV model")
    parser.add_argument("--out", type=str, default="count_latency_results.json", help="Output JSON path (relative to this directory by default)")
    parser.add_argument("--skip-large-direct", action="store_true", help="Estimate KV time for very large n instead of measuring directly")
    parser.add_argument("--measure_threshold_tokens", type=int, default=150_000, help="If n > threshold and skip-large-direct, use estimation for KV")
    args = parser.parse_args()

    torch, _F = try_import_torch()
    base_dir = Path(__file__).resolve().parent

    device = pick_device(args.device)
    torch_dtype = map_dtype(args.dtype, device)

    default_hs, default_layers, default_heads = load_default_dims_from_coeffs(base_dir)
    hidden_size = int(args.hidden_size or default_hs)
    num_layers = int(args.num_layers or default_layers)
    num_attention_heads = int(args.num_attention_heads or default_heads)

    report: Dict = {
        "meta": {
            "device": device,
            "dtype": str(torch_dtype),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "cwd": os.getcwd(),
        },
        "prefix_lengths": list(args.prefix_lengths),
        "trials": int(args.trials),
        "warmup": int(args.warmup),
        "calibration_lengths": list(args.calib_lengths),
        "with_kv_cache": {
            "method": "measured_or_estimated",
            "per_prefix": {},
        },
        "without_kv_cache": {
            "method": "estimated_from_linear_fit",
            "fit_per_trial": [],
            "per_prefix": {},
        },
    }

    # Prepare containers for results
    with_kv_times_per_prefix: Dict[int, List[float]] = {n: [] for n in args.prefix_lengths}
    no_kv_times_per_prefix: Dict[int, List[float]] = {n: [] for n in args.prefix_lengths}

    # Per-trial: calibrate linear model on small lengths, then measure/estimate per prefix
    for trial_idx in range(args.trials):
        # Calibrate linear fit for per-layer KV time
        calib_xs: List[float] = []
        calib_ys_ms: List[float] = []
        for n in args.calib_lengths:
            per_layer_times_ms = measure_attention_per_layer_ms(
                n_tokens=int(n),
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                device=device,
                torch_dtype=torch_dtype,
                warmup=args.warmup if trial_idx == 0 else 0,
                iters=1,
            )
            # Average over iters=1 is itself
            calib_xs.append(float(n))
            calib_ys_ms.append(float(per_layer_times_ms[0]))

        a_ms_per_token, b_ms = fit_linear(calib_xs, calib_ys_ms)
        report["without_kv_cache"]["fit_per_trial"].append({
            "a_per_token_ms": a_ms_per_token,
            "b_ms": b_ms,
            "calibration_points": [{"n": int(x), "per_layer_ms": y} for x, y in zip(calib_xs, calib_ys_ms)],
        })

        # For each requested prefix length: measure KV (or estimate if requested), estimate no-KV
        for n in args.prefix_lengths:
            measure_direct = not args.skip_large_direct or (int(n) <= int(args.measure_threshold_tokens))

            if measure_direct:
                # Measure KV per-layer, average across iters, scale by num_layers
                kv_per_layer_times_ms = measure_attention_per_layer_ms(
                    n_tokens=int(n),
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    device=device,
                    torch_dtype=torch_dtype,
                    warmup=args.warmup if trial_idx == 0 else 0,
                    iters=1,
                )
                kv_time_ms = float(kv_per_layer_times_ms[0]) * float(num_layers)
            else:
                # Estimate KV via linear fit
                kv_per_layer_ms_est = a_ms_per_token * float(n) + b_ms
                kv_time_ms = kv_per_layer_ms_est * float(num_layers)

            with_kv_times_per_prefix[int(n)].append(kv_time_ms)

            # Estimate no-KV via summed linear model
            no_kv_per_layer_ms_est = estimate_no_kv_time_ms_from_fit(int(n), a_ms_per_token, b_ms)
            no_kv_time_ms = no_kv_per_layer_ms_est * float(num_layers)
            no_kv_times_per_prefix[int(n)].append(no_kv_time_ms)

    # Aggregate statistics
    for n, arr in with_kv_times_per_prefix.items():
        report["with_kv_cache"]["per_prefix"][str(n)] = {
            "times_ms": arr,
            "mean_ms": statistics.fmean(arr) if arr else 0.0,
            "stdev_ms": statistics.pstdev(arr) if len(arr) > 1 else 0.0,
        }
    for n, arr in no_kv_times_per_prefix.items():
        report["without_kv_cache"]["per_prefix"][str(n)] = {
            "times_ms": arr,
            "mean_ms": statistics.fmean(arr) if arr else 0.0,
            "stdev_ms": statistics.pstdev(arr) if len(arr) > 1 else 0.0,
        }

    # Resolve output path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved latency results to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


