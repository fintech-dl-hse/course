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
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def pick_device(user_device: str) -> str:
    if user_device:
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def map_dtype(dtype_str: str, device: str):
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
    if device == "cuda":
        torch.cuda.synchronize()


def extract_model_dims(cfg: Dict) -> Tuple[int, int, int]:
    """Extract (hidden_size, num_layers, num_attention_heads) from HF config dict."""
    hidden_size = cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model")
    num_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    num_heads = cfg.get("num_attention_heads") or cfg.get("n_head")
    if hidden_size is None or num_layers is None or num_heads is None:
        raise ValueError("Could not extract model dimensions from config; please try a different model.")
    return int(hidden_size), int(num_layers), int(num_heads)


def measure_attention_per_layer_ms(
    *,
    model: Any,
    n_tokens: int,
    device: str,
    torch_dtype,
    warmup: int,
    iters: int,
) -> List[float]:
    """Measure per-layer time (ms) to generate one token using a HF model with KV cache.

    1) Prefill on a dummy prefix of length n with use_cache=True
    2) Time a single next-token step with past_key_values
    Returns per-layer times: total_time_per_step / num_layers.
    """
    model.eval()
    token_id = model.config.bos_token_id
    if token_id is None:
        token_id = model.config.eos_token_id if model.config.eos_token_id is not None else 1

    input_ids = torch.full((1, int(n_tokens)), int(token_id), device=device, dtype=torch.long)
    attn_mask = torch.ones((1, int(n_tokens)), device=device, dtype=torch.long)

    with torch.inference_mode():
        synchronize(device)
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
        past = getattr(outputs, "past_key_values", None)

        for _ in range(max(0, warmup)):
            next_ids = torch.full((1, 1), int(token_id), device=device, dtype=torch.long)
            next_mask = torch.ones((1, int(n_tokens) + 1), device=device, dtype=torch.long)
            _ = model(input_ids=next_ids, attention_mask=next_mask, use_cache=True, past_key_values=past)

        times_ms: List[float] = []
        for _ in range(iters):
            next_ids = torch.full((1, 1), int(token_id), device=device, dtype=torch.long)
            next_mask = torch.ones((1, int(n_tokens) + 1), device=device, dtype=torch.long)
            synchronize(device)
            t0 = time.perf_counter()
            _ = model(input_ids=next_ids, attention_mask=next_mask, use_cache=True, past_key_values=past)
            synchronize(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1e3)

    num_layers = int(getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "n_layer", 1))
    if num_layers <= 0:
        num_layers = 1
    return [t / float(num_layers) for t in times_ms]


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
    parser.add_argument("--model_name", type=str, required=True, help="HF model id (e.g., meta-llama/Llama-3.1-8B)")
    parser.add_argument("--device", type=str, default="", help="cuda|cpu (auto if empty)")
    parser.add_argument("--dtype", type=str, default="float16", help="float16|bfloat16|float32")
    parser.add_argument("--prefix-lengths", type=int, nargs="*", default=[10_000, 100_000, 250_000], help="Prefix lengths to test")
    parser.add_argument("--trials", type=int, default=10, help="Trials per measurement")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs before timing (per measurement)")
    parser.add_argument("--calib-lengths", type=int, nargs="*", default=[4096, 8192, 16384], help="Prefix lengths to calibrate linear KV model")
    parser.add_argument("--out", type=str, default="count_latency_results.json", help="Output JSON path (relative to this directory by default)")
    parser.add_argument("--skip-large-direct", action="store_true", help="Estimate KV time for very large n instead of measuring directly")
    parser.add_argument("--measure_threshold_tokens", type=int, default=150_000, help="If n > threshold and skip-large-direct, use estimation for KV")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    device = pick_device(args.device)
    torch_dtype = map_dtype(args.dtype, device)

    # Load pretrained config and model
    cfg = AutoConfig.from_pretrained(args.model_name)
    hidden_size, num_layers, num_attention_heads = extract_model_dims(cfg.to_dict())
    model: Any = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch_dtype)  # type: ignore[call-arg]
    model.to(device)

    report: Dict = {
        "meta": {
            "device": device,
            "dtype": str(torch_dtype),
            "model_name": args.model_name,
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
                model=model,
                n_tokens=int(n),
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
                # Measure KV per-layer via real model, then scale by num_layers
                kv_per_layer_times_ms = measure_attention_per_layer_ms(
                    model=model,
                    n_tokens=int(n),
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


