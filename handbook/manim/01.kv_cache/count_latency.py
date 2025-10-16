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
from transformers import AutoConfig, AutoModelForCausalLM, StaticCache
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def kv_cache_bytes_per_token_from_cfg(cfg: Dict, bytes_per_elem: int) -> int:
    """Approximate KV cache bytes per token using config fields.

    Uses: 2 * num_key_value_heads * head_dim * num_layers * bytes_per_element
    Assumes float16/bfloat16 => 2 bytes per element; float32 => 4.
    """
    hidden_size = int(cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model") or 0)
    num_layers = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0)
    num_heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or 0)
    num_kv_heads = int(cfg.get("num_key_value_heads") or cfg.get("n_head_kv") or cfg.get("n_key_value_heads") or num_heads or 0)
    head_dim = hidden_size // max(1, num_heads)
    return int(2 * num_kv_heads * head_dim * num_layers * bytes_per_elem)


def measure_attention_per_layer_ms(
    *,
    model: Any,
    n_tokens: int,
    device: str,
    torch_dtype,
    warmup: int,
    iters: int,
    no_kv_cache=False,
) -> List[float]:
    """Measure per-layer time (ms) to generate one token using a HF model with KV cache.

    1) Prefill on a dummy prefix of length n with use_cache=True
    2) Time a single next-token step with past_key_values
    Returns per-layer times: total_time_per_step / num_layers.
    """
    model.eval()
    token_id = model.config.bos_token_id
    if token_id is None:
        token_id = model.config.eos_token_id if getattr(model.config, "eos_token_id", None) is not None else 1
    token_id = int(token_id)

    input_ids = torch.full((1, int(n_tokens)), int(token_id), device=device, dtype=torch.long)
    attn_mask = torch.ones((1, int(n_tokens)), device=device, dtype=torch.long)

    with torch.no_grad():
        synchronize(device)

        use_cache = True
        if no_kv_cache:
            use_cache = False

        past = None
        next_ids = input_ids
        next_mask = attn_mask
        if not no_kv_cache:
            print("Creating StaticCache, prefill", n_tokens)
            past = StaticCache(config=model.config, max_cache_len=n_tokens + warmup + iters + 2)
            outputs = model(input_ids=next_ids, attention_mask=next_mask, use_cache=use_cache, past_key_values=past, logits_to_keep=1)
            past = outputs.past_key_values

            next_ids = torch.full((1, 1), int(token_id), device=device, dtype=torch.long)
            next_mask = torch.ones((1, int(n_tokens) + 1), device=device, dtype=torch.long)

            del outputs
            torch.cuda.empty_cache()

        print("Warmup", n_tokens)
        for _ in range(max(0, warmup)):
            outputs = model(input_ids=next_ids, attention_mask=next_mask, use_cache=use_cache, past_key_values=past, logits_to_keep=1)
            del outputs
            # torch.cuda.empty_cache()

        print("Measure", n_tokens, 'use_cache', use_cache, 'past', past)
        times_ms: List[float] = []
        for iter_i in range(iters + 1):
            synchronize(device)
            t0 = time.perf_counter()
            outputs = model(input_ids=next_ids, attention_mask=next_mask, use_cache=use_cache, past_key_values=past, logits_to_keep=1)
            del outputs
            synchronize(device)
            t1 = time.perf_counter()
            if iter_i == 0:
                continue
            times_ms.append((t1 - t0) * 1e3)

    num_layers = int(getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "n_layer", 1))
    if num_layers <= 0:
        num_layers = 1
    return [t / float(num_layers) for t in times_ms]


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="Measure per-token generation latency with/without KV-Cache.")
    parser.add_argument("--model_name", type=str, required=True, help="HF model id (e.g., meta-llama/Llama-3.1-8B)")
    parser.add_argument("--device", type=str, default="", help="cuda|cpu (auto if empty)")
    parser.add_argument("--dtype", type=str, default="float16", help="float16|bfloat16|float32")
    parser.add_argument("--prefix-lengths", type=int, nargs="*", default=[10_000, 20_000 ], help="Prefix lengths to test")
    # parser.add_argument("--prefix-lengths", type=int, nargs="*", default=[25_000, 50_000, 75_000, 100_000 ], help="Prefix lengths to test")
    parser.add_argument("--trials", type=int, default=2, help="Trials per measurement")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs before timing (per measurement)")
    parser.add_argument("--out", type=str, default="count_latency_results.json", help="Output JSON path (relative to this directory by default)")
    parser.add_argument("--plot", action="store_true", help="Also save a plot comparing KV vs No-KV latencies")
    parser.add_argument("--plot-out", type=str, default="count_latency_plot.png", help="Where to save the plot image")
    parser.add_argument("--plot-kv-size-out", type=str, default="kv_cache_size_plot.png", help="Where to save KV size vs length plot")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    device = pick_device(args.device)
    torch_dtype = map_dtype(args.dtype, device)

    # Load pretrained config and model
    cfg_obj = AutoConfig.from_pretrained(args.model_name)
    cfg = cfg_obj.to_dict()
    hidden_size, num_layers, num_attention_heads = extract_model_dims(cfg)
    # Infer bytes-per-element from dtype
    if torch_dtype in (torch.float16, torch.bfloat16):
        kv_bytes_per_elem = 2
    elif torch_dtype == torch.float32:
        kv_bytes_per_elem = 4
    else:
        kv_bytes_per_elem = 2
    per_token_kv_bytes = kv_cache_bytes_per_token_from_cfg(cfg, kv_bytes_per_elem)
    model: Any = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch_dtype, attn_implementation="sdpa")  # type: ignore[call-arg,]
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
        "with_kv_cache": {
            "per_prefix": {},
        },
        "without_kv_cache": {
            "fit_per_trial": [],
            "per_prefix": {},
        },
        "kv_cache_size": {
            "bytes_per_token": per_token_kv_bytes,
            "per_prefix_bytes": {},
        },
    }

    # Prepare containers for results
    with_kv_times_per_prefix: Dict[int, List[float]] = {n: [] for n in args.prefix_lengths}
    no_kv_times_per_prefix: Dict[int, List[float]] = {n: [] for n in args.prefix_lengths}

    # Per-trial: calibrate linear model on small lengths, then measure/estimate per prefix
    for n in tqdm(sorted(args.prefix_lengths, reverse=True), desc="Prefix lengths"):
            kv_per_layer_times_ms = measure_attention_per_layer_ms(
                model=model,
                n_tokens=int(n),
                device=device,
                torch_dtype=torch_dtype,
                warmup=args.trials,
                iters=20,
            )
            kv_time_ms = float(kv_per_layer_times_ms[0]) * float(num_layers)

            with_kv_times_per_prefix[int(n)].append(kv_time_ms)
            report["kv_cache_size"]["per_prefix_bytes"][str(int(n))] = int(per_token_kv_bytes * int(n))

            if device == "cuda":
                torch.cuda.empty_cache()

            no_kv_per_layer_times_ms = measure_attention_per_layer_ms(
                model=model,
                n_tokens=int(n),
                device=device,
                torch_dtype=torch_dtype,
                warmup=args.warmup,
                iters=args.trials,
                no_kv_cache=True,
            )
            no_kv_per_layer_times_ms = float(kv_per_layer_times_ms[0]) * float(num_layers)
            no_kv_times_per_prefix[int(n)].append(no_kv_per_layer_times_ms)

            if device == "cuda":
                torch.cuda.empty_cache()


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

    # Optional plot
    if args.plot:
        # Prepare data
        xs = sorted(int(n) for n in report["with_kv_cache"]["per_prefix"].keys())
        kv_means = [report["with_kv_cache"]["per_prefix"][str(n)]["mean_ms"] for n in xs]
        kv_stdevs = [report["with_kv_cache"]["per_prefix"][str(n)]["stdev_ms"] for n in xs]
        no_kv_means = [report["without_kv_cache"]["per_prefix"][str(n)]["mean_ms"] for n in xs]
        no_kv_stdevs = [report["without_kv_cache"]["per_prefix"][str(n)]["stdev_ms"] for n in xs]

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        ax.errorbar(xs, kv_means, yerr=kv_stdevs, fmt="-o", capsize=3, label="With KV-Cache")
        ax.errorbar(xs, no_kv_means, yerr=no_kv_stdevs, fmt="-o", capsize=3, label="Without KV-Cache")
        ax.set_xlabel("Prefix length (tokens)")
        ax.set_ylabel("Time per generated token (ms)")
        ax.set_title(f"Latency vs Prefix length — {args.model_name}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plot_path = Path(args.plot_out)
        if not plot_path.is_absolute():
            plot_path = base_dir / plot_path
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved plot to: {plot_path}")

        # KV cache size plot
        xs_kv = sorted(int(n) for n in report["kv_cache_size"]["per_prefix_bytes"].keys())
        ys_kv_bytes = [int(report["kv_cache_size"]["per_prefix_bytes"][str(n)]) for n in xs_kv]
        ys_kv_gb = [y / 1e9 for y in ys_kv_bytes]

        fig2, ax2 = plt.subplots(figsize=(8, 4.5), dpi=150)
        ax2.plot(xs_kv, ys_kv_gb, "-o", label="KV-Cache size")
        ax2.set_xlabel("Sequence length (tokens)")
        ax2.set_ylabel("KV-Cache size (GB)")
        ax2.set_title(f"KV-Cache size vs sequence length — {args.model_name}")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plot_kv_path = Path(args.plot_kv_size_out)
        if not plot_kv_path.is_absolute():
            plot_kv_path = base_dir / plot_kv_path
        fig2.tight_layout()
        fig2.savefig(plot_kv_path)
        plt.close(fig2)
        print(f"Saved KV size plot to: {plot_kv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


