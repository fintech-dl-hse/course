#!/usr/bin/env python3
import argparse
import json
import math
import sys
from typing import Any, Dict, Optional, Tuple


def try_import_transformers():
    try:
        import transformers  # type: ignore
        return transformers
    except Exception:
        return None


def get_config_from_model(model_name_or_path: str, trust_remote_code: bool = False) -> Dict[str, Any]:
    transformers = try_import_transformers()
    if transformers is None:
        raise RuntimeError("transformers is not installed. Install it or pass explicit --hidden-size/--num-layers/--num-attention-heads.")
    AutoConfig = transformers.AutoConfig
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    return cfg.to_dict()


def extract_model_dims(cfg: Dict[str, Any], overrides: Dict[str, Optional[int]]) -> Tuple[int, int, int, int]:
    hidden_size = overrides.get("hidden_size") or cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model")
    num_hidden_layers = overrides.get("num_hidden_layers") or cfg.get("num_hidden_layers") or cfg.get("n_layer")
    num_attention_heads = overrides.get("num_attention_heads") or cfg.get("num_attention_heads") or cfg.get("n_head")
    num_key_value_heads = (
        overrides.get("num_key_value_heads")
        or cfg.get("num_key_value_heads")
        or cfg.get("n_head_kv")
        or cfg.get("n_key_value_heads")
        or num_attention_heads
    )

    missing = []
    if hidden_size is None:
        missing.append("hidden_size")
    if num_hidden_layers is None:
        missing.append("num_hidden_layers")
    if num_attention_heads is None:
        missing.append("num_attention_heads")
    if num_key_value_heads is None:
        missing.append("num_key_value_heads")

    if missing:
        raise ValueError(f"Missing required dims: {', '.join(missing)}. Provide via CLI flags or use a supported model config.")

    # Type assertions post validation
    hidden_size = int(hidden_size)  # type: ignore[arg-type]
    num_hidden_layers = int(num_hidden_layers)  # type: ignore[arg-type]
    num_attention_heads = int(num_attention_heads)  # type: ignore[arg-type]
    num_key_value_heads = int(num_key_value_heads)  # type: ignore[arg-type]

    if hidden_size % max(1, num_attention_heads) != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads}) to compute head_dim."
        )

    return hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads


def bytes_per_element_for_dtype(dtype: str) -> int:
    dt = dtype.lower()
    if dt in ("float16", "fp16", "bfloat16", "bf16"):
        return 2
    if dt in ("float32", "fp32"):
        return 4
    if dt in ("float8", "fp8", "int8", "uint8"):
        return 1
    if dt in ("float64", "fp64", "double"):
        return 8
    raise ValueError(f"Unsupported dtype: {dtype}")


def compute_kv_cache_bytes_per_token(
    *,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    dtype: str,
) -> int:
    bytes_per_elem = bytes_per_element_for_dtype(dtype)
    head_dim = hidden_size // num_attention_heads
    # For each layer we store K and V for each KV head, each of size head_dim
    elements_per_token = 2 * num_key_value_heads * head_dim * num_hidden_layers
    return int(elements_per_token * bytes_per_elem)


def human_readable_bytes(n: int) -> str:
    if n == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    k = 1000.0
    i = int(math.floor(math.log(n, k)))
    i = max(0, min(i, len(units) - 1))
    return f"{n / (k ** i):.3f} {units[i]}"


def save_coefficients(
    *,
    out_path: str,
    a_bytes_per_token: int,
    b_bytes: int,
    meta: Dict[str, Any],
) -> None:
    payload = {
        "formula": "size_bytes = a * n_tokens + b",
        "a_bytes_per_token": a_bytes_per_token,
        "b_bytes": b_bytes,
        "meta": meta,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute KV-Cache size per token and extrapolate for token counts.")
    grp_model = p.add_mutually_exclusive_group(required=False)
    grp_model.add_argument("--model", type=str, help="HF model id or local path (e.g., meta-llama/Llama-3.1-8B).")
    grp_model.add_argument("--config-json", type=str, help="Path to a JSON file with model config fields.")

    p.add_argument("--hidden-size", type=int, default=None, help="Override hidden size (d_model).")
    p.add_argument("--num-layers", type=int, default=None, help="Override number of transformer layers.")
    p.add_argument("--num-attention-heads", type=int, default=None, help="Override number of attention heads.")
    p.add_argument("--num-kv-heads", type=int, default=None, help="Override number of key/value heads (defaults to attention heads).")

    p.add_argument("--dtype", type=str, default="float16", help="KV cache dtype: float16|bfloat16|float32|float8|int8.")
    p.add_argument("--trust-remote-code", action="store_true", help="Allow remote code when loading HF config.")
    p.add_argument("--out", type=str, default="kv_cache_coeffs.json", help="Where to write coefficients JSON.")

    p.add_argument("--counts", type=int, nargs="*", default=[10_000, 100_000, 1_000_000], help="Token counts to extrapolate.")
    return p.parse_args(argv)


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            return json.load(f)
    if args.model:
        return get_config_from_model(args.model, trust_remote_code=bool(args.trust_remote_code))
    return {}


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args)

    overrides = {
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_kv_heads,
    }

    hidden_size, num_layers, num_heads, num_kv_heads = extract_model_dims(cfg, overrides)
    per_token_bytes = compute_kv_cache_bytes_per_token(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        dtype=args.dtype,
    )

    # Linear model: size_bytes = a * n_tokens + b, here b=0 for pure KV cache
    a = per_token_bytes
    b = 0

    # Print summary
    model_label = args.model or (args.config_json or "<manual>")
    print("KV-Cache size per token")
    print(f"  model: {model_label}")
    print(f"  dtype: {args.dtype}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_attention_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {hidden_size // num_heads}")
    print(f"  bytes/token: {per_token_bytes} ({human_readable_bytes(per_token_bytes)})")

    print("\nExtrapolated sizes:")
    for n in args.counts:
        size = a * n + b
        print(f"  {n:>7,} tokens: {size} bytes ({human_readable_bytes(size)})")

    meta = {
        "model": model_label,
        "dtype": args.dtype,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": hidden_size // num_heads,
        "counts": args.counts,
    }
    save_coefficients(out_path=args.out, a_bytes_per_token=a, b_bytes=b, meta=meta)
    print(f"\nSaved coefficients to: {args.out}")
    return 0


if __name__ == "__main__":
    # sys.exit(main())

    main(["--model", "unsloth/Llama-3.2-3B", "--out", "handbook/manim/01.kv_cache/kv_cache_coeffs_llama3.2-3B.json"])
    main(["--model", "unsloth/Meta-Llama-3.1-8B", "--out", "handbook/manim/01.kv_cache/kv_cache_coeffs_llama3.1-7B.json"])

