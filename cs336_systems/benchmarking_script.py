from __future__ import annotations

import argparse
import timeit

import torch

from cs336_basics.model import BasicsTransformerLM


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cs336_basics model forward or forward+backward pass."
    )

    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)

    parser.add_argument("--warmup-steps", type=int, default=10, help="w")
    parser.add_argument("--measure-steps", type=int, default=50, help="n")
    parser.add_argument(
        "--mode",
        choices=("forward", "forward_backward"),
        default="forward",
    )

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    if args.sequence_length > args.context_length:
        raise ValueError("sequence_length must be <= context_length.")

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device=device, dtype=dtype)

    model.train(args.mode == "forward_backward")

    token_ids = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.sequence_length),
        device=device,
        dtype=torch.long,
    )

    def forward_step() -> None:
        with torch.no_grad():
            _ = model(token_ids)
        _sync_if_cuda(device)

    def forward_backward_step() -> None:
        model.zero_grad(set_to_none=True)
        logits = model(token_ids)
        loss = logits.mean()
        loss.backward()
        _sync_if_cuda(device)

    step_fn = forward_step if args.mode == "forward" else forward_backward_step

    for _ in range(args.warmup_steps):
        step_fn()

    _sync_if_cuda(device)
    total_seconds = timeit.timeit(step_fn, number=args.measure_steps)
    _sync_if_cuda(device)

    ms_per_step = (total_seconds / args.measure_steps) * 1000.0

    print("Benchmark Results")
    print(f"mode: {args.mode}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"warmup_steps (w): {args.warmup_steps}")
    print(f"measure_steps (n): {args.measure_steps}")
    print(f"total_time_seconds: {total_seconds:.6f}")
    print(f"time_per_step_ms: {ms_per_step:.3f}")


if __name__ == "__main__":
    main()
