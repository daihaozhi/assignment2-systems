from __future__ import annotations

import argparse
import contextlib
import timeit

import torch

from cs336_basics.model import BasicsTransformerLM


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cs336_basics model forward, forward+backward, or forward+backward+optimizer pass."
    )

    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)

    parser.add_argument("--warmup-steps", type=int, default=10, help="w")
    parser.add_argument("--measure-steps", type=int, default=50, help="n")
    parser.add_argument(
        "--mode",
        choices=("forward", "forward_backward", "forward_backward_optimizer"),
        default="forward",
    )
    parser.add_argument("--optimizer-lr", type=float, default=1e-3)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--use-bf16",
        action="store_true",
        help="Force bfloat16 precision for model weights and compute.",
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


@contextlib.contextmanager
def _nvtx_range(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    if args.sequence_length > args.context_length:
        raise ValueError("sequence_length must be <= context_length.")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.use_bf16 else _resolve_dtype(args.dtype)
    use_nvtx = device.type == "cuda"

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device=device, dtype=dtype)

    model.train(args.mode != "forward")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.optimizer_lr)

    token_ids = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.sequence_length),
        device=device,
        dtype=torch.long,
    )

    def forward_step() -> None:
        with torch.no_grad():
            with _nvtx_range("forward", use_nvtx):
                _ = model(token_ids)
        _sync_if_cuda(device)

    def forward_backward_step() -> None:
        with _nvtx_range("zero_grad", use_nvtx):
            model.zero_grad(set_to_none=True)
        with _nvtx_range("forward", use_nvtx):
            logits = model(token_ids)
        with _nvtx_range("loss", use_nvtx):
            loss = logits.mean()
        with _nvtx_range("backward", use_nvtx):
            loss.backward()
        _sync_if_cuda(device)

    def forward_backward_optimizer_step() -> None:
        with _nvtx_range("zero_grad", use_nvtx):
            optimizer.zero_grad(set_to_none=True)
        with _nvtx_range("forward", use_nvtx):
            logits = model(token_ids)
        with _nvtx_range("loss", use_nvtx):
            loss = logits.mean()
        with _nvtx_range("backward", use_nvtx):
            loss.backward()
        with _nvtx_range("optimizer_step", use_nvtx):
            optimizer.step()
        _sync_if_cuda(device)

    step_fn_by_mode = {
        "forward": forward_step,
        "forward_backward": forward_backward_step,
        "forward_backward_optimizer": forward_backward_optimizer_step,
    }
    step_fn = step_fn_by_mode[args.mode]

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
    if args.mode == "forward_backward_optimizer":
        print(f"optimizer_lr: {args.optimizer_lr}")
    print(f"warmup_steps (w): {args.warmup_steps}")
    print(f"measure_steps (n): {args.measure_steps}")
    print(f"total_time_seconds: {total_seconds:.6f}")
    print(f"time_per_step_ms: {ms_per_step:.3f}")


if __name__ == "__main__":
    main()
