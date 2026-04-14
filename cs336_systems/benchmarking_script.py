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
        "--use-amp",
        action="store_true",
        help="Use mixed precision autocast (model params stay fp32 by default).",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="Autocast dtype when --use-amp is enabled.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--memory-snapshot-path",
        type=str,
        default=None,
        help="If set and running on CUDA, dump torch.cuda.memory snapshot pickle to this path.",
    )
    parser.add_argument(
        "--memory-history-max-entries",
        type=int,
        default=1000000,
        help="max_entries for torch.cuda.memory._record_memory_history().",
    )

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


def _autocast_or_nullcontext(
    use_amp: bool, device: torch.device, amp_dtype: torch.dtype
) -> contextlib.AbstractContextManager:
    if not use_amp:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


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
    if args.use_amp and device.type != "cuda":
        raise ValueError("--use-amp currently requires --device cuda.")

    amp_dtype = _resolve_dtype(args.amp_dtype)
    dtype = torch.float32 if args.use_amp else _resolve_dtype(args.dtype)
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
    use_grad_scaler = args.use_amp and amp_dtype == torch.float16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

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
                with _autocast_or_nullcontext(args.use_amp, device, amp_dtype):
                    _ = model(token_ids)
        _sync_if_cuda(device)

    def forward_backward_step() -> None:
        with _nvtx_range("zero_grad", use_nvtx):
            model.zero_grad(set_to_none=True)
        with _nvtx_range("forward", use_nvtx):
            with _autocast_or_nullcontext(args.use_amp, device, amp_dtype):
                logits = model(token_ids)
        with _nvtx_range("loss", use_nvtx):
            loss = logits.mean()
        with _nvtx_range("backward", use_nvtx):
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
        _sync_if_cuda(device)

    def forward_backward_optimizer_step() -> None:
        with _nvtx_range("zero_grad", use_nvtx):
            optimizer.zero_grad(set_to_none=True)
        with _nvtx_range("forward", use_nvtx):
            with _autocast_or_nullcontext(args.use_amp, device, amp_dtype):
                logits = model(token_ids)
        with _nvtx_range("loss", use_nvtx):
            loss = logits.mean()
        with _nvtx_range("backward", use_nvtx):
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
        with _nvtx_range("optimizer_step", use_nvtx):
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        _sync_if_cuda(device)

    step_fn_by_mode = {
        "forward": forward_step,
        "forward_backward": forward_backward_step,
        "forward_backward_optimizer": forward_backward_optimizer_step,
    }
    step_fn = step_fn_by_mode[args.mode]
    memory_history_supported = (
        hasattr(torch.cuda, "memory")
        and hasattr(torch.cuda.memory, "_record_memory_history")
        and hasattr(torch.cuda.memory, "_dump_snapshot")
    )
    enable_memory_snapshot = (
        device.type == "cuda" and args.memory_snapshot_path is not None and memory_history_supported
    )
    memory_snapshot_dumped = False

    for _ in range(args.warmup_steps):
        step_fn()

    _sync_if_cuda(device)
    if enable_memory_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=args.memory_history_max_entries)
    elif device.type == "cuda" and args.memory_snapshot_path is not None and not memory_history_supported:
        print("warning: torch.cuda.memory snapshot APIs are unavailable in this PyTorch build.")

    try:
        total_seconds = timeit.timeit(step_fn, number=args.measure_steps)
        _sync_if_cuda(device)
    finally:
        if enable_memory_snapshot:
            try:
                torch.cuda.memory._dump_snapshot(args.memory_snapshot_path)
                memory_snapshot_dumped = True
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)

    ms_per_step = (total_seconds / args.measure_steps) * 1000.0

    print("Benchmark Results")
    print(f"mode: {args.mode}")
    print(f"device: {device}")
    if args.use_amp:
        print(f"precision: amp ({args.amp_dtype})")
        print("model_param_dtype: torch.float32")
    else:
        print(f"precision: full ({dtype})")
    if args.mode == "forward_backward_optimizer":
        print(f"optimizer_lr: {args.optimizer_lr}")
    print(f"warmup_steps (w): {args.warmup_steps}")
    print(f"measure_steps (n): {args.measure_steps}")
    print(f"total_time_seconds: {total_seconds:.6f}")
    print(f"time_per_step_ms: {ms_per_step:.3f}")
    if memory_snapshot_dumped:
        print(f"memory_snapshot_path: {args.memory_snapshot_path}")


if __name__ == "__main__":
    main()
