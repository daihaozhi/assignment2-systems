import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run_benchmark(rank, world_size, backend, sizes_mb):
    # Setup distributed environment
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['GLOO_SOCKET_IFNAME'] = '127.0.0.1'
    
    try:
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    except Exception as e:
        if rank == 0:
            print(f"Failed to initialize process group with backend '{backend}': {e}")
        return

    # Determine device
    if backend == "nccl":
        # Map to available GPUs. If only 1 GPU, multiple ranks might map to cuda:0
        device_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    warmup_iters = 5
    measure_iters = 10

    results = {}

    for size_mb in sizes_mb:
        # 1 float32 = 4 bytes
        num_elements = (size_mb * 1024 * 1024) // 4
        tensor = torch.randn(num_elements, dtype=torch.float32, device=device)

        # Warmup
        for _ in range(warmup_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dist.barrier()

        # Measurement
        start_time = time.perf_counter()
        for _ in range(measure_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / measure_iters) * 1000.0
        results[size_mb] = avg_time_ms

    if rank == 0:
        print(f"\n--- Backend: {backend.upper()} | Device: {device.type.upper()} | Processes: {world_size} ---")
        for size_mb, avg_time in results.items():
            print(f"Size: {size_mb:4d} MB | Avg Time: {avg_time:8.2f} ms")

    dist.destroy_process_group()

def main():
    # all-reduce data sizes as required: 1MB, 10MB, 100MB, 1GB (1024MB)
    sizes_mb = [1, 10, 100, 1024]
    
    # As requested: maximum 2 processes to save GPU resources
    world_size = 2  

    print(f"Starting benchmarks with {world_size} processes...")

    # 1. Gloo + CPU
    print(f"\n[1/2] Running Gloo + CPU benchmark...")
    mp.spawn(run_benchmark,
             args=(world_size, "gloo", sizes_mb),
             nprocs=world_size,
             join=True)

    # 2. NCCL + GPU
    if torch.cuda.is_available() and dist.is_nccl_available():
        print(f"\n[2/2] Running NCCL + GPU benchmark...")
        mp.spawn(run_benchmark,
                 args=(world_size, "nccl", sizes_mb),
                 nprocs=world_size,
                 join=True)
    else:
        print("\n[2/2] Skipping NCCL + GPU benchmark because CUDA or NCCL is not available on this system.")
        print("      (Note: PyTorch on Windows natively does not support NCCL. Run this on Linux/WSL for GPU results)")

if __name__ == "__main__":
    main()
