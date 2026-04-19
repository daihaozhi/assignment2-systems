import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

# ==========================================
# 1. 定义一个中等规模的模型
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=8192, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. 单卡基准训练函数
# ==========================================
def train_single_gpu(d_size, global_batch_size, epochs, input_dim=4096):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[Single Device] Starting training on {device}...")
    
    # 初始化模型、优化器和损失函数
    model = SimpleMLP(input_dim=input_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 随机生成总规模为 d 的数据
    inputs = torch.randn(d_size, input_dim, device=device)
    targets = torch.randint(0, 10, (d_size,), device=device)

    # 预热一次 (排除初次初始化 CUDA 上下文的开销)
    optimizer.zero_grad()
    loss = criterion(model(inputs[:global_batch_size]), targets[:global_batch_size])
    loss.backward()
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()
    
    # 模拟训练
    for epoch in range(epochs):
        for i in range(0, d_size, global_batch_size):
            x = inputs[i:i+global_batch_size]
            y = targets[i:i+global_batch_size]
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    print(f"[Single Device] Completed! Time taken: {total_time:.4f} seconds")
    return total_time

# ==========================================
# 3. 分布式手工 DDP 训练函数
# ==========================================
def train_distributed(rank, world_size, d_size, global_batch_size, epochs, input_dim=4096):
    # 配置分布式环境变量
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12356'
    os.environ['GLOO_SOCKET_IFNAME'] = '127.0.0.1'
    
    backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # 设置当前进程对应的设备
    if torch.cuda.is_available():
        device_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # 初始化模型
    model = SimpleMLP(input_dim=input_dim).to(device)
    
    # 【关键】把 Rank 0 的初始参数广播给所有进程，确保初始化完全一致
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 每个进程只处理 d/n 规模的数据
    local_d_size = d_size // world_size
    # 局部批次大小 (维持相同的更新步数)
    local_batch_size = global_batch_size // world_size

    # 随机生成每个进程私有的局部数据
    inputs = torch.randn(local_d_size, input_dim, device=device)
    targets = torch.randint(0, 10, (local_d_size,), device=device)

    # 预热一次
    optimizer.zero_grad()
    loss = criterion(model(inputs[:local_batch_size]), targets[:local_batch_size])
    loss.backward()
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dist.barrier()

    if rank == 0:
        print(f"\n[Distributed] Starting training on {world_size} processes using {backend}...")
    
    start_time = time.perf_counter()
    
    total_compute_time = 0.0
    total_comm_time = 0.0

    # 模拟训练
    for epoch in range(epochs):
        for i in range(0, local_d_size, local_batch_size):
            x = inputs[i:i+local_batch_size]
            y = targets[i:i+local_batch_size]
            
            # --- 记录计算开始时间 ---
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # 反向传播计算局部梯度
            loss.backward()
            
            # --- 记录计算结束 / 通信开始时间 ---
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            total_compute_time += (t1 - t0)
            
            # 【核心手工 DDP 逻辑】：同步所有进程的梯度，求平均
            for param in model.parameters():
                if param.grad is not None:
                    # 将所有卡上的 param.grad 相加
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    # 求平均
                    param.grad.data /= world_size

            # 参数更新
            optimizer.step()
            
            # --- 记录通信结束时间 ---
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t2 = time.perf_counter()
            total_comm_time += (t2 - t1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dist.barrier()
    end_time = time.perf_counter()

    if rank == 0:
        total_time = end_time - start_time
        print(f"[Distributed] Completed! Time taken: {total_time:.4f} seconds")
        print(f"  - Total Compute Time: {total_compute_time:.4f} s")
        print(f"  - Total Comm & Update Time: {total_comm_time:.4f} s")

    dist.destroy_process_group()

# ==========================================
# 4. 主函数入口
# ==========================================
def main():
    d_size = 200000            # 总数据规模增大一倍
    global_batch_size = 2000   # 批次大小增大
    epochs = 15                # 迭代轮数增加
    world_size = 2             # 进程数量 (2张显卡)

    # 1. 运行单卡训练并计时
    time_single = train_single_gpu(d_size, global_batch_size, epochs)

    # 2. 运行分布式手工 DDP 训练并计时
    mp.spawn(train_distributed,
             args=(world_size, d_size, global_batch_size, epochs),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
