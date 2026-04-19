import torch
import torch.distributed as dist
from torch import nn

class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._handles = []  # 存储异步通信句柄
        self._broadcast_parameters()
        self._setup_gradient_hooks()

    def _broadcast_parameters(self):
        for name, param in self.module.named_parameters():
            # We must broadcast all parameters (and buffers usually) to ensure exact equivalence.
            # But the test checks `no_grad_fixed_param` as well, so we should broadcast everything.
            dist.broadcast(param.data, src=0)

    def _setup_gradient_hooks(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, param):
        # The hook receives the parameter tensor. Its .grad field is fully populated.
        param.grad.data /= dist.get_world_size()
        handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
        self._handles.append(handle)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self._handles:
            handle.wait()
        self._handles.clear()

    
class DDP_BUCKETED(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self._broadcast_parameters()
        self._build_buckets()
        self._setup_gradient_hooks()

    def _broadcast_parameters(self):
        for name, param in self.module.named_parameters():
            dist.broadcast(param.data, src=0)

    def _build_buckets(self):
        self.buckets = []
        current_bucket = []
        current_bucket_size = 0
        
        # 1. 按照模型参数的逆序排列（为了匹配反向传播生成梯度的顺序）
        params = [p for p in self.module.parameters() if p.requires_grad]
        params = list(reversed(params))
        
        limit_bytes = self.bucket_size_mb * 1024 * 1024 if self.bucket_size_mb is not None else float('inf')
        
        for p in params:
            p_size = p.numel() * p.element_size()
            # 如果加上当前参数超出了 bucket 的大小限制，并且当前 bucket 不为空，就新建一个 bucket
            if current_bucket_size + p_size > limit_bytes and len(current_bucket) > 0:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_bucket_size = 0
            
            current_bucket.append(p)
            current_bucket_size += p_size
            
        if len(current_bucket) > 0:
            self.buckets.append(current_bucket)
            
        # 2. 为每个 Bucket 预分配连续显存池，并记录参数在这个池子里的切片偏移
        self.bucket_states = []
        self.param_to_bucket = {}
        for i, b_params in enumerate(self.buckets):
            total_elements = sum(p.numel() for p in b_params)
            dtype = b_params[0].dtype
            device = b_params[0].device
            buffer = torch.zeros(total_elements, dtype=dtype, device=device)
            
            offset = 0
            param_slices = {}
            for p in b_params:
                numel = p.numel()
                param_slices[p] = slice(offset, offset + numel)
                self.param_to_bucket[p] = i
                offset += numel
                
            self.bucket_states.append({
                'buffer': buffer,
                'param_slices': param_slices,
                'ready_count': 0,
                'total_params': len(b_params),
                'handle': None
            })

    def _setup_gradient_hooks(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, param):
        bucket_idx = self.param_to_bucket[param]
        state = self.bucket_states[bucket_idx]
        
        # 1. 将刚算好的参数梯度复制到预分配的连续 bucket 内存对应的切片中
        slc = state['param_slices'][param]
        state['buffer'][slc].copy_(param.grad.data.view(-1))
        
        state['ready_count'] += 1
        
        # 2. 当这个 bucket 里面所有的参数都算完梯度了，立即触发异步 All-Reduce
        if state['ready_count'] == state['total_params']:
            state['buffer'].div_(dist.get_world_size())
            state['handle'] = dist.all_reduce(state['buffer'], op=dist.ReduceOp.SUM, async_op=True)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        # 反向传播结束后调用，阻塞等待所有通信句柄完成，并把同步后的梯度写回各层
        for state in self.bucket_states:
            if state['handle'] is not None:
                state['handle'].wait()
                state['handle'] = None
            
            # 将归约后的连续梯度显存切分并覆盖回原始的 param.grad.data
            for p, slc in state['param_slices'].items():
                p.grad.data.copy_(state['buffer'][slc].view_as(p.grad.data))

    def reset_buckets(self):
        # 每一个新的 iteration (batch) 开始前调用，清空计数器和句柄
        for state in self.bucket_states:
            state['ready_count'] = 0
            state['handle'] = None