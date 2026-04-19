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
        for param in self.module.parameters():
            if param.requires_grad:
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