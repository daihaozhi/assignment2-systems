import torch
import torch.distributed as dist
from torch import nn

class DDP:
    def __init__(self, module: nn.Module):
        self.module = module
        self._handles = []  # 存储异步通信句柄
        self._broadcast_parameters()
        self._setup_gradient_hooks()

    def _broadcast_parameters(self):
        for param in self.module.parameters():
            if param.requires_grad:
                if dist.get_rank() == 0:
                    dist.broadcast(param.data, src=0)
                else:
                    dist.broadcast(param.data, src=0)

    def _setup_gradient_hooks(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, grad):
        handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
        self._handles.append(handle)

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self._handles:
            handle.wait()
        self._handles.clear()