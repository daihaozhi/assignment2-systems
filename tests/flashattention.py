import torch


class flashattentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v,is_causal=False):
        Nq=q.shape[0]
        Nk=k.shape[0]
        d=q.shape[-1]
        Bq=16
        Bk=16
        Tq=(Nq+Bq-1)//Bq
        Tk=(Nk+Bk-1)//Bk

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("Backward pass not implemented")