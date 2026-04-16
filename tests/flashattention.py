import math
import torch


class flashattentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        if is_causal:
            raise NotImplementedError("Causal attention not implemented")
        B = Q.shape[0]
        Nq = Q.shape[1]
        Nk = K.shape[1]
        d = Q.shape[-1]
        Bq = 32
        Bk = 32
        Tq = (Nq + Bq - 1) // Bq
        Tk = (Nk + Bk - 1) // Bk

        O = torch.zeros_like(Q)
        L = torch.zeros((B, Nq), device=Q.device, dtype=Q.dtype)
        scale = 1.0 / math.sqrt(d)

        for b in range(B):
            for i in range(Tq):
                q_start = i * Bq
                q_end = min((i + 1) * Bq, Nq)
                Q_i = Q[b, q_start:q_end, :]
                q_len = Q_i.shape[0]
                O_i = Q.new_zeros((q_len, d))
                mi = Q.new_full((q_len,), float('-inf'))  # running max
                li = Q.new_zeros((q_len,))                # running sum(exp)

                for j in range(Tk):
                    k_start = j * Bk
                    k_end = min((j + 1) * Bk, Nk)
                    K_j = K[b, k_start:k_end, :]
                    V_j = V[b, k_start:k_end, :]
                    S = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale

                    # Compute current block's max
                    current_max = S.max(dim=1).values  # shape: (q_len,)

                    if j == 0:
                        mj = current_max
                        P_tilde = torch.exp(S - mj.unsqueeze(1))
                        O_i = P_tilde @ V_j
                        li = P_tilde.sum(dim=1)
                        mi = mj
                    else:
                        mj = torch.max(mi, current_max)  # new global max
                        exp_factor = torch.exp(mi - mj)  # shape: (q_len,)
                        P_tilde = torch.exp(S - mj.unsqueeze(1))

                        # Update output with scaling
                        O_i = O_i * exp_factor.unsqueeze(-1) + P_tilde @ V_j
                        # Update normalization sum
                        li = li * exp_factor + P_tilde.sum(dim=1)
                        mi = mj  # update running max
                # Final normalization
                O[b, q_start:q_end, :] = O_i / li.unsqueeze(-1)
                L[b, q_start:q_end] = mi + torch.log(li)  # logsumexp = m + log(l)

        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, L)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, _L = ctx.saved_tensors
        q = Q.detach().requires_grad_(True)
        k = K.detach().requires_grad_(True)
        v = V.detach().requires_grad_(True)
        d = q.shape[-1]

        with torch.enable_grad():
            S = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
            if ctx.is_causal:
                q_idx = torch.arange(q.shape[-2], device=q.device)
                k_idx = torch.arange(k.shape[-2], device=k.device)
                mask = q_idx[:, None] >= k_idx[None, :]
                S = torch.where(mask[None, :, :], S, torch.full_like(S, -1e6))
            P = torch.softmax(S, dim=-1)
            O = torch.matmul(P, v)

        dQ, dK, dV = torch.autograd.grad(
            O,
            (q, k, v),
            grad_outputs=grad_out,
            retain_graph=False,
            allow_unused=False,
        )
        return dQ, dK, dV, None
