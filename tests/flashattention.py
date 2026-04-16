import math
import torch


class flashattentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        if is_causal:
            raise NotImplementedError("Causal attention not implemented")
        Nq = Q.shape[0]
        Nk = K.shape[0]
        d = Q.shape[-1]
        Bq = 32
        Bk = 32
        Tq = (Nq + Bq - 1) // Bq
        Tk = (Nk + Bk - 1) // Bk
        
        O = torch.zeros_like(Q)
        L = torch.zeros(Nq, device=Q.device, dtype=Q.dtype)
        scale = 1.0 / math.sqrt(d)
        
        for i in range(Tq):
            Q_i = Q[i*Bq:(i+1)*Bq, :]
            q_len = Q_i.shape[0]
            O_i = Q.new_zeros((q_len, d))
            mi = Q.new_full((q_len,), float('-inf'))  # running max
            li = Q.new_zeros((q_len,))                # running sum(exp)
            
            for j in range(Tk):
                K_j = K[j*Bk:(j+1)*Bk, :]
                V_j = V[j*Bk:(j+1)*Bk, :]
                S = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale
                
                # Compute current block's max
                current_max = S.max(dim=1)[0]  # shape: (q_len,)
                
                if j == 0:
                    mj = current_max
                    P_tilde = torch.exp(S - mj.unsqueeze(1))
                    O_i = P_tilde @ V_j
                    li = P_tilde.sum(dim=1)
                    mi = mj
                else:
                    mj = torch.max(mi, current_max)  # new global max
                    # Key fix: use (mi - mj), not (mj - mi)!
                    exp_factor = torch.exp(mi - mj)  # shape: (q_len,)
                    P_tilde = torch.exp(S - mj.unsqueeze(1))
                    
                    # Update output with scaling
                    O_i = O_i * exp_factor.unsqueeze(-1) + P_tilde @ V_j
                    # Update normalization sum
                    li = li * exp_factor + P_tilde.sum(dim=1)
                    mi = mj  # update running max 
            # Final normalization
            O[i*Bq:(i+1)*Bq, :] = O_i / li.unsqueeze(-1)
            L[i*Bq:(i+1)*Bq] = mi + torch.log(li)  # logsumexp = m + log(l)
        return O,L

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("Backward pass not implemented")
