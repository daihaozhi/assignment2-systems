import math
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


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


def _check_triton_ready() -> None:
    if triton is None or tl is None:
        raise ImportError("Triton is not installed. Please `pip install triton` first.")


def _check_flashattention_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> None:
    if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
        raise ValueError("Triton FlashAttention requires CUDA tensors.")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Triton FlashAttention currently expects fp16, bf16, or fp32 tensors.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must have the same dtype.")
    if q.ndim not in (3, 4) or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError("Expected q, k, v shape: [B, N, D] or [B, H, N, D].")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("Batch/Head dimensions mismatch among q, k, v.")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("Head dim mismatch among q, k, v.")
    if not isinstance(is_causal, bool):
        raise ValueError("is_causal must be bool.")


if triton is not None:
    @triton.jit
    def _flashattn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr,L_ptr,
        stride_qb,stride_qq,stride_qd,
        stride_kb,stride_kk,stride_kd,
        stride_vb,stride_vk,stride_vd,
        stride_ob,stride_oq,stride_od,
        stride_lb,stride_lq,
        N_QUERIES,N_KEYS,
        scale,
        IS_CAUSAL: tl.constexpr,
        D:tl.constexpr,
        Q_TILE_SIZE:tl.constexpr,
        K_TILE_SIZE:tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        q_tile_idx = tl.program_id(0)
        batch_idx = tl.program_id(1)

        offs_m = q_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        offs_d = tl.arange(0, BLOCK_D)
        q_mask = (offs_m[:, None] < N_QUERIES) & (offs_d[None, :] < D)

        q_ptrs = (
            Q_ptr
            + batch_idx * stride_qb
            + offs_m[:, None] * stride_qq
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        acc = tl.zeros((Q_TILE_SIZE, BLOCK_D), dtype=tl.float32)

        for k_start in range(0, N_KEYS, K_TILE_SIZE):
            offs_n = k_start + tl.arange(0, K_TILE_SIZE)
            k_mask = (offs_n[:, None] < N_KEYS) & (offs_d[None, :] < D)
            v_mask = k_mask

            k_ptrs = (
                K_ptr
                + batch_idx * stride_kb
                + offs_n[:, None] * stride_kk
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                V_ptr
                + batch_idx * stride_vb
                + offs_n[:, None] * stride_vk
                + offs_d[None, :] * stride_vd
            )
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * scale
            qk = tl.where(offs_n[None, :] < N_KEYS, qk, -float("inf"))
            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_ij)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        o = acc / l_i[:, None]
        o_ptrs = (
            O_ptr
            + batch_idx * stride_ob
            + offs_m[:, None] * stride_oq
            + offs_d[None, :] * stride_od
        )
        o_mask = (offs_m[:, None] < N_QUERIES) & (offs_d[None, :] < D)
        tl.store(o_ptrs, o, mask=o_mask)

        l_ptrs = L_ptr + batch_idx * stride_lb + offs_m * stride_lq
        l_mask = offs_m < N_QUERIES
        tl.store(l_ptrs, m_i + tl.log(l_i), mask=l_mask)

    @triton.jit
    def _flashattn_bwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        do_ptr,
        dq_ptr,
        dk_ptr,
        dv_ptr,
        lse_ptr,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_on,
        stride_od,
        stride_dob,
        stride_doh,
        stride_don,
        stride_dod,
        stride_dqb,
        stride_dqh,
        stride_dqn,
        stride_dqd,
        stride_dkb,
        stride_dkh,
        stride_dkn,
        stride_dkd,
        stride_dvb,
        stride_dvh,
        stride_dvn,
        stride_dvd,
        stride_lb,
        stride_lh,
        stride_ln,
        B,
        H,
        NQ,
        NK,
        D,
        SCALE,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # TODO: implement Triton backward kernel.
        # Suggested strategy:
        # 1) recompute/stream needed logits statistics from Q/K/LSE
        # 2) accumulate dV, dK, dQ in block-wise loops
        # 3) write back to global memory
        return
else:
    def _flashattn_fwd_kernel(*_args, **_kwargs):
        raise ImportError("Triton is not installed.")

    def _flashattn_bwd_kernel(*_args, **_kwargs):
        raise ImportError("Triton is not installed.")


class FlashAttentionTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, sm_scale=None):
        _check_triton_ready()
        _check_flashattention_inputs(q, k, v, is_causal)

        if q.ndim == 4:
            b, h, nq, d = q.shape
            nk = k.shape[-2]
            q_in = q.reshape(b * h, nq, d)
            k_in = k.reshape(b * h, nk, d)
            v_in = v.reshape(b * h, nk, d)
            merged_batch = b * h
            restore_shape = (b, h, nq, d)
        else:
            merged_batch, nq, d = q.shape
            nk = k.shape[-2]
            q_in = q
            k_in = k
            v_in = v
            restore_shape = q.shape

        q_tile_size = 64
        k_tile_size = 64
        block_d = triton.next_power_of_2(d)
        scale = (1.0 / math.sqrt(d)) if sm_scale is None else float(sm_scale)

        o = torch.empty_like(q_in, device=q.device)
        l = torch.empty((merged_batch, nq), device=q.device, dtype=torch.float32)

        grid = (triton.cdiv(nq, q_tile_size), merged_batch)
        _flashattn_fwd_kernel[grid](
            q_in,
            k_in,
            v_in,
            o,
            l,
            q_in.stride(0),
            q_in.stride(1),
            q_in.stride(2),
            k_in.stride(0),
            k_in.stride(1),
            k_in.stride(2),
            v_in.stride(0),
            v_in.stride(1),
            v_in.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            l.stride(0),
            l.stride(1),
            nq,
            nk,
            scale,
            IS_CAUSAL=is_causal,
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            BLOCK_D=block_d,
        )

        if q.ndim == 4:
            l_to_save = l.reshape(b, h, nq).mean(dim=1)
        else:
            l_to_save = l

        ctx.save_for_backward(q, k, v, l_to_save)
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.original_ndim = q.ndim

        if q.ndim == 4:
            return o.reshape(restore_shape)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, _l = ctx.saved_tensors
        q_grad = q.detach().requires_grad_(True)
        k_grad = k.detach().requires_grad_(True)
        v_grad = v.detach().requires_grad_(True)

        with torch.enable_grad():
            scores = torch.matmul(q_grad, k_grad.transpose(-2, -1)) * ctx.scale
            if ctx.is_causal:
                q_idx = torch.arange(q.shape[-2], device=q.device)
                k_idx = torch.arange(k.shape[-2], device=k.device)
                causal_mask = q_idx[:, None] >= k_idx[None, :]
                view_shape = [1] * (scores.ndim - 2) + [q.shape[-2], k.shape[-2]]
                causal_mask = causal_mask.view(*view_shape)
                scores = torch.where(causal_mask, scores, torch.full_like(scores, -1e6))
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v_grad)

        dq, dk, dv = torch.autograd.grad(
            out,
            (q_grad, k_grad, v_grad),
            grad_outputs=do,
            retain_graph=False,
            allow_unused=False,
        )
        return dq, dk, dv, None, None


def flashattention_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False, sm_scale: float | None = None) -> torch.Tensor:
    """
    Triton FlashAttention wrapper.

    Expected shapes:
        q, k, v: [B, H, N, D]
    """
    return FlashAttentionTritonFunction.apply(q, k, v, is_causal, sm_scale)
