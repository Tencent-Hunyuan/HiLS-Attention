"""
Fused Grouped Cross-Attention
===============
This is a Triton implementation of the Hierarchical Sparse Attention
Author: Xiang Hu
Extra Credits:
- Original flash attention2 paper (https://tridao.me/publications/flash2/flash2.pdf)
- OpenAI kernel team
"""

import pytest
import torch
import torch.nn.functional as F
import math
import triton
import triton.language as tl
from einops import rearrange, repeat


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

@triton.jit
def _has_nan(val):
    is_nan = (val != val)
    any_nan = tl.sum(is_nan)
    return any_nan > 0


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    qk_scale,  #
                    fp8_v: tl.constexpr):
    # -- compute qk ----
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (dim, N)
    qk = tl.dot(q, k)  # (group_size * M, N)
    m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
    qk = qk * qk_scale - m_ij[:, None]  # (group_size, M, N)
    p = tl.exp2(qk)
    l_ij = tl.sum(p, 1)
    # -- update m_i and l_i

    alpha = tl.exp2(-m_ij)  # (group_size * M)
    l_i = l_i * alpha + l_ij

    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    if fp8_v:
        p = p.to(tl.float8e5)
    else:
        p = p.to(v.type.element_ty)
    acc = tl.dot(p, v, acc)
    # update m_i and l_i
    m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, W, indices, sm_scale, sm_n, M, Out, # M (Batch, Q_CTX, neighbor_num)
              stride_qz, stride_ql, stride_qh, stride_qd,  #
              stride_kz, stride_kl, stride_kh, stride_kd,  #
              stride_vz, stride_vl, stride_vh, stride_vd,  #
              stride_oz, stride_ol, stride_oh, stride_od,  #
              stride_wz, stride_wq, stride_wh, stride_wk,  # (N, L // Q_X, h, K)
              stride_idxz, stride_idxq, stride_idxh, stride_idxk,  # (N, L // Q_X, h, K)
              stride_mz, stride_mk, stride_ml, stride_mh,  # (N, L // Q_X, H, K)
              L, L_kv, GROUP_NUM: tl.constexpr, NEIGHBOR_NUM: tl.constexpr, # Z: batch_size, H, head_num
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_KV: tl.constexpr,  # chunk size
              BLOCK_Q: tl.constexpr
              ):
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    # tl.static_assert(Q_CTX > BLOCK_M)
    start_m = tl.program_id(0)  # block_m id
    off_z = tl.program_id(1)
    off_h = tl.program_id(2)
    q_id = start_m * BLOCK_M // BLOCK_Q
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * GROUP_NUM * stride_qh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * GROUP_NUM * stride_oh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    w_offset = off_z.to(tl.int64) * stride_wz + off_h.to(tl.int64) * stride_wh + q_id.to(tl.int64) * stride_wq
    m_offset = off_z.to(tl.int64) * stride_mz + off_h.to(tl.int64) * GROUP_NUM * stride_mh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(L, GROUP_NUM, HEAD_DIM),
        strides=(stride_ql, stride_qh, stride_qd),
        offsets=(start_m * BLOCK_M, 0, 0),
        block_shape=(BLOCK_M, GROUP_NUM, HEAD_DIM),
        order=(2, 1, 0),
    )
    # q_block_offsets = (start_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_qm + tl.arange(0, HEAD_DIM)[None, :] * stride_qk
    # Q_block_ptr = Q + q_offset + q_block_offsets
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(L, GROUP_NUM, HEAD_DIM),
        strides=(stride_ol, stride_oh, stride_od),
        offsets=(start_m * BLOCK_M, 0, 0),
        block_shape=(BLOCK_M, GROUP_NUM, HEAD_DIM),
        order=(2, 1, 0),
    )

    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.442695040888963  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    q = q.reshape(GROUP_NUM * BLOCK_M, HEAD_DIM)

    acc = tl.zeros([GROUP_NUM * BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # outk_offsets = (start_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_okm + tl.arange(0, HEAD_DIM)[None, :] * stride_okk
    kv_idx_offset = off_z.to(tl.int64) * stride_idxz + q_id.to(tl.int64) * stride_idxq + off_h.to(tl.int64) * stride_idxh
    indices += kv_idx_offset
    V += v_offset
    K += k_offset
    M += m_offset
    W += w_offset
    for block_i in range(0, NEIGHBOR_NUM):
        kv_idx = tl.load(indices + block_i * stride_idxk).to(tl.int32)
        assert kv_idx == kv_idx, 'kv_idx is nan'

        V_block_ptr = tl.make_block_ptr(
            base=V,
            shape=(L_kv, HEAD_DIM),
            strides=(stride_vl, stride_vd),
            offsets=(kv_idx * BLOCK_KV, 0),
            block_shape=(BLOCK_KV, HEAD_DIM),
            order=v_order,
        )
        K_block_ptr = tl.make_block_ptr(
            base=K,
            shape=(HEAD_DIM, L_kv),
            strides=(stride_kd, stride_kl),
            offsets=(0, kv_idx * BLOCK_KV),
            block_shape=(HEAD_DIM, BLOCK_KV),
            order=(0, 1),
        )
        m_block_ptr = tl.make_block_ptr(
            base=M + block_i * stride_mk,
            shape=(L, GROUP_NUM),
            strides=(stride_ml, stride_mh),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, GROUP_NUM),
            order=(1, 0),
        )
        # v = tl.load(V_block_ptr, boundary_check=(0,))

        weight = tl.load(W + block_i * stride_wk)

        tmp = tl.zeros([GROUP_NUM * BLOCK_M, HEAD_DIM], dtype=tl.float32)
        # initialize pointer to m and l
        m_i = tl.zeros([GROUP_NUM * BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([GROUP_NUM * BLOCK_M], dtype=tl.float32) + sm_n


        tmp, l_i, m_i = _attn_fwd_inner(tmp, l_i, m_i, q,
                                        K_block_ptr, V_block_ptr,  #
                                        qk_scale,  #
                                        V.dtype.element_ty == tl.float8e5)
        # epilogue
        m_i += tl.log2(l_i)

        tmp = tmp / l_i[:, None]
        tl.store(m_block_ptr, m_i.reshape(BLOCK_M, GROUP_NUM), boundary_check=(0,))
        acc += weight * tmp


    acc = acc.reshape(BLOCK_M, GROUP_NUM, HEAD_DIM)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(1,))


@triton.jit
def kernel_attn_bwd_dq(Q, K, V, weights,  #
              DO, DQ, M, D,  # M: (N, L, H, K)
              indices, reg_lamda, reg_C,
              # shared by Q/K/V/DO.
              stride_qz, stride_ql, stride_qh, stride_qd,  # qz/qh: batch-level/head-level stride for query
              stride_kz, stride_kl, stride_kh, stride_kd,
              stride_vz, stride_vl, stride_vh, stride_vd,
              stride_wz, stride_wq, stride_wh, stride_wk,  # (N, q_blocks, h, kv_blocks)
              stride_doz, stride_dol, stride_doh, stride_dod,
              stride_dqz, stride_dql, stride_dqh, stride_dqd,
              stride_mz, stride_mk, stride_ml, stride_mh,
              stride_dz, stride_dk, stride_dl, stride_dh,
              stride_idxz, stride_idxq, stride_idxh, stride_idxk,
              L, L_kv, TOPK: tl.constexpr,  #
              GROUP_NUM: tl.constexpr,
              BLOCK_Q: tl.constexpr,  # block size for iterating O, Q
              BLOCK_KV: tl.constexpr,  # block size for KV,
              BLOCK_M: tl.constexpr,
              HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    m_id = tl.program_id(0)  # q block id
    b_id = tl.program_id(1)  # batch id
    h_id = tl.program_id(2)  # kv head_id
    q_id = m_id * BLOCK_M // BLOCK_Q

    q_offs = stride_qz * b_id.to(tl.int64) + stride_qh * h_id.to(tl.int64) * GROUP_NUM
    Q += q_offs

    do_offs = b_id.to(tl.int64) * stride_doz + h_id.to(tl.int64) * stride_doh * GROUP_NUM
    DO += do_offs

    dq_offs = b_id.to(tl.int64) * stride_dqz + h_id.to(tl.int64) * stride_dqh * GROUP_NUM
    DQ += dq_offs

    idx_offs = b_id.to(tl.int64) * stride_idxz + q_id.to(tl.int64) * stride_idxq + h_id.to(tl.int64) * stride_idxh
    indices += idx_offs

    M_offs = b_id.to(tl.int64) * stride_mz + h_id.to(tl.int64) * stride_mh * GROUP_NUM
    M += M_offs

    # indices: (N, q_blocks, K)
    # weight: (N, q_blocks, K)
    q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(L, GROUP_NUM, HEAD_DIM),
        strides=(stride_ql, stride_qh, stride_qd),
        offsets=(m_id * BLOCK_M, 0, 0),
        block_shape=(BLOCK_M, GROUP_NUM, HEAD_DIM),
        order=(2, 1, 0)
    )
    do_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(L, GROUP_NUM, HEAD_DIM),
        strides=(stride_dol, stride_doh, stride_dod),
        offsets=(m_id * BLOCK_M, 0, 0),
        block_shape=(BLOCK_M, GROUP_NUM, HEAD_DIM),
        order=(2, 1, 0)
    )

    dq_block_ptr = tl.make_block_ptr(
        base=DQ,
        shape=(L, GROUP_NUM, HEAD_DIM),
        strides=(stride_dql, stride_dqh, stride_dqd),
        offsets=(m_id * BLOCK_M, 0, 0),
        block_shape=(BLOCK_M, GROUP_NUM, HEAD_DIM),
        order=(2, 1, 0)
    )

    q = tl.load(q_block_ptr, boundary_check=(0, 1, 2), padding_option="zero")
    q = q.reshape(GROUP_NUM * BLOCK_M, HEAD_DIM)
    dq = tl.zeros([GROUP_NUM * BLOCK_M, HEAD_DIM], dtype=tl.float32)
    do = tl.load(do_block_ptr, boundary_check=(0, 1, 2), padding_option="zero")
    do = do.reshape(GROUP_NUM * BLOCK_M, HEAD_DIM)  # (G * block_m, dim)

    k_offs = b_id.to(tl.int64) * stride_kz + h_id.to(tl.int64) * stride_kh
    K += k_offs

    v_offs = b_id.to(tl.int64) * stride_vz + h_id.to(tl.int64) * stride_vh
    V += v_offs

    di_offs = b_id.to(tl.int64) * stride_dz + h_id.to(tl.int64) * stride_dh * GROUP_NUM
    # Di: (N, L, H, K)
    D += di_offs

    # weights: (N, q_blocks, h, K)
    weights_offs = b_id.to(tl.int64) * stride_wz + q_id.to(tl.int64) * stride_wq + h_id.to(tl.int64) * stride_wh
    weights += weights_offs

    for chunk_i in range(TOPK):
        kv_block_idx = tl.load(indices + chunk_i * stride_idxk).to(tl.int32)


        k_i_ptr = tl.make_block_ptr(
            base=K,
            shape=(HEAD_DIM, L_kv),
            strides=(stride_kd, stride_kl),
            offsets=(0, kv_block_idx * BLOCK_KV),
            block_shape=(HEAD_DIM, BLOCK_KV),
            order=(0, 1)
        )

        v_i_ptr = tl.make_block_ptr(
            base=V,
            shape=(L_kv, HEAD_DIM),
            strides=(stride_vl, stride_vd),
            offsets=(kv_block_idx * BLOCK_KV, 0),
            block_shape=(BLOCK_KV, HEAD_DIM),
            order=(1, 0)
        )

        D_block_ptr = tl.make_block_ptr(
            base=D + chunk_i.to(tl.int64) * stride_dk,
            shape=(L, GROUP_NUM),
            strides=(stride_dl, stride_dh),
            offsets=(m_id * BLOCK_M, 0),
            block_shape=(BLOCK_M, GROUP_NUM),
            order=(1, 0)
        )

        m_ptr = tl.make_block_ptr(
            base=M + chunk_i.to(tl.int64) * stride_mk,
            shape=(L, GROUP_NUM),
            strides=(stride_ml, stride_mh),
            offsets=(m_id * BLOCK_M, 0),
            block_shape=(BLOCK_M, GROUP_NUM),
            order=(1, 0)
        )

        ki = tl.load(k_i_ptr, boundary_check=(1,), padding_option="zero")  # (dim, block_kv)
        vi = tl.load(v_i_ptr, boundary_check=(0,), padding_option="zero")  # (block_kv, dim)
        mi = tl.load(m_ptr, boundary_check=(0,), padding_option="zero")  # (block_m, G)
        weight = tl.load(weights + chunk_i.to(tl.int64) * stride_wk).to(tl.float32)  # 1

        mi = mi.reshape(GROUP_NUM * BLOCK_M)  # (G * block_m, 1)

        qk = tl.dot(q, ki)  # (G * block_m, block_kv)
        # mi = tl.max(qk, 1)
        p = tl.exp2(qk - mi[:, None])  # (G * block_m, block_kv)
        o = tl.dot(p, vi.to(tl.float32))  # (G * block_m, dim)
        Di = tl.sum(o * do, 1)  # (G * block_m)
        tl.store(D_block_ptr, Di.reshape(BLOCK_M, GROUP_NUM), boundary_check=(0,))

        dp = tl.dot(do, tl.trans(vi)).to(tl.float32)   # (G * block_m, block_kv)
        ds = weight * p * (dp - Di[:, None])  # (G * block_m, block_kv)
        dq += tl.dot(ds, tl.trans(ki).to(tl.float32))  # (G * block_m, dim)

    dq *= LN2
    tl.store(dq_block_ptr, dq.reshape(BLOCK_M, GROUP_NUM, HEAD_DIM).to(DQ.type.element_ty), boundary_check=(0,))


@triton.jit
def _attn_bwd_dkdv(DK, DV,  #
                   Q, k, v, w, #
                   DO, M, D, kv_mask, rev_idx,
                   sm_scale, reg_lamda, reg_C,
                   # shared by Q/K/V/DO.
                   stride_dkz, stride_dkl, stride_dkh, stride_dkd,
                   stride_dvz, stride_dvl, stride_dvh, stride_dvd,
                   stride_qz, stride_ql, stride_qh, stride_qd,  #
                   stride_kz, stride_kl, stride_kh, stride_kd,  #
                   stride_vz, stride_vl, stride_vh, stride_vd,  #
                   stride_doz, stride_dol, stride_doh, stride_dod,
                   stride_wz, stride_wq, stride_wh, stride_wk,  # (N, q_blocks, h, K)
                   stride_mz, stride_mk, stride_ml, stride_mh,  #
                   stride_dz, stride_dk, stride_dl, stride_dh,  #
                   stride_mskz, stride_mskc, stride_mskh, stride_mskq,  #
                   stride_ridxz, stride_ridxc, stride_ridxh, stride_ridxq,  #
                   L,
                   BLOCK_Q: tl.constexpr,  #
                   BLOCK_KV: tl.constexpr,  #
                   BLOCK_M: tl.constexpr,
                   HEAD_DIM: tl.constexpr,  #
                   GROUP_NUM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    c_id = tl.program_id(0)  # chunk id for kv
    b_id = tl.program_id(1)
    h_id = tl.program_id(2)

    k_offs = b_id.to(tl.int64) * stride_kz + h_id.to(tl.int64) * stride_kh + c_id.to(tl.int64) * BLOCK_KV * stride_kl
    k += k_offs
    k_block_ptr = tl.make_block_ptr(
        base=k,
        shape=(BLOCK_KV, HEAD_DIM),
        strides=(stride_kl, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0)
    )

    v_offs = b_id.to(tl.int64) * stride_vz + h_id.to(tl.int64) * stride_vh + c_id.to(tl.int64) * BLOCK_KV * stride_vl
    v += v_offs
    v_block_ptr = tl.make_block_ptr(
        base=v,
        shape=(BLOCK_KV, HEAD_DIM),
        strides=(stride_vl, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0)
    )

    dk_offs = b_id.to(tl.int64) * stride_dkz + h_id.to(tl.int64) * stride_dkh + c_id.to(tl.int64) * BLOCK_KV * stride_dkl
    DK += dk_offs
    dk_block_ptr = tl.make_block_ptr(
        base=DK,
        shape=(BLOCK_KV, HEAD_DIM),
        strides=(stride_dkl, stride_dkd),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0)
    )

    dv_offs = b_id.to(tl.int64) * stride_dvz + h_id.to(tl.int64) * stride_dvh + c_id.to(tl.int64) * BLOCK_KV * stride_dvl
    DV += dv_offs
    dv_block_ptr = tl.make_block_ptr(
        base=DV,
        shape=(BLOCK_KV, HEAD_DIM),
        strides=(stride_dvl, stride_dvd),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0)
    )

    mask_offs = b_id.to(tl.int64) * stride_mskz + c_id.to(tl.int64) * stride_mskc + h_id.to(tl.int64) * stride_mskh
    kv_mask += mask_offs

    q_offs = b_id.to(tl.int64) * stride_qz + h_id.to(tl.int64) * stride_qh * GROUP_NUM
    Q += q_offs

    do_offs = b_id.to(tl.int64) * stride_doz + h_id.to(tl.int64) * stride_doh * GROUP_NUM
    DO += do_offs

    # M: (N, K, L, HQ)
    M_offs = b_id.to(tl.int64) * stride_mz + h_id.to(tl.int64) * stride_mh * GROUP_NUM
    M += M_offs

    # D: (N, K, L, HQ)
    D_offs = b_id.to(tl.int64) * stride_dz + h_id.to(tl.int64) * stride_dh * GROUP_NUM
    D += D_offs

    # w : (N, q_blocks, h, K)
    w_offs = b_id.to(tl.int64) * stride_wz + h_id.to(tl.int64) * stride_wh
    w += w_offs

    revidx_offs = b_id.to(tl.int64) * stride_ridxz + c_id.to(tl.int64) * stride_ridxc + h_id.to(tl.int64) * stride_ridxh
    rev_idx += revidx_offs

    k = tl.load(k_block_ptr, boundary_check=(0,), padding_option="zero")  # (block_kv, dim)
    v = tl.load(v_block_ptr, boundary_check=(0,), padding_option="zero")

    dv = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # m_start = (c_id.to(tl.int64) * BLOCK_KV + BLOCK_KV) // BLOCK_M
    m_start = 0
    m_blocks = L // BLOCK_M
    # tl.device_print('m_blocks', m_blocks)
    for block_m in range(m_start, m_blocks):
        q_i = block_m * BLOCK_M // BLOCK_Q
        m_i = tl.load(kv_mask + q_i.to(tl.int64) * stride_mskq)
        if m_i:
            K_idx = tl.load(rev_idx + q_i.to(tl.int64) * stride_ridxq)
            if K_idx >= 0:
                q_block_ptr = tl.make_block_ptr(
                    base=Q,
                    shape=(HEAD_DIM, L, GROUP_NUM),
                    strides=(stride_qd, stride_ql, stride_qh),
                    offsets=(0, block_m.to(tl.int32) * BLOCK_M, 0),
                    block_shape=(HEAD_DIM, BLOCK_M, GROUP_NUM),
                    order=(0, 2, 1)
                )
                do_block_ptr = tl.make_block_ptr(
                    base=DO,
                    shape=(L, GROUP_NUM, HEAD_DIM),
                    strides=(stride_dol, stride_doh, stride_dod),
                    offsets=(block_m.to(tl.int32) * BLOCK_M, 0, 0),
                    block_shape=(BLOCK_M, GROUP_NUM, HEAD_DIM),
                    order=(2, 1, 0)
                )
                m_block_ptr = tl.make_block_ptr(
                    base=M + K_idx.to(tl.int64) * stride_mk,
                    shape=(L, GROUP_NUM),
                    strides=(stride_ml, stride_mh),
                    offsets=(block_m.to(tl.int32) * BLOCK_M, 0),
                    block_shape=(BLOCK_M, GROUP_NUM),
                    order=(1, 0)
                )
                D_block_ptr = tl.make_block_ptr(
                    base=D + K_idx.to(tl.int64) * stride_dk,
                    shape=(L, GROUP_NUM),
                    strides=(stride_dl, stride_dh),
                    offsets=(block_m.to(tl.int32) * BLOCK_M, 0),
                    block_shape=(BLOCK_M, GROUP_NUM),
                    order=(1, 0)
                )
                weight = tl.load(w + q_i.to(tl.int64) * stride_wq + K_idx.to(tl.int64) * stride_wk).to(tl.float32)
                Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
                Di = Di.reshape(GROUP_NUM * BLOCK_M)
                m = tl.load(m_block_ptr, boundary_check=(0,), padding_option="zero")
                m = m.reshape(GROUP_NUM * BLOCK_M)

                qT = tl.load(q_block_ptr, boundary_check=(1,), padding_option="zero")  # (HEAD_DIM, BLOCK_M, GROUP_NUM)
                qT = qT.reshape(HEAD_DIM, GROUP_NUM * BLOCK_M)  # (HEAD_DIM, (BLOCK_M * GROUP_NUM))

                do = tl.load(do_block_ptr, boundary_check=(0,))
                do = do.reshape(GROUP_NUM * BLOCK_M, HEAD_DIM)

                qkT = tl.dot(k, qT)  # (BLOCK_KV, BLOCK_M * GROUP_NUM), k has already been scaled
                pT = tl.exp2(qkT - m[None, :])  # (BLOCK_KV, BLOCK_M * GROUP_NUM)
                # if block_m == m_blocks - 1 and c_id == 0:
                #     tl.device_print('weight', weight)
                # dk += tl.sum(qT, 1) #tl.sum(qkT, 1)
                # tl.device_assert(tl.min(tl.sum(pT, 0) <= 1.1), msg='dkdv p sum > 1')
                pT = weight * pT  # (BLOCK_KV, BLOCK_M * GROUP_NUM)
                ppT = pT  # (BLOCK_KV, BLOCK_M * GROUP_NUM)
                ppT = ppT.to(do.type.element_ty)  #  (BLOCK_KV, BLOCK_M * GROUP_NUM)
                dv += tl.dot(ppT, do)  # (BLOCK_KV, dim)

                dpT = tl.dot(v, tl.trans(do)) # (BLOCK_KV, M)
                dsT = pT * (dpT - Di[None, :])  # (BLOCK_KV, M)
                dsT = dsT.to(qT.type.element_ty)
                dk += tl.dot(dsT, tl.trans(qT))  # (BLOCK_KV, dim)

    dk *= sm_scale
    tl.store(dk_block_ptr, dk.to(DK.type.element_ty))
    tl.store(dv_block_ptr, dv.to(DV.type.element_ty))


@triton.jit
def _fill_chunk_mask(
    indice_ptr,
    mask_ptr,
    reverse_idx_ptr,
    K,
    stride_idxz, stride_idxq, stride_idxh, stride_idxk,
    stride_mskz, stride_mskk, stride_mskh, stride_mskq,
    stride_rvidxz, stride_rvidxk, stride_rvidxh, stride_rvidxq,
    H: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr
):
    q_blk_id = tl.program_id(0)
    h_id = tl.program_id(1)
    b_id = tl.program_id(2)

    prev_idx = -1
    for k_blk_id in range(K):
        idx_offset = b_id.to(tl.int64) * stride_idxz + q_blk_id.to(tl.int64) * stride_idxq + k_blk_id.to(tl.int64) * stride_idxk + h_id.to(tl.int64) * stride_idxh

        kv_blk_idx = tl.load(indice_ptr + idx_offset).to(tl.int32)
        if kv_blk_idx != prev_idx:
            prev_idx = kv_blk_idx
            # q_offset = q_blk_id.to(tl.int64) * BLOCK_Q
            k_end = kv_blk_idx * BLOCK_KV + BLOCK_KV

            mask_offset = b_id.to(tl.int64) * stride_mskz + kv_blk_idx * stride_mskk + q_blk_id.to(tl.int64) * stride_mskq + h_id.to(tl.int64) * stride_mskh
            rv_idx_offset = b_id.to(tl.int64) * stride_rvidxz + kv_blk_idx * stride_rvidxk + q_blk_id.to(tl.int64) * stride_rvidxq + h_id.to(tl.int64) * stride_rvidxh

            tl.store(mask_ptr + mask_offset, True)
            tl.store(reverse_idx_ptr + rv_idx_offset, k_blk_id)


def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, weights, indices, sm_n, chunk_size, sm_scale, l2_lamda, l2_C):
        # q: (N, L, h, d)
        # k: (N, L, h, d)
        # weights: (N, blocks, K)
        # indices: (N, blocks, K)
        # print(f'call fwd kernel')
        assert not torch.any(torch.isnan(q))
        assert not torch.any(torch.isnan(k))
        assert not torch.any(torch.isnan(v))
        assert not torch.any(torch.isnan(weights))
        # assert torch.all(weights.sum(dim=-1) <= 1.1)
        assert torch.all(indices >= 0)
        N, L, HQ, HEAD_DIM = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
        L_kv = k.shape[1]
        assert L_kv % chunk_size == 0
        assert indices.is_contiguous()
        assert weights.is_contiguous()
        assert q.is_contiguous()
        assert k.is_contiguous()
        assert v.is_contiguous()
        if sm_scale is None:
            sm_scale = 1 / math.sqrt(HEAD_DIM)
        H = k.shape[2]
        K = indices.shape[-1]
        block_num = indices.shape[1]
        assert q.shape[1] % block_num == 0
        Q_X = q.shape[1] // block_num
        BLOCK_M = min(Q_X, 16)
        assert HQ % H == 0 and HQ // H * BLOCK_M >= 16, f'{HQ} // {H} * {BLOCK_M} < 16'
        GROUP_NUM = HQ // H
        M = torch.empty((N, K, L, HQ), device=q.device, dtype=torch.float32)
        o = torch.empty_like(q)
        assert weights.shape == indices.shape
        grid = (L // BLOCK_M, N, H)
        _attn_fwd[grid](
            q, k, v, weights, indices, sm_scale, sm_n, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            weights.stride(0), weights.stride(1), weights.stride(2), weights.stride(3),
            indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            L, L_kv, GROUP_NUM, K,
            HEAD_DIM,
            BLOCK_M,
            chunk_size,
            Q_X
        )

        ctx.save_for_backward(q, k, v, weights, M, indices)
        ctx.chunk_size = chunk_size
        ctx.sm_scale = sm_scale
        ctx.l2_lamda = l2_lamda
        ctx.l2_C = l2_C

        assert not torch.any(torch.isnan(o))
        return o

    @staticmethod
    def backward(ctx, do):
        # generate chunk mask
        q, k, v, weights, M, indices = ctx.saved_tensors
        l2_lamda = ctx.l2_lamda
        l2_C = ctx.l2_C
        # indices: (N, Q blocks, K)
        chunk_size = ctx.chunk_size
        sm_scale = ctx.sm_scale
        BATCH, L, HQ, HEAD_DIM = q.shape
        L_kv = k.shape[1]
        H = k.shape[2]
        GROUP_NUM = HQ // H
        KV_CHUNK = L_kv // chunk_size
        K = weights.shape[-1]
        q_chunks = indices.shape[1]
        BLOCK_Q = q.shape[1] // q_chunks
        block_mask = torch.zeros(BATCH, KV_CHUNK, H, q_chunks, dtype=torch.bool, device=indices.device)
        rev_idx = torch.zeros(BATCH, KV_CHUNK, H, q_chunks, dtype=torch.int32, device=indices.device).fill_(-1)
        grid = (q_chunks, H, BATCH)

        _fill_chunk_mask[grid](
            indices, block_mask, rev_idx, K,
            indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
            block_mask.stride(0), block_mask.stride(1), block_mask.stride(2),block_mask.stride(3),
            rev_idx.stride(0), rev_idx.stride(1), rev_idx.stride(2), rev_idx.stride(3),
            H,
            BLOCK_Q, chunk_size
        )


        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (sm_scale * RCP_LN2)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        D = torch.zeros(BATCH, K, L, HQ, device=q.device, dtype=torch.float32)
        BLOCK_M = min(BLOCK_Q, 16)
        grid = (L // BLOCK_M, BATCH, H)

        scaled_reg_lambda = l2_lamda / (BATCH * L * K * chunk_size * HQ)
        kernel_attn_bwd_dq[grid](
            q, arg_k, v, weights,
            do, dq, M, D,
            indices, scaled_reg_lambda, l2_C,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            weights.stride(0), weights.stride(1), weights.stride(2), weights.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            D.stride(0), D.stride(1), D.stride(2), D.stride(3),
            indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
            L, L_kv, K,
            GROUP_NUM, BLOCK_Q, chunk_size, BLOCK_M, HEAD_DIM
        )

        grid = (L_kv // chunk_size, BATCH, H)
        # print(f'weight: {weights[0, -1, 0, :]}, indices: {indices[0, -1, 0, :]}')
        _attn_bwd_dkdv[grid](
            dk, dv,
            q, arg_k, v, weights,
            do, M, D, block_mask, rev_idx,
            sm_scale, scaled_reg_lambda, l2_C,
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            weights.stride(0), weights.stride(1), weights.stride(2), weights.stride(3),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            D.stride(0), D.stride(1), D.stride(2), D.stride(3),
            block_mask.stride(0), block_mask.stride(1), block_mask.stride(2), block_mask.stride(3),
            rev_idx.stride(0), rev_idx.stride(1), rev_idx.stride(2), rev_idx.stride(3),
            L,
            BLOCK_Q, chunk_size, BLOCK_M, HEAD_DIM, GROUP_NUM
        )

        delta_w = rearrange(D, 'B K (C Q) (h G)-> B C (Q G) h K', Q=BLOCK_Q, h=H)
        assert delta_w.dtype == torch.float32
        dw = delta_w.sum(dim=2)  # (N, C, h, K)
        assert dw.dtype == torch.float32

        return dq, dk, dv, dw, None, None, None, None, None, None


# attention = _attention.apply
def HSA(q, k, v, weights, indices, sm_n, chunk_size, sm_scale, reg_lamda=0.01, reg_C=50.0):
    return _attention.apply(q, k, v, weights, indices, sm_n, chunk_size, sm_scale, reg_lamda, reg_C)

def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg

# @pytest.mark.parametrize("Z, HQ, H, N_CTX, HEAD_DIM, K, S, Q_X", [(2, 16, 1, 256, 64, 4, 64, 1), (2, 16, 4, 256, 64, 4, 64, 16), (1, 16, 1, 2048, 16, 4, 32, 1)])
@pytest.mark.parametrize("Z, HQ, H, N_CTX, HEAD_DIM, K, S, Q_X", [(1, 16, 1, 32, 16, 2, 16, 1)])#, (2, 32, 2, 1024, 16, 4, 32, 1), (2, 32, 2, 1324, 16, 4, 32, 1), (2, 32, 2, 1024, 32, 4, 64, 1), (2, 32, 2, 1324, 32, 4, 64, 1)])
def test_group_qa(Z, HQ, H, N_CTX, HEAD_DIM, K, S, Q_X, dtype=torch.float32):
    import torch.nn.functional as F
    import os
    torch.manual_seed(22)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    torch.set_printoptions(threshold=float('inf'))
    q = (torch.empty((Z, N_CTX, HQ, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.05).requires_grad_())
    k = (torch.empty((Z, N_CTX // S * S, H, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.05).requires_grad_())
    v = (torch.empty((Z, N_CTX // S * S, H, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.05).requires_grad_())


    # print(f'v shape: {v.shape}')
    chunk_num = N_CTX // S
    q_blocks = N_CTX // Q_X
    RCP_LN2 = 1.4426950408889634
    # indices = torch.randint(0, chunk_num, size=(Z, q_blocks, H, K), device='cuda')
    indices = torch.arange(0, K).unsqueeze(0).unsqueeze(0).repeat(Z, q_blocks, 1)

    for i in range(Z):
        for j in range(q_blocks):
            max_visible_k_block = min(j * Q_X // S, K)
            if max_visible_k_block > 0:
                indices[i, j, max_visible_k_block - 1] = 0
                indices[i, j, 0] = max_visible_k_block - 1
    # print(indices)
    indices = indices.unsqueeze(2).expand(-1, -1, H, -1)

    print(f'indices: {indices}')

    scores = torch.tensor(torch.rand(Z, q_blocks, H, K), dtype=dtype, device='cuda')
    # scores = torch.tensor(torch.zeros(Z, q_blocks, H, K), dtype=dtype, device='cuda')
    score_mask = torch.zeros(Z, q_blocks, H, K, dtype=torch.bool)
    for i in range(Z):
        for j in range(q_blocks):
            for h in range(H):
                for _k in range(K):
                    q_offset = j * Q_X
                    k_end = indices[i, j, h, _k] * S + S
                    score_mask[i, j, h, _k] = q_offset >= k_end
    # score_mask = score_mask.expand(-1, -1, H, -1)
    scores.masked_fill_(~score_mask.cuda(), float('-inf'))

    # (N, q_blocks, K)
    weights = F.softmax(scores.to(torch.float32), dim=-1).nan_to_num_(0.0).requires_grad_()
    # print(f'weights: {weights}')

    sm_scale = 1 / math.sqrt(HEAD_DIM)
    dout = torch.randn_like(q)
    assert dout.dtype == torch.float32

    chunked_k = rearrange(k, 'Z (N S) h d->Z N S h d', S=S)
    chunked_v = rearrange(v, 'Z (N S) h d->Z N S h d', S=S)
    h = chunked_k.shape[-2]
    d = chunked_k.shape[-1]

    indices = indices.cuda()
    indices_ = rearrange(indices, 'Z B h K->Z (B K) h')
    indices_ = indices_.unsqueeze(2).unsqueeze(-1)
    indices_expanded = indices_.expand(-1, -1, S, -1, d)
    gather_k = chunked_k.gather(dim=1, index=indices_expanded)
    gather_v = chunked_v.gather(dim=1, index=indices_expanded)
    # print(f'gather v shape: {gather_v.shape}')

    gather_k = rearrange(gather_k, 'Z (B K) S h d->Z B S K h d', K=K)
    gather_v = rearrange(gather_v, 'Z (B K) S h d->Z B S K h d', K=K)

    k_ = torch.repeat_interleave(gather_k, dim=-2, repeats=HQ // H)  # (Z B K S 4*h d)
    v_ = torch.repeat_interleave(gather_v, dim=-2, repeats=HQ // H)  # (Z B K S 4*h d)
    q_chunked = rearrange(q, 'Z (B X) h d->Z B X h d', X=Q_X)

    qk = torch.einsum('Z B X h d, Z B S K h d->Z B X S K h', q_chunked.float(), k_.float()) * sm_scale
    ref_M = qk.logsumexp(dim=-3) * 1.4426950408889634  # (Z B X K h)
    ref_M = rearrange(ref_M, 'Z B X K h->Z K (B X) h')
    def softmax_off_by_one(tensor, sm_n=0.0, dim=-1):
        max_val, _ = tensor.max(dim=dim, keepdim=True)
        exp_val = torch.exp(tensor - max_val)
        val_sum = exp_val.sum(dim=dim, keepdim=True) + sm_n * torch.exp(-max_val)
        return exp_val / val_sum

    assert qk.dtype == torch.float32
    p = softmax_off_by_one(qk, sm_n=1.0, dim=-3)  # Z B X S K h
    assert p.dtype == torch.float32
    o_k = torch.einsum('Z B X S K h, Z B S K h d->Z B X K h d', p, v_.float())
    print(f'o_k shape: {o_k.shape}')
    w_exp = torch.repeat_interleave(weights, dim=-2, repeats=HQ // H)

    ref_o = torch.einsum('Z B X K h d, Z B h K->Z B X h d', o_k, w_exp.float())
    ref_o = rearrange(ref_o, 'Z B X h d->Z (B X) h d')
    assert ref_o.dtype == torch.float32

    ref_o.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    ref_dw, weights.grad = weights.grad.clone(), None

    tri_o = HSA(q, k, v, weights, indices, 1.0, S, sm_scale, reg_lamda=0.0, reg_C=0.0)
    tri_o.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    tri_dw, weights.grad = weights.grad.clone(), None

    print(f'tri o dtype: {tri_o.dtype}, ref_o dtype: {ref_o.dtype}')

    print(f'tri_o: {tri_o[0, -5:, 0, :5]}')
    print(f'ref_o: {ref_o[0, -5:, 0, :5]}')
    print(f'tri_dq: {tri_dq[0, :, 0, :5]}')
    print(f'ref_dq: {ref_dq[0, :, 0, :5]}')
    print(f'tri_dk: {tri_dk[0, -5:, 0, :5]}')
    print(f'ref_dk: {ref_dk[0, -5:, 0, :5]}')
    # print(f'weights: {weights[0, -5:, -1]}')
    # print(f'indices: {indices[0, -5:, :].sum(dim=-1)}')
    o_k = rearrange(o_k, 'Z B X K h d->Z (B X) K h d')
    print(f'o_k: {o_k[0, -5:, -1, 0, :5]}')
    # Z B S K h d
    print(f'gather_v: {gather_v[0, -5:, :, -1, 0, :5].sum(dim=1)}')

    assert_close('o', tri_o, ref_o, 1)
    assert_close('dw', tri_dw, ref_dw, 1)
    assert_close('dq', tri_dq, ref_dq, 1)
    assert_close('dk', tri_dk, ref_dk, 1)
    assert_close('dv', tri_dv, ref_dv, 1)


if __name__ == "__main__":
    test_group_qa(Z=1, HQ=16, H=1, N_CTX=4096, HEAD_DIM=64, K=8, S=16, Q_X=1)
