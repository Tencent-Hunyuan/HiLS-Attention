import torch.nn as nn
import math
from ops.hsa_triton import HSA
# from flash_attn.ops.rms_norm import RMSNorm
from .swa import SlidingWindowAttention
from liger_kernel.transformers.rms_norm import LigerRMSNorm as RMSNorm
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
ALL_LAYERNORM_LAYERS.append(RMSNorm)
import torch.distributed as dist
from typing import Optional
from einops import rearrange, einsum, repeat
import torch
import torch.nn.functional as F
from .gather_with_grad import AllGatherWithGradient
from veomni.distributed.parallel_state import get_parallel_state
import warnings

def softmax_off_by_one(tensor, sm_n=1.0, dim=-1):
    max_val, _ = tensor.max(dim=dim, keepdim=True)
    max_val = torch.where(max_val == float('-inf'), 0, max_val)
    exp_val = torch.exp(tensor - max_val)
    val_sum = exp_val.sum(dim=dim, keepdim=True) + sm_n * torch.exp(-max_val)
    return exp_val / val_sum

class ChunkKVManager:
    def __init__(self, config, batch_size):
        self.batch_size = batch_size
        offloading = getattr(config, "offloading_to_cpu", False)
        group_size = config.num_upper_groups
        self.chunk_k = None  # (N, D, S, dim)
        self.chunk_v = None
        self.lmk_embs = None
        self.group_size = group_size
        self.head_num = config.num_kv_heads
        self.retrieval_head_num = config.num_kv_heads
        if getattr(config, 'singlehead_retrieval', False):
            self.retrieval_head_num = 1
        self.retrieval_dim = config.retrieval_dim
        assert config.hidden_size % config.num_attention_heads == 0
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.chunk_topk = config.chunk_topk
        self.chunk_size = config.chunk_size
        self._current_chunk_k = [None for _ in range(group_size)]
        self._current_chunk_v = [None for _ in range(group_size)]
        self._current_weights = [None for _ in range(group_size)]

        self.varlen_chunk_nums = [0 for _ in range(self.batch_size)]  # chunk_num = used + padding
        self.varlen_used_chunk_nums = [0 for _ in range(self.batch_size)]
        # self.varlen_chunk_masks = None
        self.padding_lens = None
        
        self._current_hidden_states = None
        self._offloading = offloading

        self._indices_cache = None
        self._k_cache = None
        self._v_cache = None

    @property
    def enable_offloading(self):
        return self._offloading

    def append_varlen(self, b_ids, chunk_nums, chunk_k, chunk_v, landmarks):
        # b_ids: (n), n <= self.batch_size
        # chunk_nums: list with length n
        # chunk_k: (?, S, dim)
        # chunk_v: (?, S, dim)
        # landmarks: (N, C, h, retrieval_dim)

        device = chunk_k.device
        current_max_chunk_num = max(self.varlen_chunk_nums)
        # recompute max chunk nums
        for local_idx, b_id in enumerate(b_ids):
            chunk_num = chunk_nums[local_idx]
            self.varlen_chunk_nums[b_id] += chunk_num

        max_chunk_num = max(self.varlen_chunk_nums)
        pad_chunk_num = max_chunk_num - current_max_chunk_num

        if self.chunk_k is None:
            self.chunk_k = torch.zeros(
                self.batch_size, 
                pad_chunk_num, 
                self.chunk_size, 
                self.head_dim * self.head_num, 
                device=device,
                dtype=chunk_k.dtype)  # (N, D, S, h * dim)
        else:
            padding = torch.zeros(
                self.batch_size, 
                pad_chunk_num, 
                self.chunk_size, 
                self.head_dim * self.head_num, 
                device=device,
                dtype=chunk_k.dtype)
            self.chunk_k = torch.cat([self.chunk_k, padding], dim=1)
        
        if self.chunk_v is None:
            self.chunk_v = torch.zeros(
                self.batch_size, 
                pad_chunk_num, 
                self.chunk_size, 
                self.head_dim * self.head_num, 
                device=device,
                dtype=chunk_v.dtype)
        else:
            padding = torch.zeros(
                self.batch_size, 
                pad_chunk_num, 
                self.chunk_size, 
                self.head_dim * self.head_num, 
                device=device,
                dtype=chunk_v.dtype)
            self.chunk_v = torch.cat([self.chunk_v, padding], dim=1)
        
        # if self.varlen_chunk_masks is None:
        #     self.varlen_chunk_masks = torch.torch.zeros(self.batch_size, pad_chunk_num, device=device).fill_(float('-inf'))
        # else:
        #     padding = torch.torch.zeros(self.batch_size, pad_chunk_num, device=device).fill_(float('-inf'))
        #     self.varlen_chunk_masks = torch.cat([self.varlen_chunk_masks, padding], dim=1)

        if self.lmk_embs is None:
            self.lmk_embs = torch.zeros(self.batch_size, pad_chunk_num, self.retrieval_head_num, self.retrieval_dim // self.retrieval_head_num, device=device, dtype=landmarks.dtype)
        else:
            padding = torch.zeros(self.batch_size, pad_chunk_num, self.retrieval_head_num, self.retrieval_dim // self.retrieval_head_num, device=device, dtype=landmarks.dtype)
            self.lmk_embs = torch.cat([self.lmk_embs, padding], dim=1)

        chunk_idx_offset = 0
        for local_idx, b_id in enumerate(b_ids):
            chunk_num = chunk_nums[local_idx]
            used_chunks = self.varlen_used_chunk_nums[b_id]
            # print(f'self.chunk_k shape: {self.chunk_k.shape} chunk_k: {chunk_k.shape}')
            # print(f'used chunks: {used_chunks}, chunk_num: {chunk_num}, chunk_idx_offset: {chunk_idx_offset}/{chunk_k.shape[0]}')
            # print(f'{self.chunk_k[b_id, used_chunks: used_chunks + chunk_num, :, :].shape} vs {chunk_k[chunk_idx_offset: chunk_idx_offset + chunk_num, :, :].shape}')
            self.chunk_k[b_id, used_chunks: used_chunks + chunk_num, :, :] = chunk_k[chunk_idx_offset: chunk_idx_offset + chunk_num, :, :]
            self.chunk_v[b_id, used_chunks: used_chunks + chunk_num, :, :] = chunk_v[chunk_idx_offset: chunk_idx_offset + chunk_num, :, :]
            self.lmk_embs[b_id, used_chunks: used_chunks + chunk_num, :, :] = landmarks[chunk_idx_offset: chunk_idx_offset + chunk_num, :, :]
            # self.varlen_chunk_masks[b_id, used_chunks: used_chunks + chunk_num] = 0
            # assert torch.all(self.varlen_chunk_masks[b_id, :used_chunks + chunk_num] == 0)
            chunk_idx_offset += chunk_num
            self.varlen_used_chunk_nums[b_id] += chunk_num
        


    def append(self, chunk_k, chunk_v, lmk_embs):
        # chunk_k: (N, D, S, h * dim)
        # lmk_embs: (N, D, h, dim)
        # assert chunk_k.shape[0] == 1
        assert len(chunk_k.shape) == 4
        assert len(chunk_v.shape) == 4
        assert len(lmk_embs.shape) == 4
        assert chunk_k.shape[1] == lmk_embs.shape[1], f'chunk_k shape: {chunk_k.shape}, lmk_embs shape: {lmk_embs.shape}'
        # if self._offloading:
        #     chunk_k = chunk_k.cpu()
        #     chunk_v = chunk_v.cpu()
            # still keep lmk_embs in GPU
        if not self._offloading:
            if self.chunk_k is None:
                self.chunk_k = chunk_k # (N, D, S, h, dim)
            else:
                assert chunk_k.shape[-2] == self.chunk_k.shape[-2]
                # print(f'chunk_k shape: {self.chunk_k.shape} vs {chunk_k.shape}')
                self.chunk_k = torch.cat([self.chunk_k, chunk_k], dim=1)
            
            if self.chunk_v is None:
                self.chunk_v = chunk_v
            else:
                assert chunk_v.shape[-2] == self.chunk_v.shape[-2]
                self.chunk_v = torch.cat([self.chunk_v, chunk_v], dim=1)
            
        else:
            if self.chunk_k is None:
                self.chunk_k = [[] for _ in range(self.batch_size)]
            if self.chunk_v is None:
                self.chunk_v = [[] for _ in range(self.batch_size)]
            
            chunk_to_process = [chunk_k, chunk_v]
            chunk_to_fill = [self.chunk_k, self.chunk_v]

            for i, current_chunk in enumerate(chunk_to_process):
                current_chunk = rearrange(current_chunk, 'N D S (h d) -> N D S h d', h=self.head_num)
                for batch_i in range(current_chunk.shape[0]):
                    # self.chunk_k.append(
                    #     [chunk_k[batch_i, chunk_i].to('cpu', non_blocking=True) for chunk_i in range(chunk_k.shape[1])]
                    # )
                    for chunk_i in range(current_chunk.shape[1]):
                        chunks_per_head = []
                        for head_i in range(self.head_num):
                            chunks_per_head.append(current_chunk[batch_i, chunk_i, :, head_i, :].to('cpu', non_blocking=True))
                        chunk_to_fill[i][batch_i].append(chunks_per_head)
        

        if self.lmk_embs is None:
            self.lmk_embs = lmk_embs
        else:
            self.lmk_embs = torch.cat([self.lmk_embs, lmk_embs], dim=1)

    @property
    def past_lmk_embeds(self):
        return self.lmk_embs

    @property
    def mem_len(self):
        if self.chunk_k is None:
            return 0
        else:
            assert len(self.chunk_k.shape) == 4
            return self.chunk_k.shape[1] * self.chunk_k.shape[2]

    def current_retrieved_chunk(self, group_idx=0):
        return self._current_chunk_k[group_idx], self._current_chunk_v[group_idx], self._current_weights[group_idx]

    @property
    def lower_hidden_states(self):
        return self._current_hidden_states

    def clear_lower_hidden_states(self):
        self._current_hidden_states = None

    def cache_lower_hidden_states(self, hidden_states):
        # hidden_states: (N, L, dim)
        if self._current_hidden_states is None:
            self._current_hidden_states = hidden_states
        else:
            self._current_hidden_states = torch.cat(
                [self._current_hidden_states, hidden_states], dim=1
            )

    def retrieve_chunks(self, indices):
        # indices: (N, h, chunk_num)
        if self._indices_cache is not None and torch.all(indices == self._indices_cache):
            # print(f'hit, directly return')
            return self._k_cache, self._v_cache
        self._indices_cache = indices
        N = indices.shape[0]
        K = indices.shape[-1]
        if not self._offloading:
            batch_indices = torch.arange(N, device=indices.device).unsqueeze(1)
            # print(f'self.chunk_k shape: {self.chunk_k.shape}, indices: {indices}')
            k = self.chunk_k[batch_indices, indices]  # (N, (C K), S, d)
            v = self.chunk_v[batch_indices, indices]
            # print(f'retrieved k: {k.shape}')
            self._k_cache = k
            self._v_cache = v
            return k, v
        else:
            org_device = indices.device
            indices = indices.cpu().numpy()
            gathered_chunk_k = []
            gathered_chunk_v = []
            # print(f'chunk k shape: {len(self.chunk_k)}, {len(self.chunk_k[0])}, {len(self.chunk_k[0][0])}')
            # print(f'indices: {indices}')
            for batch_i in range(indices.shape[0]):
                chunk_vs = []
                chunk_ks = []
                for k in range(K):
                    chunk_ks_h = []
                    chunk_vs_h = []
                    for h in range(self.head_num):
                        chunk_ks_h.append(self.chunk_k[batch_i][indices[batch_i, h, k]][h].to(org_device))
                        chunk_vs_h.append(self.chunk_v[batch_i][indices[batch_i, h, k]][h].to(org_device))
                    chunk_ks_h = torch.stack(chunk_ks_h, dim=1)  # (S, H, dim)
                    chunk_vs_h = torch.stack(chunk_vs_h, dim=1)
                    # print(f'chunk ks h shape: {chunk_ks_h.shape}')
                    chunk_ks.append(chunk_ks_h)
                    chunk_vs.append(chunk_vs_h)
                chunk_ks = torch.stack(chunk_ks, dim=0)
                chunk_vs = torch.stack(chunk_vs, dim=0)  # (K, S, h, dim)
                gathered_chunk_k.append(chunk_ks)
                gathered_chunk_v.append(chunk_vs)
            chunk_k = torch.stack(gathered_chunk_k, dim=0) # (N, K, S, h, dim)
            chunk_v = torch.stack(gathered_chunk_v, dim=0)
            self._k_cache = chunk_k
            self._v_cache = chunk_v
            return chunk_k, chunk_v

    def reorder(self, beam_ids):
        # warning: not tested!
        self.chunk_k = self.chunk_k.index_select(0, beam_idx.to(self.chunk_k.device))
        self.chunk_v = self.chunk_v.index_select(0, beam_idx.to(self.chunk_v.device))
        self.lmk_embs = self.lmk_embs.index_select(0, beam_idx.to(self.lmk_embs.device))

        for group_i in range(self.group_size):
            self._current_chunk_k[group_i] = self._current_chunk_k[group_i].index_select(
                0, beam_idx.to(self._current_chunk_k.device)
            )
            self._current_chunk_v[group_i] = self._current_chunk_v[group_i].index_select(
                0, beam_idx.to(self._current_chunk_v.device)
            )

class HSACache:
    def __init__(self, config, batch_size: int, dtype: torch.dtype = torch.bfloat16, device: Optional[str] = None):
        head_dim = config.hidden_size // config.num_attention_heads
        self.chunk_k = torch.zeros(batch_size, config.num_kv_heads, config.chunk_topk, (config.chunk_size + 1) // 64 * 64, head_dim, device=device, dtype=dtype)  # (N C) h K S d
        self.chunk_v = torch.zeros(batch_size, config.num_kv_heads, config.chunk_topk, (config.chunk_size + 1) // 64 * 64, head_dim, device=device, dtype=dtype)
        self.weights = torch.zeros(batch_size, config.chunk_topk, device=device, dtype=dtype)

        self.key_value_memory_dict = {}
        self.mem_mgr = ChunkKVManager(config, batch_size)
        self.landmark_positions = None
        self.replay_mask = None
        self.replay_query_emb = None


class HierarchicalSparseAttention(nn.Module):
    """ Grouped Cross-Attention """

    def __init__(
        self,
        config,
        layer_idx=None,
        group_idx=None,
        device=None,
        mlp_cls=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.embed_dim // config.num_attention_heads
        self.chunk_size = config.chunk_size
        self.group_idx = group_idx
        self.layer_idx = layer_idx
        self.pre_norm = RMSNorm(self.embed_dim)
        # self.sm_n = 0.0 if (self.chunk_size + 1) % 64 == 0 else 1.0
        self.enable_softmax1 = getattr(config, 'enable_softmax1', False)
        if self.enable_softmax1:
            self.sm_n = 1.0
        else:
            self.sm_n = 0.0
        self.reg_lamda = getattr(config, 'reg_lamda', 0.0)
        self.reg_C = getattr(config, 'reg_C', 50.0)
        self.flash_inference = getattr(config, 'flash_inference', False)
        self._offloading = getattr(config, 'offloading_to_cpu', False)
        self.enable_qk_norm = getattr(config, 'enable_qk_norm', False)
        self.hsa_only = getattr(config, 'hsa_only', True)
        self.skip_hsa = getattr(config, 'skip_hsa', False)
        if not self.hsa_only:
            self.swa_prenorm = RMSNorm(self.embed_dim)
            self.attn = SlidingWindowAttention(config, layer_idx)

        if getattr(config, 'hsa_sliding_window', None) is not None:
            self.attn.sliding_window = config.hsa_sliding_window

        if self.enable_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
        
        if mlp_cls is not None:
            self._mlp_norm = RMSNorm(self.embed_dim)
            self._mlp = mlp_cls(config)
        else:
            self._mlp_norm = None
            self._mlp = None
        self._reset_hsa_residual = getattr(config, 'reset_hsa_residual', False)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False, **factory_kwargs)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False, **factory_kwargs)
        self.__init_weights(config)
        print(f'HSA config: self.sm_n: {self.sm_n}, reg_lamda: {self.reg_lamda}, l2_C: {self.reg_C}')


    def __init_weights(self, config):
        std = config.initializer_range
        self.q_proj.weight.data.normal_(mean=0.0, std=std)
        self.o_proj.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        hidden_states,
        weights:Optional[torch.Tensor]=None,
        mem_k:Optional[torch.Tensor]=None,
        mem_v:Optional[torch.Tensor]=None,
        landmarks:Optional[torch.Tensor]=None,
        indices: Optional[torch.Tensor]=None,
        cache_position:Optional[torch.LongTensor]=None,
        position_embeddings: Optional[torch.Tensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        cache_params=None,
    ):
        # print(f'enter {self.layer_idx} layer, type: hsa')
        # print(f'mh inputs indices: {indices}')
        residual = hidden_states

        if not self.hsa_only:
            # apply swa
            o = self.attn(
                self.swa_prenorm(hidden_states), 
                past_key_value=None if cache_params is None else cache_params.past_key_values, 
                position_embeddings=position_embeddings,
            )
            residual = hidden_states + o
            hidden_states = residual

        if self.skip_hsa:
            return hidden_states, weights, mem_k, mem_v, landmarks, indices

        hidden_states = self.pre_norm(hidden_states.to(self.q_proj.weight.dtype))
        q = self.q_proj(hidden_states)
        N = q.shape[0]

        if cache_params is None:
            assert hidden_states.shape[1] % self.chunk_size == 0
            # q: (N, L, (h d))
            # print(f'batch q repr: {q[0, -4:, :4]}, hidden: {hidden_states[0, -4:, :4]}')
            q = rearrange(q, 'N L (h d)->N L h d', h=self.num_heads)
            if self.enable_qk_norm:
                q = self.q_norm(q)
            
            # print(f'indices: {indices}')
            # print(f'weights: {weights}')
            # print(f'q shape: {q.shape}, {q.device} k shape: {mem_k.shape} {mem_k.device}, v shape: {mem_v.shape} {mem_v.device}, indices: {indices.shape} {indices.device} weights: {weights.shape} {weights.device}')
            context = HSA(q, mem_k, mem_v, weights, indices, self.sm_n, self.chunk_size, None, reg_lamda=self.reg_lamda, reg_C=self.reg_C)  # (N, L, h, d)
            # print(f'batch mode context: {context[0, -self.chunk_size:, :5]}')
            context = rearrange(context, 'N L h d->N L (h d)')
            out = self.o_proj(context)
            # print(f'out dtype: {out.dtype}')
            # print(f'HSA output: {out[:, :, :4]}')

            # print(f'has mlp? : {self._mlp is not None}')
            if self._mlp is not None and self._mlp_norm is not None:
                h = self._mlp_norm(residual + out)
                if self._reset_hsa_residual:
                    residual = residual + out
                out = self._mlp(h)
            # print('hsa mlp over')
            # print(f'batch hsa output: {out[0, 128:130, :4]}, input:{q[0, 128:130, 0, :4]}')
            return residual + out, weights, mem_k, mem_v, landmarks, indices
            # return residual, weights, mem_k, mem_v, landmarks, indices
        else:
            # print(cache_position)
            # if q.shape[1] > 1:
            #     print(f'inf q repr: {q[0, -4:, :4]}, hidden: {hidden_states[0, -4:, :4]}')
            # elif q.shape[1] == 1:
            #     print(f'inf q repr: {q[0, :, :4]}, hidden: {hidden_states[0, :, :4]}')
            if weights is not None and indices is not None and (q.shape[1] == 1 or not self.flash_inference):
                q = rearrange(q, 'N L (h d)->N L h d', h=self.num_heads)
                if self.enable_qk_norm:
                    q = self.q_norm(q)
                # print(f'q repr: {q[0, :, 0, :4]}')
                if not self._offloading or q.shape[1] > 1:
                    context = HSA(q, mem_k, mem_v, weights, indices, self.sm_n, self.chunk_size, None, reg_lamda=self.reg_lamda, reg_C=self.reg_C)  # (N, L, h, d)
                else:
                    kv_mgr = cache_params.mem_mgr
                    chunk_k, chunk_v = kv_mgr.retrieve_chunks(indices.squeeze(1))  # (N, K, S, h d)
                    chunk_k = rearrange(chunk_k, 'N K S h d->N (K S) h d', d=self.head_dim)
                    chunk_v = rearrange(chunk_v, 'N K S h d->N (K S) h d', d=self.head_dim)
                    # print(f'retrieved chunk k: {chunk_k.shape}')
                    indices_ = torch.arange(0, indices.shape[-1], device=indices.device)
                    indices_ = indices_.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(N, 1, self.num_kv_heads, -1).contiguous()
                    context = HSA(q, mem_k, mem_v, weights, indices, self.sm_n, self.chunk_size, None, reg_lamda=self.reg_lamda, reg_C=self.reg_C)
                
                context = rearrange(context, 'N L h d->N L (h d)')
                out = self.o_proj(context)
                # o_out = out
                # print(f'inference mode: {out[:, :, :4]}')
                # print(f'has mlp? : {self._mlp is not None}')
                if self._mlp is not None and self._mlp_norm is not None:
                    h = self._mlp_norm(residual + out)
                    if self._reset_hsa_residual:
                        residual = residual + out
                    out = self._mlp(h)
                # print(f'out val: {out[0, -8:, :16]}, residual: {residual[0, -8:, :16]}')
                # print(f'inf hsa output: {out[0, :, :4]}')

                # print(f'hsa out norm: {o_out.norm(dim=-1).mean()} / residual norm: {residual.norm(dim=-1).mean()} / mlp out: {out.norm(dim=-1).mean()}')
                return residual + out, weights, mem_k, mem_v, landmarks, indices
                # return residual, weights, mem_k, mem_v, landmarks, indices
            else:
                # print(f'pass gca {cache_position[0]} ~ {cache_position[0] + q.shape[1]}')
                out = torch.zeros_like(hidden_states)
                if self._mlp is not None and self._mlp_norm is not None:
                    h = self._mlp_norm(residual + out)
                    if self._reset_hsa_residual:
                        residual = residual + out
                    out = self._mlp(h)
                return residual + out, weights, mem_k, mem_v, landmarks, indices

class RetrievalLayer(nn.Module):
    def __init__(
        self,
        config,
        group_idx=None,
        layer_idx=None,
        device=None,
        disable_sp=False,
        dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layer_idx = layer_idx
        self.retrieval_dim = config.retrieval_dim
        self.hidden_size = config.hidden_size
        assert self.retrieval_dim % 64 == 0
        self.num_heads = config.num_attention_heads
        self.kv_num_heads = config.num_kv_heads
        self.pre_norm = RMSNorm(self.hidden_size)
        self.ln = nn.Linear(config.hidden_size, self.retrieval_dim, bias=False)
        self.singlehead_retrieval = getattr(config, 'singlehead_retrieval', False)
        if not self.singlehead_retrieval:
            self.retrieval_heads_num = self.kv_num_heads
            self.retrieval_head_dim = self.retrieval_dim // self.kv_num_heads
        else:
            self.retrieval_heads_num = 1
            self.retrieval_head_dim = self.retrieval_dim
        self.chunk_topk = config.chunk_topk
        self.chunk_size = config.chunk_size
        self.group_idx = group_idx
        self.block_q = 1
        self.disable_sp = disable_sp
        self.enable_gumbel_noise = getattr(config, 'enable_gumbel_noise', False)
        self.enable_softmax = getattr(config, 'enable_softmax', False)
        self.mask_slide_window = getattr(config, 'mask_slide_window', False)
        self.slide_window = max(getattr(config, 'slide_window', 0), getattr(config, 'sliding_window', 0))
        self.enable_pos_decay = getattr(config, 'enable_pos_decay', False)
        self.enable_lmk_norm = getattr(config, 'enable_lmk_norm', False)
        if self.enable_lmk_norm:
            self.lmk_q_norm = RMSNorm(self.retrieval_head_dim)
            self.lmk_k_norm = RMSNorm(self.retrieval_head_dim)
        if self.enable_pos_decay:
            self.pos_decay = nn.Parameter(torch.zeros(1,))

        print(f"is using self.chunk_topk {self.chunk_topk}, enable gumbel noise: {self.enable_gumbel_noise} ")
        print(f'mask slide window: {self.mask_slide_window}, slide_window: {self.slide_window}')

        self.causal_masks = {}
        self.__init_weights(config)

    def __init_weights(self, config):
        std = config.initializer_range
        self.ln.weight.data.normal_(mean=0.0, std=std)

    def _indices_preprocess(self, indices, topk, q_indices, device, return_mask=False):
        k_indices = torch.arange(topk, device=device)
        # print(f'q_indices shape: {q_indices.shape}, indices.shape: {indices.shape}')
        visible = (q_indices // self.chunk_size) > k_indices  # (N, C, h, K), e.g. q at 0 cannot access any chunks, q at 64 can acess the 0th chunk
        # print(f'mask {mask}')
        # print(f'indices mask: {mask[0, ::self.chunk_size // 2, :]}')
        indices_ = indices.masked_fill(~visible, -1)
        indices_, _ = indices_.sort(dim=-1, descending=True)
        # print(f'indices_: {indices_[0, :128, :4]}')
        # print(f'after sort indices: {indices[0, ::self.chunk_size // 2, :]}')
        mask = indices_ == -1
        indices_.masked_fill_(indices_ == -1, 0)

        if not return_mask:
            return indices_
        else:
          return indices_, mask

    def _compute_pos_bias(self, q_emb, pos_emb, q_offset):
        pos_bias = torch.einsum('N L d, W d -> N L W')  # (N, L, w)

    def forward(
        self,
        hidden_states: torch.Tensor,
        weights:Optional[torch.Tensor]=None,
        mem_k:Optional[torch.Tensor]=None,
        mem_v:Optional[torch.Tensor]=None,
        landmarks:Optional[torch.Tensor]=None,
        indices:Optional[torch.Tensor]=None,
        cache_position: Optional[torch.Tensor]=None,
        position_embeddings: Optional[torch.Tensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        cache_params=None,
    ):
        # print(f'enter {self.layer_idx} layer, type: retrieval')
        if cache_params is None:
            if hidden_states.dtype != self.ln.weight.dtype:
                x_ = hidden_states.to(self.ln.weight.dtype)
            else:
                x_ = hidden_states
            # gather landmark representations

            # x_ : (N, L, d)
            q_emb = self.ln(self.pre_norm(x_[:, 0::self.block_q, :]))  # (N, L, d)
            # if self.pos_emb is not None:
            #     last_pos_emb = self.pos_emb([0])  # (1, dim)
            #     q_emb = q_emb + last_pos_emb.unsqueeze(0)
            q_emb = rearrange(q_emb, 'N C (h d)->N C h d', h=self.retrieval_heads_num)  # h: kv nums, split q emb into h_kv groups
            # naive implementation / softmax
            per_head_dim = q_emb.shape[-1]
            # landmakrs = rearrange(landmarks, 'N C h d-> N C (h d)')
            assert not torch.any(torch.isnan(q_emb))
            assert not torch.any(torch.isnan(landmarks))

            if self.enable_lmk_norm:
                q_emb = self.lmk_q_norm(q_emb)
                landmarks = self.lmk_k_norm(landmarks)
            scores = torch.einsum('N C h d, N D h d->N C h D', q_emb, landmarks) / math.sqrt(per_head_dim)

            N, C, _, D = scores.shape
            assert not torch.any(torch.isnan(scores))
            # h = self.kv_num_heads
            # scores = scores.unsqueeze(-2).repeat(1, 1, h, 1)
            
            chunk_top_k = min(scores.shape[-1], self.chunk_topk)

            offset = 0
            if get_parallel_state().sp_enabled and not self.disable_sp:
                offset = get_parallel_state().sp_rank * C

            c_indices = offset + torch.arange(C, device=hidden_states.device).view(1, C, 1, 1)  #  (1, C, 1, 1)
            d_indices = 1 + torch.arange(D, device=hidden_states.device).view(1, 1, 1, D)  #  (1, 1, 1, D)
            
            if self.mask_slide_window:
                mask = (c_indices * self.block_q - self.slide_window + self.chunk_size < d_indices * self.chunk_size)
            else:
                mask = (c_indices * self.block_q < d_indices * self.chunk_size)
            mask = mask.expand(N, -1, self.retrieval_heads_num, -1)  # (N, C, h, D)
            # print(f'local rank: {get_parallel_state().sp_rank} : {mask}')
            noise = 0
            if self.enable_gumbel_noise and self.training:
                noise = -torch.empty_like(
                    scores,
                    memory_format=torch.legacy_contiguous_format,
                    requires_grad=False).exponential_().log()
            scores.masked_fill_(mask, float('-inf'))  # # (N, C, h, D)
            # if self.enable_context_parallel:
            #     print(f'cp scores: {scores[0, :5, 0, :]}')
            # else:
            #     local_rank = torch.distributed.get_rank()
            #     world_size = torch.distributed.get_world_size()
            #     offset = C // world_size * local_rank
            #     print(f'non-cp scores: {scores[0, offset: offset + 5, 0, :]}')


            _, indices = torch.topk(scores + noise, dim=-1, k=chunk_top_k)  # (N, chunk, h, K)
            # scores_ = F.pad(scores.gather(dim=2, index=indices), (0, 1), value=0.0)
            indices, mask = self._indices_preprocess(
                indices, chunk_top_k, c_indices * self.block_q, hidden_states.device,
                return_mask=True
            )
            # print(f'mask: {mask}')
            scores_ = scores.gather(dim=-1, index=indices)
            scores_.masked_fill_(mask, float('-inf'))  # (N, C, h, K)
            # print(f'batch scores: {scores_[:, :, 0, :5]}')
            # if self.enable_context_parallel:
            #     print(f'indices: {indices[0, :5, 0, :]}')
            #     print(f'scores_: {scores_[0, :5, 0, :]}')
            assert not torch.any(torch.isnan(scores_))

            if not self.enable_softmax:
                # softplus_x = torch.where(scores_ < 15.0, torch.log1p(torch.exp(scores_)), scores_)
                scores_ = scores_.to(torch.float32)
                softplus_x = F.softplus(scores_, threshold=15.0)
                assert not torch.any(torch.isnan(softplus_x))
                softplus_x_cumsum = torch.cumsum(softplus_x, dim=-1)
                chunk_weights = (scores_ - softplus_x_cumsum).exp()
                # print(f'batch chunk weights: {chunk_weights}')

                assert torch.all(chunk_weights.sum(dim=-1) <= 1.01), f'{chunk_weights.sum(dim=-1)}'

                # if self.enable_context_parallel:
                #     print(f'weights: {chunk_weights[0, :5, 0, :]}')
                # else:
                #     local_rank = torch.distributed.get_rank()
                #     world_size = torch.distributed.get_world_size()
                #     offset = C // world_size * local_rank
                #     print(f'non-cp weights: {chunk_weights[0, offset: offset + 5, 0, :]}')

                # print(f'batch mode chunk_k v: {mem_k[0, 0, 0, 0, :5]}')
                # print(f'batch_mode chunk weights: {chunk_weights}, indices: {indices}')
                assert indices is not None
            else:
                if self.enable_pos_decay:
                    scores_ += torch.arange(scores_.shape[-1], device=scores_.device).unsqueeze(0).unsqueeze(1) * self.pos_decay
                all_neg_inf = (scores_ == float('-inf')).all(dim=-1, keepdim=True)
                scores_ = torch.where(scores_ == float('-inf'), -1e7, scores_)
                attn_probs = torch.softmax(scores_.to(torch.float32), dim=-1)
                chunk_weights = attn_probs.masked_fill(all_neg_inf, 0.0)
                # print(chunk_weights)
                # chunk_weights = softmax_off_by_one(scores_, dim=-1)  # N C h K
            if self.singlehead_retrieval:
                chunk_weights = chunk_weights.repeat(1, 1, self.kv_num_heads, 1)
                indices = indices.repeat(1, 1, self.kv_num_heads, 1)
            # print('retrieval a')
            # print(f'batch chunk_weights: {chunk_weights[0, :, 0, :4]}')
            # print(f'batch indices: {indices[0, :, 0, :4]}')
            return hidden_states, chunk_weights.contiguous(), mem_k, mem_v, landmarks, indices.contiguous()
        else:
            assert cache_position is not None

            kv_mgr = cache_params.mem_mgr
            device = hidden_states.device

            chunk_weights = None
            N = hidden_states.shape[0]
            L = hidden_states.shape[1]

            q_emb = self.ln(self.pre_norm(hidden_states))  # (N, chunk_num, dim)
            q_emb = rearrange(q_emb, 'N C (h d)->N C h d', h=self.retrieval_heads_num)
            landmarks = kv_mgr.past_lmk_embeds
            padding_lens = kv_mgr.padding_lens
            # print(f'landmarks shape: {landmarks.shape}')
            if landmarks is not None:
                per_head_dim = q_emb.shape[-1]
                # print(f'q_emb type: {q_emb.dtype}, lmk dtype: {landmarks.dtype}')
                if self.enable_lmk_norm:
                    q_emb = self.lmk_q_norm(q_emb)
                    landmarks = self.lmk_k_norm(landmarks)
                scores = torch.einsum('N C h d, N D h d->N C h D', q_emb, landmarks) / math.sqrt(per_head_dim)
                D = landmarks.shape[1]
                q_pos = cache_position[0] + torch.arange(0, L, device=device).unsqueeze(0)  # (1, L)
                q_pos = q_pos.view(1, L, 1, 1)
                if padding_lens is not None:
                    q_pos = q_pos - rearrange(padding_lens, 'N->N 1 1 1')
                chunk_end = (1 + torch.arange(0, D, device=device)) * self.chunk_size  # (D)
                chunk_end = chunk_end.view(1, 1, 1, D) # (N, L, h, D)
                
                if self.mask_slide_window:
                    mask = (q_pos - self.slide_window + self.chunk_size < chunk_end)
                else:
                    mask = q_pos < chunk_end  # (1, L, h, D)

                mask = mask.expand(N, -1, self.retrieval_heads_num, -1)
                scores.masked_fill_(mask, float('-inf'))
                # print(f'org scores: {scores}')
                
                chunk_top_k = min(scores.shape[-1], self.chunk_topk)
                _, indices = torch.topk(scores, dim=-1, k=chunk_top_k)  # (N, chunk_num, h, K)

                indices, mask = self._indices_preprocess(
                    indices, chunk_top_k, q_pos, device, return_mask=True)

                scores_ = scores.gather(dim=-1, index=indices)
                scores_.masked_fill_(mask, float('-inf'))
                # print(f'cache_position: {cache_position[0]} decoding scores: {scores_[:, :, 0, :5]}')
                # softplus_x = torch.where(scores_ <= 15, torch.log(1 + torch.exp(scores_)), scores_)
                if not self.enable_softmax:
                    scores_ = scores_.to(torch.float32)
                    softplus_x = F.softplus(scores_, threshold=15.0)
                    softplus_x_cumsum = torch.cumsum(softplus_x, dim=-1)
                    chunk_weights = (scores_ - softplus_x_cumsum).exp()
                else:
                    if self.enable_pos_decay:
                        scores_ += torch.arange(scores_.shape[-1], device=scores_.device).unsqueeze(0).unsqueeze(1) * self.pos_decay
                    all_neg_inf = (scores_ == float('-inf')).all(dim=-1, keepdim=True)
                    scores_ = torch.where(scores_ == float('-inf'), -1e7, scores_)
                    attn_probs = torch.softmax(scores_.to(torch.float32), dim=-1)
                    chunk_weights = attn_probs.masked_fill(all_neg_inf, 0.0)
                if self.singlehead_retrieval:
                    chunk_weights = chunk_weights.repeat(1, 1, self.kv_num_heads, 1)
                    indices = indices.repeat(1, 1, self.kv_num_heads, 1)
                # print(f'inference chunk weights: {scores_[0, -32:, 0, :]}')
                # print(f'inf_mode chunk weights: {chunk_weights[0, -32:, 0, :]} top indices: {indices[0, -32:, 0, :]}, weights sum: {chunk_weights[0, -32:, 0, :].sum(dim=-1)}')
                # print(f'inf chunk_weights: {chunk_weights[0, :, 0, :4]}')
                # print(f'inf indices: {indices[0, :, 0, :4]}')

            # print('retrieval b')
            # print(f'weights is None ? {chunk_weights is None}, indices is None ? {indices is None}')
            return hidden_states, chunk_weights, mem_k, mem_v, landmarks, indices


class ChunkingLayer(nn.Module):
    def __init__(
        self,
        config, 
        layer_idx=None, 
        device=None, 
        encoder_cls=None,
        disable_sp=False,
        dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.has_encoder = config.encoder_layers > 0
        if self.has_encoder and encoder_cls is not None:
            self.encoder = encoder_cls(config)
            self.enable_enc_prenorm = getattr(config, 'enc_prenorm', False)
            if self.enable_enc_prenorm:
                self.enc_prenorm = RMSNorm(config.hidden_size)
                print('enable enc_prenorm')
        else:
            if self.has_encoder:
                warnings.warn('encoder_cls is None')
            self.has_encoder = False
            self.norm = RMSNorm(config.hidden_size)
        ratio = config.num_attention_heads // config.num_kv_heads
        self.num_kv_heads = config.num_kv_heads
        assert ratio >= 1
        self.d_model = config.hidden_size
        self.kv_dim = config.hidden_size // ratio
        self.proj_k = nn.Linear(config.hidden_size, config.hidden_size // ratio, bias=False, **factory_kwargs)
        self.enable_qk_norm = getattr(config, 'enable_qk_norm', False)
        self.proj_v = nn.Linear(config.hidden_size, config.hidden_size // ratio, bias=False, **factory_kwargs)
        if self.enable_qk_norm:
            self.k_rmsnorm = RMSNorm(config.hidden_size // (ratio * self.num_kv_heads))
        self.lmk_ln = nn.Linear(config.hidden_size, config.retrieval_dim, bias=False)

        self.singlehead_retrieval = getattr(config, 'singlehead_retrieval', False)
        if self.singlehead_retrieval:
            self.retrieval_heads_num = 1
        else:
            self.retrieval_heads_num = self.num_kv_heads

        # self.pre_norm = RMSNorm(config.hidden_size, **factory_kwargs)
        self.enable_enc_cls = getattr(config, 'enable_enc_cls', self.has_encoder)
        if self.enable_enc_cls:
            assert self.has_encoder
            self.cls_emb = nn.Parameter(torch.randn(1, 1, self.config.hidden_size))
        self.chunk_size = config.chunk_size
        self.disable_sp = disable_sp
        assert self.chunk_size % 64 == 0
        self.__init_weights(config)

    def __init_weights(self, config):
        std = config.initializer_range
        self.proj_k.weight.data.normal_(mean=0.0, std=std)
        self.proj_v.weight.data.normal_(mean=0.0, std=std)
        self.lmk_ln.weight.data.normal_(mean=0.0, std=std)
        if self.enable_enc_cls:
            self.cls_emb.data.normal_(mean=0.0, std=std)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.proj_k.weight.device
        hidden_states = torch.zeros((batch_size, self.chunk_size, self.d_model), device=device, dtype=dtype)
        return hidden_states, torch.zeros((batch_size,), device=device, dtype=torch.long)

    def _encode(self, x_):
        N = x_.shape[0]
        x_ = rearrange(x_, 'N (C S) d->(N C) S d', S=self.chunk_size)
        if self.enable_enc_cls:
            NC, S, d = x_.shape
            cls_emb = repeat(self.cls_emb, '1 1 d -> NC 1 d', NC=NC)
            x_ = torch.cat([x_, cls_emb], dim=1)

        if self.has_encoder:
            if self.enable_enc_prenorm:
                x_ = self.enc_prenorm(x_)
            enc_outputs = self.encoder(inputs_embeds=x_)
            enc_hiddens = rearrange(enc_outputs.last_hidden_state, '(N C) S d->N C S d', N=N)
        else:
            enc_hiddens = self.norm(x_)
            enc_hiddens = rearrange(enc_hiddens, '(N C) S d->N C S d', N=N)
        
        if enc_hiddens.dtype != self.lmk_ln.weight.dtype:
            enc_hiddens = enc_hiddens.to(self.lmk_ln.weight)
        if self.enable_enc_cls:
            landmarks = self.lmk_ln(enc_hiddens[:, :, -1, :])
            enc_hiddens = enc_hiddens[:, :, :-1, :]
        else:
            landmarks = self.lmk_ln(enc_hiddens.mean(dim=-2))  # (N, chunk_num, dim)
        landmarks = rearrange(landmarks, 'N L (h d)->N L h d', h=self.retrieval_heads_num)
        return enc_hiddens, landmarks

    def forward(
        self, 
        hidden_states,  # normed hidden_states
        weights:Optional[torch.Tensor]=None,
        mem_k:Optional[torch.Tensor]=None,
        mem_v:Optional[torch.Tensor]=None,
        landmarks:Optional[torch.Tensor]=None,
        indices:Optional[torch.Tensor]=None,
        cache_position:Optional[torch.LongTensor]=None,
        position_embeddings: Optional[torch.Tensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        cache_params=None,
    ):
        # x: (N, L, dim)
        assert weights is None
        assert mem_k is None
        assert mem_v is None
        assert landmarks is None
        # print(f'enter {self.layer_idx} layer, type: chunking')
        if cache_params is None:
            if hidden_states.dtype != self.proj_k.weight.dtype:
                x_ = hidden_states.to(self.proj_k.weight.dtype)
            else:
                x_ = hidden_states
            
            # ob = rearrange(x_, 'N (C S) d->N C S d', S=self.chunk_size)
            # print(f'batched hidden repr: {ob[:, :, :4, :8]}')
            assert not torch.any(torch.isnan(x_))
            enc_hiddens, landmarks = self._encode(x_)
            assert not torch.any(torch.isnan(landmarks))
            # print(f'batch lmks: {landmarks[:, :, 0, :4]}')
            
            mem_k = self.proj_k(enc_hiddens)  # (N, C, S, dim)
            mem_v = self.proj_v(enc_hiddens)

            # print(f'batch hiddens: {x_[:, :4, :4]}')
            
            mem_k = rearrange(mem_k, 'N C S (h d)->N (C S) h d', h=self.num_kv_heads)
            if self.enable_qk_norm:
                mem_k = self.k_rmsnorm(mem_k)
            mem_v = rearrange(mem_v, 'N C S (h d)->N (C S) h d', h=self.num_kv_heads)
            # print(f'batch kv: landmark: {landmarks[0, :, 0, :5]}, kv: {mem_k[0, ::16, 0, ::16]}')

            if get_parallel_state().sp_enabled and not self.disable_sp:
                sp_group = get_parallel_state().ulysses_group
                # print(f'sp rank: {dist.get_rank(sp_group)}, world size: {dist.get_world_size(sp_group)} local rank: {dist.get_rank()}')
                mem_k = AllGatherWithGradient.apply(mem_k, sp_group)
                mem_v = AllGatherWithGradient.apply(mem_v, sp_group)
                landmarks = AllGatherWithGradient.apply(landmarks, sp_group)

            # print(f'encoder a')
            # print(f'batch hidden_states repr in Chunking Layer: {hidden_states[0, -4:, :4]}')
            return hidden_states, weights, mem_k.contiguous(), mem_v.contiguous(), landmarks.contiguous(), None
        else:
            # if hidden_states.shape[1] > 1:
            #     print(f'inf hidden_states repr in Chunking Layer: {hidden_states[0, -4:, :4]}')
            # elif hidden_states.shape[1] == 1:
            #     print(f'inf hidden_states repr in Chunking Layer: {hidden_states[0, :, :4]}')
            # assert cache_params.replay_mask is None  # no replay for lower layers and chunk encoder
            kv_mgr = cache_params.mem_mgr
            # assert self.layer_idx in cache_params.key_value_memory_dict
            if self.layer_idx not in cache_params.key_value_memory_dict:
                # print(f'initialize inference cache')
                cache_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(
                    hidden_states.shape[0],
                    0,
                    dtype=self.proj_k.weight.dtype
                )

        
            current_hidden_cache, cache_lens = cache_params.key_value_memory_dict[self.layer_idx]
            # print(f'cache lens: {cache_lens.shape}, {cache_lens.dtype}')

            # cache_len = cache_position[0] % self.chunk_size  # length of current cache
            # print(f'cache position: {cache_position.shape}')
            # cache_lens = cache_position % self.chunk_size  # (N)

            # print(f'cache lens: {cache_lens.shape}, dtype: {cache_lens.dtype}')
            # L = hidden_states.shape[1]
            max_L = hidden_states.shape[1]
            N = hidden_states.shape[0]
            # print(attention_mask.shape)
            if attention_mask is not None:
                L = attention_mask.sum(dim=-1)  # (N), assume 1 for no masking and 0 for masking
                if not torch.all(attention_mask):
                    # print(f'max_L: {max_L}, L:{L}')
                    padding_lens = max_L - L
                    if kv_mgr.padding_lens is None:
                        kv_mgr.padding_lens = padding_lens
                    else:
                        kv_mgr.padding_lens += padding_lens
            else:
                # L = [hidden_states.shape[1]] * N
                L = torch.zeros(N, device=hidden_states.device, dtype=torch.long)
                L.fill_(hidden_states.shape[1])
                # L = hidden_states.shape[1]
            # if hidden_states.shape[1] + cache_len >= self.chunk_size:
            # print(f'L : {L}')
            concat_lens = L + cache_lens
            chunk_filled = concat_lens >= self.chunk_size
            # print(f'non zero: {chunk_filled.nonzero()}')
            b_ids = chunk_filled.nonzero().squeeze(1)
            truncate_lens = concat_lens % self.chunk_size
            # cache_lens_cpu = cache_lens.to(device='cpu', non_blocking=True)
            # print(chunk_filled)
            if torch.any(chunk_filled):
                # encode new chunks
                # print(f'b_ids shape: {b_ids.shape}')
                
                chunk_h_list = []
                chunk_nums = []
                for b_id in b_ids:
                    truncate_len = truncate_lens[b_id]
                    offset = max_L - L[b_id]
                    # print(f'bid: {b_id}, cache len: {cache_lens[b_id]}, trunc len: {truncate_len}, offset: {offset}')
                    if truncate_len > 0:
                        # print(f'cache shape: {current_hidden_cache[b_id, self.chunk_size - cache_lens[b_id]:].shape} ' + \
                        #     f'hidden shape: {hidden_states[b_id, offset:-truncate_len].shape}')
                        chunk_hidden_states = torch.cat(
                            (current_hidden_cache[b_id, self.chunk_size - cache_lens[b_id]:], hidden_states[b_id, offset:-truncate_len]),
                            dim=0
                        )
                    else:
                        # print(f'cache shape: {current_hidden_cache[b_id, self.chunk_size - cache_lens[b_id]:].shape} ' + \
                        #     f'hidden shape: {hidden_states[offset:].shape}')
                        chunk_hidden_states = torch.cat(
                            (current_hidden_cache[b_id, self.chunk_size - cache_lens[b_id]:], hidden_states[b_id, offset:]),
                            dim=0
                        )
                    # (64 * ?, dim)
                    chunk_hidden_states = rearrange(chunk_hidden_states, '(N S) d->N S d', S=self.chunk_size)
                    # print(f'bid: {b_id}, pos: {cache_position[0]} chunked repr: {chunk_hidden_states[:, :4, :8]}')
                    chunk_nums.append(chunk_hidden_states.shape[0])
                    chunk_h_list.append(chunk_hidden_states)
                
                
                chunked_hidden_states = torch.concat(chunk_h_list, dim=0)  # (?, chunk_size, dim)
                # print(f'chunked hidden_states shape: {chunked_hidden_states.shape}')

                # sample inputs: x x | x x x x | x x x 
                # sample inputs | x x x x | x x x x | x x 
                # truncate_len = (L + cache_len) % self.chunk_size

                # if truncate_len > 0:
                #     chunk_hidden_states = torch.cat(
                #         (current_hidden_cache[:, self.chunk_size - cache_len:], hidden_states[:, :-truncate_len]),
                #         dim=1
                #     )
                # else:
                #     chunk_hidden_states = torch.cat(
                #         (current_hidden_cache[:, self.chunk_size - cache_len:], hidden_states),
                #         dim=1
                #     )
                # print(f'chunk_hidden_states after cat: {chunk_hidden_states.shape}')
                assert chunked_hidden_states.shape[1] == self.chunk_size

                # print(f'before encode: {chunked_hidden_states.shape}')
                enc_hiddens, landmarks = self._encode(chunked_hidden_states)
                # print(f'landmarks: {landmarks.shape}')
                # print(f'after encode: {enc_hiddens.shape}')
                # landmarks: (N, 1, hq, rd)
                # enc_hiddens: (N, 1, S, h_kv * d)
                assert enc_hiddens.shape[1] == 1

                mem_k = self.proj_k(enc_hiddens).squeeze(1)
                if self.enable_qk_norm:
                    mem_k = rearrange(mem_k, 'N S (h d)->N S h d', h=self.num_kv_heads)
                    mem_k = self.k_rmsnorm(mem_k)
                    mem_k = rearrange(mem_k, 'N S h d->N S (h d)')
                mem_v = self.proj_v(enc_hiddens).squeeze(1)
                # landmarks = rearrange(landmarks, 'N C h d->N C (h d)')
                # kv_mgr.append(mem_k, mem_v, landmarks)
                # print(f'b ids cpu: {b_ids_cpu}')
                # print(f'mem_k shape: {mem_k.shape}')
                # print(f'append landmark dtype: {landmarks.dtype}')
                kv_mgr.append_varlen(b_ids, chunk_nums, mem_k, mem_v, landmarks.squeeze(1))
                # print(f'new mem_k: {mem_k[0, ::16, ::16]}')
                # print(f'inf new lmk_emb: {landmarks[:, :, 0, :5]}')
                # mem_k = rearrange(mem_k, 'N C S (h d)->N (C S) h d', h=self.num_kv_heads)
                # mem_v = rearrange(mem_v, 'N C S (h d)->N (C S) h d', h=self.num_kv_heads)
            
            # update hidden cache
            cache_lens = truncate_lens
            assert torch.all(cache_lens < self.chunk_size)
            current_hidden_cache.copy_(torch.roll(current_hidden_cache, shifts=-hidden_states.shape[1], dims=-2))
            # print(f'cache_position[0]: {cache_position[0]}, cache_lens: {cache_lens}')
            
            # print(f'current_hidden_cache: {current_hidden_cache.shape}, hidden states: {hidden_states.shape}')
            # if hidden_states.shape[1] > 1 or cache_position[0] >= 128 and cache_position[0] <= 135:
            #     print(f'hidden shape: {hidden_states.shape}')
            #     print(f'hidden states: {hidden_states[0, :, :4]}')
            current_hidden_cache[:, -hidden_states.shape[1]:] = hidden_states[:, -self.chunk_size:]

            cache_params.key_value_memory_dict[self.layer_idx] = (current_hidden_cache, cache_lens)

            # print(f'encoder b')
            if kv_mgr.chunk_k is not None and kv_mgr.chunk_v is not None and not kv_mgr._offloading:
                # print(f'reshape kv cache')
                # print(f'reshape mem_k: {kv_mgr.chunk_k[:, :, :4, :4]} shape: {kv_mgr.chunk_k.shape}')
                # print(f'lmks: {kv_mgr.lmk_embs[:, :, 0, :4]}')
                mem_k = rearrange(kv_mgr.chunk_k, 'N C S (h d)->N (C S) h d', h=self.num_kv_heads)
                mem_v = rearrange(kv_mgr.chunk_v, 'N C S (h d)->N (C S) h d', h=self.num_kv_heads)
            return hidden_states, None, mem_k, mem_v, None, None
