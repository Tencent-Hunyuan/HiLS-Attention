import os
import sys
import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.FlashHiLS.configuration_hils import HiLSConfig
from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM

try:
    from transformers.utils import is_flash_attn_2_available
except Exception:  # pragma: no cover
    def is_flash_attn_2_available():
        return False


def _kl_div(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """Compute mean KL(softmax(p) || softmax(q)) over all positions."""
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    kl = F.kl_div(log_q, log_p, log_target=True, reduction="batchmean")
    return kl.item()


CHUNK_SIZE = 64  # must match config.chunk_size
KL_THRESHOLD = 1e-4  # softmax KL divergence threshold for bf16 equivalence


def _build_model(device: str, dtype: torch.dtype, sliding_window: int) -> HiLSForCausalLM:
    config = HiLSConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_swa_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        use_sliding_window=True,
        sliding_window=sliding_window,
        full_attn_interleave=0,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        chunk_size=CHUNK_SIZE,
    )
    config._attn_implementation = "flash_attention_2"
    model = HiLSForCausalLM(config).to(device=device, dtype=dtype)
    model.auto_insert_lmk = True
    model.eval()
    return model

def _build_hybrid_model(device: str, dtype: torch.dtype, sliding_window: int) -> HiLSForCausalLM:
    config = HiLSConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        num_swa_layers=0,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        use_sliding_window=True,
        sliding_window=sliding_window,
        full_attn_interleave=1, # 1 swa, 1 hsa
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        chunk_size=CHUNK_SIZE,
        hils_topk=16,
    )
    config._attn_implementation = "flash_attention_2"
    model = HiLSForCausalLM(config).to(device=device, dtype=dtype)
    model.auto_insert_lmk = True
    model.eval()
    return model


def _run(model: HiLSForCausalLM, input_ids: torch.Tensor, position_ids: torch.Tensor = None, cu_seq_lens_q=None):
    """Run forward; model internally inserts lmk via auto_insert_lmk."""
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            cu_seq_lens_q=cu_seq_lens_q
        )
    return out.logits


def _skip_if_unavailable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for flash-attn varlen tests")
    if not is_flash_attn_2_available():
        pytest.skip("flash-attn-2 is not available")


def _pad_to_chunk_aligned(ids: torch.Tensor, pad_id: int, chunk_size: int):
    """
    Pad input_ids so that after lmk insertion the total length is a
    multiple of chunk_size.

    insert_special_tokens inserts 1 lmk every (chunk_size - 1) tokens.
    For L original tokens the output length is:
        full_chunks * chunk_size + remainder
    where full_chunks = L // (chunk_size - 1), remainder = L % (chunk_size - 1).
    To make remainder == 0, L must be a multiple of (chunk_size - 1).
    We pad the original input_ids to the next multiple of (chunk_size - 1).
    """
    L = ids.shape[1]
    seg = chunk_size - 1
    remainder = L % seg
    if remainder == 0:
        return ids
    pad_len = seg - remainder
    pad_tokens = torch.full((ids.shape[0], pad_len), pad_id, device=ids.device, dtype=ids.dtype)
    return torch.cat([ids, pad_tokens], dim=1)


# --------------------------------------------------------------------------- #
# Test 1: unaligned naive concat → packed varlen should DIFFER from single fwd
# --------------------------------------------------------------------------- #
def test_unaligned_packed_differs_from_unpacked():
    """
    Naively concatenating two samples without padding to chunk boundary
    causes the model-internal lmk insertion to misalign. The packed
    result should therefore differ from running each sample independently.
    """
    _skip_if_unavailable()

    device = "cuda"
    dtype = torch.bfloat16
    model = _build_model(device, dtype, sliding_window=512)

    torch.manual_seed(42)
    l1, l2, l3 = 1254, 2041, 3102  # NOT multiples of (chunk_size - 1) = 63
    seq1 = torch.randint(0, model.config.vocab_size, (1, l1), device=device)
    seq2 = torch.randint(0, model.config.vocab_size, (1, l2), device=device)
    seq3 = torch.randint(0, model.config.vocab_size, (1, l3), device=device)

    # --- single-sample fwd (correct baseline) ---
    out_1 = _run(model, seq1)
    out_2 = _run(model, seq2)
    out_single = torch.cat([out_1, out_2], dim=1)

    # --- naive packed fwd (no padding → lmk boundaries misalign) ---
    packed_raw = torch.cat([seq1, seq2], dim=1)
    out_packed = _run(model, packed_raw)

    # Outputs have different lengths (lmk count may differ), compare prefix
    min_len = min(out_single.shape[1], out_packed.shape[1])
    wrong_kl2 = _kl_div(out_packed[:, l1:min_len, :], out_single[:, l1:min_len, :])
    assert wrong_kl2 > KL_THRESHOLD, f"wrong seg2 KL {wrong_kl2} should exceed {KL_THRESHOLD}"

    model.auto_insert_lmk = False
    out_1 = _run(model, seq1)
    out_2 = _run(model, seq2)
    out_3 = _run(model, seq3)
    out_single = torch.cat([out_1, out_2, out_3], dim=1)
    # --- packed fwd with varlen position_ids (no lmk) ---
    packed_raw = torch.cat([seq1, seq2, seq3], dim=1)
    pos1 = torch.arange(l1, device=device)
    pos2 = torch.arange(l2, device=device)
    pos3 = torch.arange(l3, device=device)
    packed_pos = torch.cat([pos1, pos2, pos3], dim=0).unsqueeze(0)
    out_packed = _run(model, packed_raw, packed_pos, cu_seq_lens_q=True)
    kl_seg1 = _kl_div(out_packed[:, :l1, :], out_single[:, :l1, :])
    kl_seg2 = _kl_div(out_packed[:, l1:l1+l2, :], out_single[:, l1:l1+l2, :])
    kl_seg3 = _kl_div(out_packed[:, l1 + l2: l1 + l2 + l3, :], out_single[:, l1 + l2: l1 + l2 + l3, :])
    assert kl_seg1 < KL_THRESHOLD, f"seg1 KL divergence {kl_seg1} exceeds {KL_THRESHOLD}"
    assert kl_seg2 < KL_THRESHOLD, f"seg2 KL divergence {kl_seg2} exceeds {KL_THRESHOLD}"
    assert kl_seg3 < KL_THRESHOLD, f"seg2 KL divergence {kl_seg3} exceeds {KL_THRESHOLD}"



# --------------------------------------------------------------------------- #
# Test 2: aligned concat (pad to chunk boundary) → packed varlen MATCHES
# --------------------------------------------------------------------------- #
def test_aligned_packed_equals_unpacked():
    """
    Pad each sub-sample to (chunk_size - 1) boundary so that after lmk
    insertion each segment is chunk_size-aligned. Then concat with
    varlen position_ids (each segment resets from 0). The result on
    non-pad positions should match single-sample fwd.
    """
    _skip_if_unavailable()

    device = "cuda"
    dtype = torch.bfloat16
    model = _build_model(device, dtype, sliding_window=512)

    torch.manual_seed(42)
    l1, l2 = 100, 80
    seq1 = torch.randint(0, model.config.vocab_size, (1, l1), device=device)
    seq2 = torch.randint(0, model.config.vocab_size, (1, l2), device=device)

    # --- single-sample fwd (original unpadded inputs) ---
    out_1 = _run(model, seq1)
    out_2 = _run(model, seq2)

    # --- aligned packed fwd ---
    # pad each segment so that after lmk insertion length is chunk_size multiple
    pad_id = model.config.vocab_size  # lmk_id
    seq1_padded = _pad_to_chunk_aligned(seq1, pad_id, CHUNK_SIZE)
    seq2_padded = _pad_to_chunk_aligned(seq2, pad_id, CHUNK_SIZE)

    packed_ids = torch.cat([seq1_padded, seq2_padded], dim=1)
    # varlen position_ids: each segment resets from 0
    pos1 = torch.arange(seq1_padded.shape[1], device=device)
    pos2 = torch.arange(seq2_padded.shape[1], device=device)
    packed_pos = torch.cat([pos1, pos2], dim=0).unsqueeze(0)

    out_packed = _run(model, packed_ids, packed_pos, cu_seq_lens_q=True)

    # After lmk insertion + non_lmk_mask removal, each padded segment
    # contributes padded_len tokens (lmk tokens are stripped by mask).
    # seg1_padded_len = seq1_padded.shape[1], seg2 starts right after.
    seg1_out_len = seq1_padded.shape[1]  # 126 (lmk removed, only real+pad tokens remain)
    orig_out_len_1 = out_1.shape[1]
    orig_out_len_2 = out_2.shape[1]

    out_seg1 = out_packed[:, :orig_out_len_1, :]
    out_seg2 = out_packed[:, seg1_out_len:seg1_out_len + orig_out_len_2, :]

    kl1 = _kl_div(out_seg1, out_1)
    kl2 = _kl_div(out_seg2, out_2)
    assert kl1 < KL_THRESHOLD, f"seg1 KL divergence {kl1} exceeds {KL_THRESHOLD}"
    assert kl2 < KL_THRESHOLD, f"seg2 KL divergence {kl2} exceeds {KL_THRESHOLD}"


# --------------------------------------------------------------------------- #
# Test 2: aligned concat (pad to chunk boundary) → packed varlen MATCHES
# --------------------------------------------------------------------------- #
def test_hsa_aligned_packed_equals_unpacked():
    """
    Pad each sub-sample to (chunk_size - 1) boundary so that after lmk
    insertion each segment is chunk_size-aligned. Then concat with
    varlen position_ids (each segment resets from 0). The result on
    non-pad positions should match single-sample fwd.
    """
    _skip_if_unavailable()

    device = "cuda"
    dtype = torch.bfloat16
    model = _build_hybrid_model(device, dtype, sliding_window=512)

    torch.manual_seed(63)
    l1, l2, l3 = 1256, 1232, 1231
    seq1 = torch.randint(0, model.config.vocab_size, (1, l1), device=device)
    seq2 = torch.randint(0, model.config.vocab_size, (1, l2), device=device)
    seq3 = torch.randint(0, model.config.vocab_size, (1, l3), device=device)

    # --- single-sample fwd (original unpadded inputs) ---
    out_1 = _run(model, seq1)
    out_2 = _run(model, seq2)
    out_3 = _run(model, seq3)

    # --- aligned packed fwd ---
    # pad each segment so that after lmk insertion length is chunk_size multiple
    pad_id = model.config.vocab_size  # lmk_id
    seq1_padded = _pad_to_chunk_aligned(seq1, pad_id, CHUNK_SIZE)
    seq2_padded = _pad_to_chunk_aligned(seq2, pad_id, CHUNK_SIZE)
    seq3_padded = _pad_to_chunk_aligned(seq3, pad_id, CHUNK_SIZE)
    print(f'seq1_padded.shape: {seq1_padded.shape}, seq2_padded.shape: {seq2_padded.shape}, seq3_padded.shape: {seq3_padded.shape}')

    packed_ids = torch.cat([seq1_padded, seq2_padded, seq3_padded], dim=1)
    # varlen position_ids: each segment resets from 0
    pos1 = torch.arange(seq1_padded.shape[1], device=device)
    pos2 = torch.arange(seq2_padded.shape[1], device=device)
    pos3 = torch.arange(seq3_padded.shape[1], device=device)
    packed_pos = torch.cat([pos1, pos2, pos3], dim=0).unsqueeze(0)

    out_packed = _run(model, packed_ids, packed_pos, cu_seq_lens_q=True)
    wrong_out_packed = _run(model, packed_ids)
    # After lmk insertion + non_lmk_mask removal, each padded segment
    # contributes padded_len tokens (lmk tokens are stripped by mask).
    # seg1_padded_len = seq1_padded.shape[1], seg2 starts right after.
    seg1_out_len = seq1_padded.shape[1]  # 126 (lmk removed, only real+pad tokens remain)
    seg2_out_len = seq1_padded.shape[1] + seq2_padded.shape[1] # 126 (lmk removed, only real+pad tokens remain)
    orig_out_len_1 = out_1.shape[1]
    orig_out_len_2 = out_2.shape[1]
    orig_out_len_3 = out_3.shape[1]

    out_seg1 = out_packed[:, :orig_out_len_1, :]
    out_seg2 = out_packed[:, seg1_out_len:seg1_out_len + orig_out_len_2, :]
    out_seg3 = out_packed[:, seg2_out_len:seg2_out_len + orig_out_len_3, :]

    kl1 = _kl_div(out_seg1, out_1)
    kl2 = _kl_div(out_seg2, out_2)
    kl3 = _kl_div(out_seg3, out_3)
    assert kl1 < KL_THRESHOLD, f"seg1 KL divergence {kl1} exceeds {KL_THRESHOLD}"
    assert kl2 < KL_THRESHOLD, f"seg2 KL divergence {kl2} exceeds {KL_THRESHOLD}"
    assert kl3 < KL_THRESHOLD, f"seg3 KL divergence {kl3} exceeds {KL_THRESHOLD}"

    # --- wrong_out_packed (no varlen position_ids) should DIFFER from each segment ---
    wrong_seg2 = wrong_out_packed[:, seg1_out_len:seg1_out_len + orig_out_len_2, :]
    wrong_seg3 = wrong_out_packed[:, seg2_out_len:seg2_out_len + orig_out_len_3, :]

    wrong_kl2 = _kl_div(wrong_seg2, out_2)
    wrong_kl3 = _kl_div(wrong_seg3, out_3)
    assert wrong_kl2 > KL_THRESHOLD, f"wrong seg2 KL {wrong_kl2} should exceed {KL_THRESHOLD}"
    assert wrong_kl3 > KL_THRESHOLD, f"wrong seg3 KL {wrong_kl3} should exceed {KL_THRESHOLD}"

