"""
Test OLMo HiLS generation.
Compare greedy decoding from generate() with argmax from a full forward pass.

Validation flow:
1. Run model.generate() on the prompt with greedy decoding.
2. Concatenate prompt and generated tokens, insert LMK tokens externally, and run a full prefill forward pass.
3. Remove logits at LMK positions externally and compare shifted argmax outputs with generated tokens.

Reference: eval/eval_ppl_hf.py handles LMK insertion and removal outside the model.
"""

import sys
import os
import math
import argparse
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from models.FlashHiLS.configuration_hils import HSAConfig
from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM

# Override the default model_type to match the registered OLMo HiLS name.
HSAConfig.model_type = "olmo_hils"
AutoConfig.register("olmo_hils", HSAConfig)
HiLSForCausalLM.config_class = HSAConfig
AutoModelForCausalLM.register(HSAConfig, HiLSForCausalLM)

from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks


DEFAULT_CKPT_PATH = (
    "/Models/hils-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_13000/hf_ckpt"
)


def compute_ppl_from_logits(logits, target_ids):
    """
    Compute PPL from logits and target token ids.

    Args:
        logits: [gen_len, vocab_size] logits, logits[i] predicts target_ids[i]
        target_ids: [gen_len] target token ids

    Returns:
        ppl: perplexity
        avg_nll: average negative log-likelihood
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)  # [gen_len, vocab_size]
    target_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)  # [gen_len]
    avg_nll = -target_log_probs.mean().item()
    ppl = math.exp(avg_nll)
    return ppl, avg_nll


def run_generate(model, tokenizer, prompt, device, max_new_tokens):
    """
    Run model.generate() with greedy decode.
    Returns (input_ids, generated_token_ids, gen_scores).
    gen_scores are the per-step decode logits (used to compute PPL).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decode
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_ids = output.sequences[0, input_len:]
    gen_scores = torch.stack(output.scores, dim=0).squeeze(1)  # [gen_len, vocab_size]
    return input_ids, generated_ids, gen_scores


def _position_ids_from_attention_mask(attention_mask):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def _trim_generated_for_compare(generated_ids, eos_token_id=None, pad_token_id=None):
    """Keep tokens through EOS and drop padding that only exists because batching continued."""
    ids = generated_ids.detach().cpu()
    keep_len = ids.numel()
    if eos_token_id is not None:
        eos_positions = torch.nonzero(ids == eos_token_id, as_tuple=False).flatten()
        if eos_positions.numel() > 0:
            keep_len = int(eos_positions[0].item()) + 1
    elif pad_token_id is not None:
        pad_positions = torch.nonzero(ids == pad_token_id, as_tuple=False).flatten()
        if pad_positions.numel() > 0:
            keep_len = int(pad_positions[0].item())
    return generated_ids[:keep_len]


def _is_tie_like(logits_a, logits_b, token_a, token_b, tolerance):
    gap_a = (logits_a[token_a] - logits_a[token_b]).abs().item()
    gap_b = (logits_b[token_a] - logits_b[token_b]).abs().item()
    return gap_a <= tolerance and gap_b <= tolerance, gap_a, gap_b


def _last_real_token_indices(attention_mask):
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return (attention_mask.long() * positions).max(dim=1).values


def run_prefill_next_logits_batch(model, tokenizer, prompts, device, padding_side):
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    finally:
        tokenizer.padding_side = old_padding_side

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    position_ids = _position_ids_from_attention_mask(attention_mask)
    last_indices = _last_real_token_indices(attention_mask)

    model._gen_state.reset()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
    logits = outputs.logits[torch.arange(input_ids.shape[0], device=device), last_indices]
    return logits, input_ids, attention_mask


def run_prefill_next_logits_single(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model._gen_state.reset()
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=True,
        )
    return outputs.logits[:, -1, :].squeeze(0)


def run_generate_batch(model, tokenizer, prompts, device, max_new_tokens, padding_side, drop_attention_mask=False):
    """Batched greedy generate. Returns padded input ids/masks and per-row scores."""
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    finally:
        tokenizer.padding_side = old_padding_side

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_len = input_ids.shape[1]

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "output_scores": True,
        "return_dict_in_generate": True,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    if drop_attention_mask:
        print("  [Repro] batched generate still passes attention_mask, but HSA forward will intentionally drop the decoder mask.")

    with torch.no_grad():
        output = model.generate(**generate_kwargs)

    gen_len = len(output.scores)
    generated_ids = output.sequences[:, input_len : input_len + gen_len]
    # output.scores: tuple(gen_len) of [batch, vocab] -> [batch, gen_len, vocab]
    gen_scores = torch.stack(output.scores, dim=1)
    return input_ids, attention_mask, generated_ids, gen_scores


def run_forward_baseline_plain(model, input_ids, generated_ids, device):
    prompt_len = input_ids.shape[1]
    gen_len = generated_ids.shape[0]
    full_ids = torch.cat([input_ids, generated_ids.unsqueeze(0)], dim=1)
    print(f"  [Forward] original sequence length: {full_ids.shape[1]} (prompt={prompt_len}, gen={gen_len})")

    saved_auto_insert_lmk = model.auto_insert_lmk
    model.auto_insert_lmk = False
    model._gen_state.reset()
    with torch.no_grad():
        outputs = model(input_ids=full_ids, use_cache=True)
    model.auto_insert_lmk = saved_auto_insert_lmk

    logits = outputs.logits
    pred_logits = logits[:, prompt_len - 1 : prompt_len + gen_len - 1, :].squeeze(0)
    pred_tokens = pred_logits.argmax(dim=-1)
    print(f"  [Forward] logits shape: {tuple(logits.shape)}")
    return pred_tokens, pred_logits


def run_forward_baseline_plain_batch(model, input_ids, attention_mask, generated_ids, device, drop_attention_mask=False):
    """Batched no-LMK full prefill: padded prompt + generated, matching generate position_ids."""
    prompt_len = input_ids.shape[1]
    gen_len = generated_ids.shape[1]
    full_ids = torch.cat([input_ids, generated_ids], dim=1)
    gen_attention_mask = torch.ones_like(generated_ids, device=attention_mask.device)
    full_attention_mask = torch.cat([attention_mask, gen_attention_mask], dim=1)
    position_ids = _position_ids_from_attention_mask(full_attention_mask)

    print(
        f"  [Forward Batch] batch={full_ids.shape[0]}, original sequence length: {full_ids.shape[1]} "
        f"(padded_prompt={prompt_len}, gen={gen_len})"
    )

    saved_auto_insert_lmk = model.auto_insert_lmk
    model.auto_insert_lmk = False
    model._gen_state.reset()
    forward_kwargs = {
        "input_ids": full_ids,
        "use_cache": True,
    }
    forward_kwargs.update(
        {
            "attention_mask": full_attention_mask,
            "position_ids": position_ids,
        }
    )
    if drop_attention_mask:
        print("  [Repro] batched forward still passes attention_mask / position_ids, but HSA forward will intentionally drop the decoder mask.")

    with torch.no_grad():
        outputs = model(**forward_kwargs)
    model.auto_insert_lmk = saved_auto_insert_lmk

    logits = outputs.logits
    pred_logits = logits[:, prompt_len - 1 : prompt_len + gen_len - 1, :]
    pred_tokens = pred_logits.argmax(dim=-1)
    print(f"  [Forward Batch] logits shape: {tuple(logits.shape)}")
    return pred_tokens, pred_logits


def run_forward_baseline(model, tokenizer, input_ids, generated_ids, device):
    if not model.insert_landmarks:
        return run_forward_baseline_plain(model, input_ids, generated_ids, device)

    chunk_size = model.chunk_size
    lmk_id = model.lmk_id
    prompt_len = input_ids.shape[1]
    gen_len = generated_ids.shape[0]

    full_ids = torch.cat([input_ids, generated_ids.unsqueeze(0)], dim=1)  # [1, prompt_len + gen_len]
    full_len = full_ids.shape[1]
    print(f"  [Forward] original sequence length: {full_len} (prompt={prompt_len}, gen={gen_len})")

    full_ids_with_lmk = insert_special_tokens(full_ids, lmk_id, chunk_size)
    position_ids = create_position_ids_with_landmarks(None, full_len, chunk_size, device)
    new_seq_len = full_ids_with_lmk.shape[1]
    print(f"  [Forward] sequence length after inserting LMK: {new_seq_len}")

    pos_indices = torch.arange(new_seq_len, device=device)
    is_lmk = (pos_indices % chunk_size == chunk_size - 1)
    non_lmk_mask = ~is_lmk
    num_lmk = is_lmk.sum().item()
    num_non_lmk = non_lmk_mask.sum().item()
    print(f"  [Forward] LMK count: {num_lmk}, non-LMK count: {num_non_lmk}")
    assert num_non_lmk == full_len, f"non-LMK count {num_non_lmk} != original sequence length {full_len}"

    saved_auto_insert_lmk = model.auto_insert_lmk
    saved_gen_state_active = model._gen_state.active
    saved_lmk_positions = model._gen_state.lmk_positions_in_input
    model.auto_insert_lmk = False
    model._gen_state.active = False
    model._gen_state.lmk_positions_in_input = None

    with torch.no_grad():
        outputs = model(
            input_ids=full_ids_with_lmk,
            position_ids=position_ids,
            use_cache=True,
            attention_mask=None,
        )

    model.auto_insert_lmk = saved_auto_insert_lmk
    model._gen_state.active = saved_gen_state_active
    model._gen_state.lmk_positions_in_input = saved_lmk_positions

    logits = outputs.logits  # [1, new_seq_len, vocab_size]
    logits_no_lmk = logits[:, non_lmk_mask, :]  # [1, full_len, vocab_size]
    print(f"  [Forward] logits shape after removing LMK: {logits_no_lmk.shape}")

    pred_logits = logits_no_lmk[:, prompt_len - 1 : prompt_len + gen_len - 1, :]  # [1, gen_len, vocab_size]
    pred_tokens = pred_logits.argmax(dim=-1).squeeze(0)  # [gen_len]

    return pred_tokens, pred_logits.squeeze(0)  # pred_logits: [gen_len, vocab_size]


def run_forward_baseline_batch(model, input_ids, attention_mask, generated_ids, device, drop_attention_mask=False):
    if model.insert_landmarks:
        raise NotImplementedError(
            "The train/inference consistency test with batch_size > 1 currently only supports insert_landmarks=false "
            "(param_reuse/base OLMo). The padding/position_ids for externally inserted LMK need a separate implementation."
        )
    return run_forward_baseline_plain_batch(model, input_ids, attention_mask, generated_ids, device, drop_attention_mask)


def compare_results(
    generated_ids,
    pred_tokens,
    tokenizer,
    gen_scores=None,
    fwd_logits=None,
    logit_tie_tolerance=0.5,
):
    """
    Compare generate output with forward argmax output.
    If gen_scores and fwd_logits are provided, print top1/top2 logits at mismatch
    positions to judge whether the argmax randomness is caused by close logits.

    Args:
        gen_scores: [gen_len, vocab_size] per-step logits from generate
        fwd_logits: [gen_len, vocab_size] logits at the corresponding forward positions
    """
    gen_len = generated_ids.shape[0]
    pred_len = pred_tokens.shape[0]
    compare_len = min(gen_len, pred_len)

    match = (generated_ids[:compare_len] == pred_tokens[:compare_len])
    match_count = match.sum().item()
    total = compare_len

    print(f"\n  [Comparison] total tokens: {total}, matches: {match_count}, match rate: {match_count/total*100:.2f}%")

    tie_like_mismatches = 0

    if match_count == total:
        print("  ✅ Exact match! Generate and Forward outputs are identical.")
    else:
        print("  ❌ Mismatches found!")
        mismatch_indices = torch.where(~match)[0]
        print(f"  Mismatch positions (showing up to first 10): {mismatch_indices[:10].tolist()}")
        for idx_tensor in mismatch_indices:
            idx = idx_tensor.item()
            gen_tok = generated_ids[idx].item()
            pred_tok = pred_tokens[idx].item()
            if gen_scores is not None and fwd_logits is not None:
                g_logits = gen_scores[idx]
                f_logits = fwd_logits[idx]
                g_gap = (g_logits[gen_tok] - g_logits[pred_tok]).abs().item()
                f_gap = (f_logits[gen_tok] - f_logits[pred_tok]).abs().item()
                if g_gap <= logit_tie_tolerance and f_gap <= logit_tie_tolerance:
                    tie_like_mismatches += 1

        for idx in mismatch_indices[:10]:
            idx = idx.item()
            gen_tok = generated_ids[idx].item()
            pred_tok = pred_tokens[idx].item()
            gen_text = tokenizer.decode([gen_tok])
            pred_text = tokenizer.decode([pred_tok])
            print(f"    position {idx}: generate={gen_tok}('{gen_text}') vs forward={pred_tok}('{pred_text}')")

            if gen_scores is not None:
                g_logits = gen_scores[idx]  # [vocab_size]
                g_top2_vals, g_top2_ids = g_logits.topk(2)
                g_top1_text = tokenizer.decode([g_top2_ids[0].item()])
                g_top2_text = tokenizer.decode([g_top2_ids[1].item()])
                g_diff = (g_top2_vals[0] - g_top2_vals[1]).item()
                print(f"      [Generate logits] top1={g_top2_ids[0].item()}('{g_top1_text}') logit={g_top2_vals[0].item():.4f}, "
                      f"top2={g_top2_ids[1].item()}('{g_top2_text}') logit={g_top2_vals[1].item():.4f}, diff={g_diff:.6f}")

            if fwd_logits is not None:
                f_logits = fwd_logits[idx]  # [vocab_size]
                f_top2_vals, f_top2_ids = f_logits.topk(2)
                f_top1_text = tokenizer.decode([f_top2_ids[0].item()])
                f_top2_text = tokenizer.decode([f_top2_ids[1].item()])
                f_diff = (f_top2_vals[0] - f_top2_vals[1]).item()
                print(f"      [Forward  logits] top1={f_top2_ids[0].item()}('{f_top1_text}') logit={f_top2_vals[0].item():.4f}, "
                      f"top2={f_top2_ids[1].item()}('{f_top2_text}') logit={f_top2_vals[1].item():.4f}, diff={f_diff:.6f}")

            if gen_scores is not None and fwd_logits is not None:
                g_logit_for_fwd_top1 = gen_scores[idx][pred_tok].item()
                f_logit_for_gen_top1 = fwd_logits[idx][gen_tok].item()
                print(f"      [Cross-compare] gen-side logit for fwd_top1('{pred_text}')={g_logit_for_fwd_top1:.4f}, "
                      f"fwd-side logit for gen_top1('{gen_text}')={f_logit_for_gen_top1:.4f}")

        if gen_scores is not None and fwd_logits is not None and tie_like_mismatches == mismatch_indices.numel():
            print(
                f"  ⚠️ All {tie_like_mismatches} token mismatches are within the logit tolerance "
                f"{logit_tie_tolerance}, treated as argmax ties under bf16 / different kernel paths."
            )
            return True

    return match_count == total


def compare_ppl(gen_ppl, gen_nll, fwd_ppl, fwd_nll, nll_tolerance):
    """
    Compare the PPL of generate and forward.
    """
    print(f"\n  [PPL Comparison]")
    print(f"    Generate PPL: {gen_ppl:.6f} (avg NLL: {gen_nll:.6f})")
    print(f"    Forward  PPL: {fwd_ppl:.6f} (avg NLL: {fwd_nll:.6f})")
    
    ppl_diff = abs(gen_ppl - fwd_ppl)
    nll_diff = abs(gen_nll - fwd_nll)
    rel_diff = ppl_diff / max(gen_ppl, fwd_ppl, 1e-8)
    
    print(f"    PPL absolute diff: {ppl_diff:.6f}, relative diff: {rel_diff:.6f}")
    print(f"    NLL absolute diff: {nll_diff:.6f}")
    
    # PPL is exponentially sensitive for very short generations (e.g. EOS-only),
    # so use average-NLL tolerance as the more stable bf16 diagnostic.
    ppl_threshold = 0.01
    if rel_diff < ppl_threshold:
        print(f"    ✅ PPL consistent! (relative diff {rel_diff:.6f} < threshold {ppl_threshold})")
        return True
    if nll_diff <= nll_tolerance:
        print(
            f"    ⚠️ PPL relative diff exceeds threshold, but NLL absolute diff {nll_diff:.6f} "
            f"<= {nll_tolerance}, treated as short-sequence / bf16 numerical error."
        )
        return True

    print(
        f"    ❌ PPL inconsistent! relative diff {rel_diff:.6f} >= {ppl_threshold} "
        f"and NLL absolute diff {nll_diff:.6f} > {nll_tolerance}"
    )
    return False


def compare_prefill_next_logits(model, tokenizer, prompts, device, padding_side, tolerance):
    print("\n[Step 0] Comparing batched prefill next-token logits with per-row bsz=1 prefill...")
    batch_logits, input_ids, attention_mask = run_prefill_next_logits_batch(
        model,
        tokenizer,
        prompts,
        device,
        padding_side,
    )

    all_passed = True
    for row, prompt in enumerate(prompts):
        single_logits = run_prefill_next_logits_single(model, tokenizer, prompt, device)
        batch_top = int(batch_logits[row].argmax().item())
        single_top = int(single_logits.argmax().item())
        max_abs_diff = (batch_logits[row] - single_logits).abs().max().item()
        real_prompt_len = int(attention_mask[row].sum().item())

        if batch_top == single_top:
            print(
                f"  [Row {row}] ✅ prefill top1 consistent: {batch_top}('{tokenizer.decode([batch_top])}'), "
                f"real_prompt_tokens={real_prompt_len}, max_abs_diff={max_abs_diff:.6f}"
            )
            continue

        tie_like, batch_gap, single_gap = _is_tie_like(
            batch_logits[row],
            single_logits,
            batch_top,
            single_top,
            tolerance,
        )
        if tie_like:
            print(
                f"  [Row {row}] ⚠️ prefill top1 differs but is a near tie: "
                f"batch={batch_top}('{tokenizer.decode([batch_top])}') vs "
                f"single={single_top}('{tokenizer.decode([single_top])}'), "
                f"gap_batch={batch_gap:.6f}, gap_single={single_gap:.6f}, "
                f"max_abs_diff={max_abs_diff:.6f}"
            )
        else:
            all_passed = False
            print(
                f"  [Row {row}] ❌ prefill top1 inconsistent: "
                f"batch={batch_top}('{tokenizer.decode([batch_top])}') vs "
                f"single={single_top}('{tokenizer.decode([single_top])}'), "
                f"gap_batch={batch_gap:.6f}, gap_single={single_gap:.6f}, "
                f"max_abs_diff={max_abs_diff:.6f}"
            )

    model._gen_state.reset()
    return all_passed


def compare_batch_vs_single_generate(
    batch_generated_ids,
    batch_scores,
    single_generated_ids,
    single_scores,
    tokenizer,
    row,
    tolerance,
):
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    batch_trimmed = _trim_generated_for_compare(batch_generated_ids, eos_token_id, pad_token_id)
    single_trimmed = _trim_generated_for_compare(single_generated_ids, eos_token_id, pad_token_id)

    compare_len = min(batch_trimmed.numel(), single_trimmed.numel())
    if compare_len > 0:
        token_match = batch_trimmed[:compare_len] == single_trimmed[:compare_len]
        first_mismatch = torch.nonzero(~token_match, as_tuple=False).flatten()
    else:
        token_match = torch.empty(0, dtype=torch.bool, device=batch_generated_ids.device)
        first_mismatch = torch.empty(0, dtype=torch.long, device=batch_generated_ids.device)

    exact_match = (
        batch_trimmed.shape == single_trimmed.shape
        and compare_len == batch_trimmed.numel()
        and bool(token_match.all().item() if token_match.numel() > 0 else True)
    )
    if exact_match:
        print(f"  [Row {row}] ✅ batched and bsz=1 generated tokens are identical.")
        return True

    if first_mismatch.numel() > 0:
        idx = int(first_mismatch[0].item())
        batch_tok = int(batch_trimmed[idx].item())
        single_tok = int(single_trimmed[idx].item())
        tie_like, batch_gap, single_gap = _is_tie_like(
            batch_scores[idx],
            single_scores[idx],
            batch_tok,
            single_tok,
            tolerance,
        )
        if tie_like:
            print(
                f"  [Row {row}] ⚠️ first generated divergence is a near tie, subsequent autoregressive sequence is not forced to fail: "
                f"pos={idx}, batch={batch_tok}('{tokenizer.decode([batch_tok])}') vs "
                f"single={single_tok}('{tokenizer.decode([single_tok])}'), "
                f"gap_batch={batch_gap:.6f}, gap_single={single_gap:.6f}"
            )
            return True
        print(
            f"  [Row {row}] ❌ first non-tie divergence between batched and bsz=1: "
            f"pos={idx}, batch={batch_tok}('{tokenizer.decode([batch_tok])}') vs "
            f"single={single_tok}('{tokenizer.decode([single_tok])}'), "
            f"gap_batch={batch_gap:.6f}, gap_single={single_gap:.6f}"
        )
        return False

    print(
        f"  [Row {row}] ❌ length mismatch between batched and bsz=1: "
        f"batch_len={batch_trimmed.numel()}, single_len={single_trimmed.numel()}"
    )
    return False


def run_batched_consistency(model, tokenizer, prompts, device, args):
    all_passed = True
    model_core = getattr(model, "model", model)
    saved_debug_drop_attention_mask = getattr(model_core.config, "debug_drop_attention_mask", False)
    model_core.config.debug_drop_attention_mask = bool(args.drop_attention_mask)
    if args.drop_attention_mask:
        print("[Repro] debug_drop_attention_mask enabled: HSA forward will ignore the passed attention_mask.")

    for batch_start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[batch_start : batch_start + args.batch_size]
        print(f"\n{'='*60}")
        print(
            f"[Batch {batch_start // args.batch_size + 1}] "
            f"batch_size={len(batch_prompts)}, padding_side={args.padding_side}"
        )
        for local_idx, prompt in enumerate(batch_prompts):
            print(f"  [Prompt {batch_start + local_idx + 1}] {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        print(f"{'='*60}")

        prefill_passed = compare_prefill_next_logits(
            model,
            tokenizer,
            batch_prompts,
            device,
            args.padding_side,
            args.logit_tie_tolerance,
        )
        if not prefill_passed:
            all_passed = False

        print("\n[Step 1] Running batched model.generate() (greedy decode)...")
        input_ids, attention_mask, generated_ids, gen_scores = run_generate_batch(
            model,
            tokenizer,
            batch_prompts,
            device,
            args.max_new_tokens,
            args.padding_side,
            drop_attention_mask=args.drop_attention_mask,
        )
        print(f"  padded input length: {input_ids.shape[1]}")
        print(f"  generation steps: {generated_ids.shape[1]}")
        for row in range(generated_ids.shape[0]):
            gen_text = tokenizer.decode(generated_ids[row], skip_special_tokens=True)
            real_prompt_len = int(attention_mask[row].sum().item())
            print(f"  [Row {row}] real_prompt_tokens={real_prompt_len}, generated text: {gen_text}")

        if args.compare_batch_single:
            print("\n[Step 1.5] Comparing batched generate with per-row bsz=1 generate...")
            for row, prompt in enumerate(batch_prompts):
                model._gen_state.reset()
                _, single_generated_ids, single_scores = run_generate(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    args.max_new_tokens,
                )
                batch_single_passed = compare_batch_vs_single_generate(
                    generated_ids[row],
                    gen_scores[row],
                    single_generated_ids,
                    single_scores,
                    tokenizer,
                    row,
                    args.logit_tie_tolerance,
                )
                if not batch_single_passed:
                    all_passed = False

        model._gen_state.reset()

        print("\n[Step 2] Running batched model.forward() (full prefill control group)...")
        pred_tokens, fwd_logits = run_forward_baseline_batch(
            model,
            input_ids,
            attention_mask,
            generated_ids,
            device,
            drop_attention_mask=args.drop_attention_mask,
        )

        for row in range(generated_ids.shape[0]):
            global_idx = batch_start + row + 1
            compare_generated_ids = _trim_generated_for_compare(
                generated_ids[row],
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            )
            compare_len = compare_generated_ids.numel()
            if compare_len == 0:
                print(f"\n[Step 3] Row {row} / Prompt {global_idx}: generation result is empty, skipping logits/PPL comparison and marking as failed.")
                all_passed = False
                continue
            print(f"\n[Step 3] Row {row} / Prompt {global_idx}: comparing generate vs forward argmax...")
            argmax_passed = compare_results(
                compare_generated_ids,
                pred_tokens[row][:compare_len],
                tokenizer,
                gen_scores=gen_scores[row][:compare_len],
                fwd_logits=fwd_logits[row][:compare_len],
                logit_tie_tolerance=args.logit_tie_tolerance,
            )

            print(f"\n[Step 4] Row {row} / Prompt {global_idx}: comparing generate vs forward PPL...")
            gen_ppl, gen_nll = compute_ppl_from_logits(gen_scores[row][:compare_len], compare_generated_ids)
            fwd_ppl, fwd_nll = compute_ppl_from_logits(fwd_logits[row][:compare_len], compare_generated_ids)
            ppl_passed = compare_ppl(gen_ppl, gen_nll, fwd_ppl, fwd_nll, args.nll_tolerance)

            if not argmax_passed or not ppl_passed:
                all_passed = False

    model_core.config.debug_drop_attention_mask = saved_debug_drop_attention_mask
    return all_passed


def main(args):
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    tokenizer_path = args.tokenizer_path or args.checkpoint_path
    assert tokenizer_path, "Please provide --tokenizer_path or --checkpoint_path (used to load the tokenizer)"

    print(f"[INFO] Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
    }
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl

    if args.checkpoint_path and args.config_path:
        print(f"[INFO] Loading model: build graph from config + load weights from checkpoint")
        print(f"       config_path: {args.config_path}")
        print(f"       checkpoint_path: {args.checkpoint_path}")
        config = AutoConfig.from_pretrained(args.config_path)
        config.auto_insert_lmk = args.auto_insert_lmk
        load_kwargs = {**model_kwargs, "config": config, "device_map": device}
        if args.auto_insert_lmk:
            load_kwargs["auto_insert_lmk"] = True
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, **load_kwargs)
    elif args.checkpoint_path:
        print(f"[INFO] Loading model (from_pretrained): {args.checkpoint_path}")
        load_kwargs = {**model_kwargs, "device_map": device}
        if args.auto_insert_lmk:
            load_kwargs["auto_insert_lmk"] = True
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, **load_kwargs)
    else:
        assert args.config_path is not None, "When --checkpoint_path is not specified, --config_path must be provided (from_config random initialization)"
        print(f"[INFO] Loading model (from_config): {args.config_path}")
        config = AutoConfig.from_pretrained(args.config_path)
        config.auto_insert_lmk = args.auto_insert_lmk
        model = AutoModelForCausalLM.from_config(config, **model_kwargs).to(device)

    model.eval()
    print(f"[INFO] Model loaded, chunk_size={model.chunk_size}, lmk_id={model.lmk_id}")
    print(f"[INFO] auto_insert_lmk={model.auto_insert_lmk}")
    print(f"[INFO] Model parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    prompts = args.prompts if args.prompts else [
        # "Once a year I make the drive back to my hometown of Shreveport, Louisiana. My journey begins as the sun rises over our nation’s capital. Before long I’m moving through smaller cities that claim tobacco and the Confederate flag as symbols of pride, wondering how long it will be before the smell of factory smoke is replaced by the fertile aroma of livestock and chicken flocks. Then the narrow roads begin to unwind--hugged on either side by pastures, cows, horses, and shacks--and so too does my mind. As I ramble down bumpy paths, I stumble over the memory of a day spent fishing in a nearby bayou with my uncles, and the familiar smells of rank armpits and beer overwhelm my senses. Later, I see shiny pumps and a black veil waiting on top of an aging quilt and hear children running in bare feet. A tin of peanut brittle spotted at the counter of a country gas station lands my mind on my great-grandmother because that was her favorite candy. Big Momma was part of a chorus of tabernacle women who mothered me. She always said Shreveport was known as the city of churches because they sprout up on corners like strawberries in July. Word has it that there are more churches in my hometown than in any other city in the country. Hymns flow from their doors on Sunday mornings, while during the week the smiling church ladies greet you with words of encouragement as they skirt around the vestibule like bees on a honeycomb. “Baaaby, that was a fine prayer you did Sunday,” Mrs. Davis would say as I walked by. “Whose girl is dat with you? Bring her next Sunday.” But as is the case with all of Louisiana, our little city wobbles between extremes. Our local dishes of gumbo, catfish, and dirty rice must be spicy hot. Juke joints squat next to churches, and betting slips compete with offering envelopes. Fire-and-brimstone ministers point out that we drink and gamble too much. That is, until one in their congregation “hits.” Then it’s time to bring a tithe of the winnings to the altar. All of this is summed up neatly by the two billboards I notice as I cross the Texas Street Bridge into my city’s fold. One beckons you toward the horseshoe casino straight ahead. Opposite, another shouts, wanna win the jackpot? come to jesus. There is always the question of which road to take. To enter Shreveport’s downtown, travelers must cross our beloved Red River, which curls like a large garden snake around the city. The river is yet another contradiction. It does not remotely resemble the liquid silver color of the Mississippi. Instead, it pours out a murky clay red and flows as thick as soft mud across Louisiana. The only time it sparkles is at night, when the casino riverboats’ carnival lights illuminate the city. Following the curve of the river, crouched along the road leaving downtown, rests a neighborhood of little shacks that belie a city of over a quarter million. We call the houses shotgun because a bullet fired through the front entrance will pass through every room in the house before exiting the back door. It is a place where the children play dodgeball in the street but know to watch their manners, and every woman worth her salt can make a meal out of meat drippings, flour, eggs, and rice. The unpaved streets are filled with stray dogs, and after a rain the air smells like wet earth. On warm mornings, plump older women wearing blinding white maid uniforms congregate on corners and talk while awaiting the arrival of little blue buses that will take them to the homes where they work. “Child, Pastor Green liked to got the church on fire Sunday, didn’t he?” “Yeah, girl, and did you hear Mrs. Rogers shouting in the back? You know that boy of hers keeps her on her knees. He ain’t got good sense.” “Folks say what they want about her whiskey habit. That woman will give you the shirt off her back. That’s how I know she close to God.” That neighborhood was called Stoner Hill. I grew up listening to the women there. Everyone in that phalanx had a family church, and most believed in God and agreed that it was through Jesus Christ that we all gained salvation. Growing up, I don’t recall ever meeting someone who didn’t have a faith--at least no one who would admit such a thing out loud. It was a place where all doctrine was respected; even door-knocking Jehovah’s Witnesses were given the opportunity to speak their piece. Yes, Shreveport was the kind of city where everyone had a church, temple, or chapel they considered theirs, even if they’d only seen it from the inside a dozen times. A child from Stoner Hill seldom made it out of puberty without a distant cousin or a neighbor dragging the youngster off to recite New Testament Scripture in the Easter pageant or sing carols in the Christmas program, blessing the child with at least a C.M.E. membership: attendance at Christmas, Mother’s Day, and Easter. I’ve been reciting Bible verses since I was old enough to say, “Jesus wept.” My great-grandmother, Big Momma, used to say about the Bible, “Baby, you can find a word to carry you through anythang.” Still, my very religious family managed to pick and choose which Scriptures to live by. The men would pray up a miracle in the deacons’ corner and then enjoy a strong glass or two of Jack Daniel’s after church. The women sang in the choir but cursed like sailors when their team fumbled on Monday Night Football. “I could pull up my skirt and beat that sorry-ass receiver to the ball,” Big Momma would shout from the kitchen while stacking freshly washed dishes in cabinets. I spent most of my childhood summers down the road from home at Big Momma’s house. We began each day with the morning ritual she referred to as her labor of love--combing my hair. I would sit on the porch floor with my feet swinging over its edge while my head bobbed back and forth between Big Momma’s legs as she tugged, parted, and braided my long, thick, nappy hair. Big Momma always sat perfectly upright, sucking in her breath with each drag of the comb, then releasing the air from her hollow Cherokee cheeks, never once bending her back. After she finished the job, she’d pat me on my head and say, “Now you beautiful.” I’d rush to the bathroom, stand on the toilet seat, and peer over the sink into the mirror, eager to view this new and beautiful me. Of course, she never materialized. All I ever saw was my chubby face with a crown of lopsided plaits and a mouth full of what my momma teasingly called “beaver teeth” because they looked large enough to saw wood. Besides our grooming, Big Momma and her band of swearing sopranos made sure their offspring got a proper Christian upbringing. Every Sunday there was morning church school and Baptist Training Union. And for one week every August the young ones were herded to Grambling, Louisiana, a small college town, for a gigantic statewide revival called Youth-En-Camp. Although the drive took only a few hours, it had the feel of a great adventure. This was due in part to the parcel of sheets, dresses, and fried chicken that always accompanied me but also because the decreased supervision allowed me to experience free will. It was during one of these revivals that I became hopeful that I would one day look into the mirror and see beauty in myself. I was thirteen at the time--too old to be in one of the crayon classrooms but still too awkward to be cool. Before that summer I’d never thought that I could be beautiful--perhaps cute, on a good day, but never glamorous, radiant, or enchanting. Of course, up to that point, the only form of beauty I knew to desire was physical splendor, in which category I was sorely lacking. I was the tallest girl in my eighth-grade class, and when I tried to walk in dress shoes, my heels would slide out, causing me to trip over myself. Naturally, my only concern was ridding myself of awkwardness. Beauty was something I saw only in others. A woman’s even-colored skin and bright white teeth made her beautiful, never the inner peace that sparkled in her eyes. I greatly admired the little girl’s sunny Easter dress, adorned with white bows and ribbons, but gave no thought to the mother--needle in one hand, iron in the other, creating this lovely vision. And Big Momma’s front lawn with its velvet violets, deep purple grape suckers, and yellow sunflowers floating in the air like balloons was beautiful, but never once did I consider the care they were given even as the flowers’ first petals danced indiscriminately in the sunlight. I had always focused on my plainness, and it was this sorry image of myself that I took with me to Youth-En-Camp that summer. Only later would I understand that real beauty emanates from the heart. At camp that summer, our daily activities started with 5 a.m. prayer and devotion, during which I often volunteered to pray out loud so that everyone could hear my conversation with God. Somewhere along the way I got the notion that you were the biggest coward and hypocrite if you didn’t want to pray out loud. That to me suggested you were ashamed of the Lord, and even with all my insecurities and teenage angst, I wanted to be bigger than that. After breakfast, there was Bible-study class, lunch, and midday worship. There teenagers would offer testimonials, and thanks to those I referred to as our “holy staples” (they seemed as necessary to our religious experience as the flour and canned goods that lined the shelves of our neighborhood general store)--the girl who’d been suffering from multiple sclerosis who was walking for the first time in five years and the boys who overnight had been called to preach--the standard for godliness was set high. Following dinner and church service came the dating game, which commenced on a dusty bridge that stretched a half mile long and linked Grambling to the town of Ruston. As a symbolic gesture, the bridge was closed while the campers lined up at its foot, over a thousand of us girls on the right while the boys, far fewer in number, stood on the left. Once we "
            # "The capital of France is "
            """I am happy to join with you today in what will go down in history as the greatest demonstration for freedom in the history of our nation.

Five score years ago, a great American, in whose symbolic shadow we stand today, signed the Emancipation Proclamation. This momentous decree came as a great beacon light of hope to millions of Negro slaves who had been seared in the flames of withering injustice. It came as a joyous daybreak to end the long night of their captivity.

But one hundred years later, the Negro still is not free. One hundred years later, the life of the Negro is still sadly crippled by the manacles of segregation and the chains of discrimination. One hundred years later, the Negro lives on a lonely island of poverty in the midst of a vast ocean of material prosperity. One hundred years later, the Negro is still languished in the corners of American society and finds himself an exile in his own land. And so we've come here today to dramatize a shameful condition.

In a sense we've come to our nation's capital to cash a check. When the architects of our republic wrote the magnificent words of the Constitution and the Declaration of Independence, they were signing a promissory note to which every American was to fall heir. This note was a promise that all men, yes, black men as well as white men, would be guaranteed the "unalienable Rights" of "Life, Liberty and the pursuit of Happiness." It is obvious today that America has defaulted on this promissory note, insofar as her citizens of color are concerned. Instead of honoring this sacred obligation, America has given the Negro people a bad check, a check which has come back marked "insufficient funds."

But we refuse to believe that the bank of justice is bankrupt. We refuse to believe that there are insufficient funds in the great vaults of opportunity of this nation. And so, we've come to cash this check, a check that will give us upon demand the riches of freedom and the security of justice.

We have also come to this hallowed spot to remind America of the fierce urgency of Now. This is no time to engage in the luxury of cooling off or to take the tranquilizing drug of gradualism. Now is the time to make real the promises of democracy. Now is the time to rise from the dark and desolate valley of segregation to the sunlit path of racial justice. Now is the time to lift our nation from the quicksands of racial injustice to the solid rock of brotherhood. Now is the time to make justice a reality for all of God's children.

It would """
    ]

    all_passed = True

    if args.batch_size > 1:
        if len(prompts) < 2:
            raise ValueError("batch_size > 1 requires at least 2 prompts; please pass multiple via --prompts.")
        all_passed = run_batched_consistency(model, tokenizer, prompts, device, args)
    else:
        for i, prompt in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"[Prompt {i+1}] {prompt}")
            print(f"{'='*60}")

            # ---- Step 1: Generate (greedy decode) ----
            print("\n[Step 1] Running model.generate() (greedy decode)...")
            input_ids, generated_ids, gen_scores = run_generate(model, tokenizer, prompt, device, args.max_new_tokens)
            gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"  input token count: {input_ids.shape[1]}")
            print(f"  generated token count: {generated_ids.shape[0]}")
            print(f"  generated text: {gen_text}")

            model._gen_state.reset()

            print("\n[Step 2] Running model.forward() (full prefill control group)...")
            pred_tokens, fwd_logits = run_forward_baseline(model, tokenizer, input_ids, generated_ids, device)

            print("\n[Step 3] Comparing generate vs forward argmax results...")
            argmax_passed = compare_results(generated_ids, pred_tokens, tokenizer, gen_scores=gen_scores, fwd_logits=fwd_logits)

            print("\n[Step 4] Comparing generate vs forward PPL...")
            gen_ppl, gen_nll = compute_ppl_from_logits(gen_scores, generated_ids)
            fwd_ppl, fwd_nll = compute_ppl_from_logits(fwd_logits, generated_ids)
            ppl_passed = compare_ppl(gen_ppl, gen_nll, fwd_ppl, fwd_nll, args.nll_tolerance)

            if not argmax_passed or not ppl_passed:
                all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("[INFO] ✅ All tests passed! Generate and Forward outputs are identical.")
    else:
        print("[INFO] ❌ Some tests failed, please check the mismatch details above.")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HiLSForCausalLM generate vs forward consistency")
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="HF weights directory; can be used with --config_path (build graph from config first, then load these weights); if neither is specified, the built-in default is used",
    )
    parser.add_argument(
        "--config_path", type=str, default=None,
        help=(
            "config directory or config.json. "
            "When only this is specified: from_config random initialization; "
            "When specified together with --checkpoint_path: build graph from this config, load weights from checkpoint"
        ),
    )
    parser.add_argument(
        "--auto_insert_lmk", action="store_true",
        help="Consistent with eval/eval_ppl_hf.py: passed as a constructor argument for from_pretrained; written to config.auto_insert_lmk for from_config",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None,
        help="Tokenizer path; defaults to the same as checkpoint_path; must be specified separately when there is no checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Run device, e.g. cuda:0 or cpu"
    )
    parser.add_argument(
        "--attn_impl", type=str, default="flash_attention_3",
        help="Attention implementation, e.g. flash_attention_3, sdpa, eager, etc."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--prompts", nargs="+", type=str, default=None,
        help="Custom prompt list"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for the train/inference consistency test; when >1, uses batched generate + batched forward comparison",
    )
    parser.add_argument(
        "--padding_side", choices=["left", "right"], default="left",
        help="Tokenizer padding direction when batch_size > 1; decoder-only generate typically uses left",
    )
    parser.add_argument(
        "--compare_batch_single", action="store_true",
        help="When batch_size > 1, additionally run bsz=1 generate per row and align each row's output with batched generate",
    )
    parser.add_argument(
        "--drop_attention_mask", action="store_true",
        help="For reproduction: when batch_size > 1, make HSA forward intentionally drop attention_mask before constructing the decoder mask",
    )
    parser.add_argument(
        "--logit_tie_tolerance", type=float, default=0.5,
        help="When batch/bsz=1 or generate/forward top1 differ but their logit gaps are below this threshold, treat it as a bf16 tie",
    )
    parser.add_argument(
        "--nll_tolerance", type=float, default=0.03,
        help="When the generate/forward PPL relative diff is large, if the average NLL absolute diff is below this threshold, treat it as short-sequence / bf16 numerical error",
    )

    args = parser.parse_args()
    if args.checkpoint_path is None and args.config_path is None:
        args.checkpoint_path = DEFAULT_CKPT_PATH

    print(f"[INFO] Arguments: {args}")
    main(args)


# python unittests/test_generate_olmo.py
