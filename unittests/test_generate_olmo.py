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

# 将项目根目录加入 sys.path，以便导入 models 和 utils
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 注册自定义模型到 transformers
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
    从 logits 和 target token ids 计算 PPL。
    
    Args:
        logits: [gen_len, vocab_size] 的 logits，logits[i] 对应预测 target_ids[i]
        target_ids: [gen_len] 的 target token ids
    
    Returns:
        ppl: 困惑度
        avg_nll: 平均负对数似然
    """
    # 计算交叉熵（逐 token）
    log_probs = F.log_softmax(logits.float(), dim=-1)  # [gen_len, vocab_size]
    # 取每个位置对应 target token 的 log probability
    target_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)  # [gen_len]
    # 平均负对数似然
    avg_nll = -target_log_probs.mean().item()
    ppl = math.exp(avg_nll)
    return ppl, avg_nll


def run_generate(model, tokenizer, prompt, device, max_new_tokens):
    """
    运行 model.generate()，greedy decode。
    返回 (input_ids, generated_token_ids, gen_scores)。
    其中 gen_scores 是每步 decode 的 logits（用于计算 PPL）。
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decode
            output_scores=True,  # 返回每步的 logits
            return_dict_in_generate=True,
        )

    generated_ids = output.sequences[0, input_len:]  # 只取新生成的部分
    # output.scores 是一个 tuple，每个元素是 [batch_size, vocab_size] 的 logits
    # 拼接成 [gen_len, vocab_size]
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
        print("  [Repro] batched generate 仍传入 attention_mask，但 HSA forward 内会故意丢弃 decoder mask。")

    with torch.no_grad():
        output = model.generate(**generate_kwargs)

    gen_len = len(output.scores)
    generated_ids = output.sequences[:, input_len : input_len + gen_len]
    # output.scores: tuple(gen_len) of [batch, vocab] -> [batch, gen_len, vocab]
    gen_scores = torch.stack(output.scores, dim=1)
    return input_ids, attention_mask, generated_ids, gen_scores


def run_forward_baseline_plain(model, input_ids, generated_ids, device):
    """无 LMK 的全量 prefill 对照：prompt + generated 一次 forward。"""
    prompt_len = input_ids.shape[1]
    gen_len = generated_ids.shape[0]
    full_ids = torch.cat([input_ids, generated_ids.unsqueeze(0)], dim=1)
    print(f"  [Forward] 原始序列长度: {full_ids.shape[1]} (prompt={prompt_len}, gen={gen_len})")

    saved_auto_insert_lmk = model.auto_insert_lmk
    model.auto_insert_lmk = False
    model._gen_state.reset()
    with torch.no_grad():
        outputs = model(input_ids=full_ids, use_cache=True)
    model.auto_insert_lmk = saved_auto_insert_lmk

    logits = outputs.logits
    pred_logits = logits[:, prompt_len - 1 : prompt_len + gen_len - 1, :].squeeze(0)
    pred_tokens = pred_logits.argmax(dim=-1)
    print(f"  [Forward] logits 形状: {tuple(logits.shape)}")
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
        f"  [Forward Batch] batch={full_ids.shape[0]}, 原始序列长度: {full_ids.shape[1]} "
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
        print("  [Repro] batched forward 仍传入 attention_mask / position_ids，但 HSA forward 内会故意丢弃 decoder mask。")

    with torch.no_grad():
        outputs = model(**forward_kwargs)
    model.auto_insert_lmk = saved_auto_insert_lmk

    logits = outputs.logits
    pred_logits = logits[:, prompt_len - 1 : prompt_len + gen_len - 1, :]
    pred_tokens = pred_logits.argmax(dim=-1)
    print(f"  [Forward Batch] logits 形状: {tuple(logits.shape)}")
    return pred_tokens, pred_logits


def run_forward_baseline(model, tokenizer, input_ids, generated_ids, device):
    """
    全量 forward 对照组：
    - insert_landmarks=True: 外部 insert LMK，再 forward（与 eval_ppl_hf 一致）
    - insert_landmarks=False: 直接 forward 原始 token 序列（param_reuse / base OLMo）
    """
    if not model.insert_landmarks:
        return run_forward_baseline_plain(model, input_ids, generated_ids, device)

    chunk_size = model.chunk_size
    lmk_id = model.lmk_id
    prompt_len = input_ids.shape[1]
    gen_len = generated_ids.shape[0]

    # 1. 拼接完整序列：prompt + generated
    full_ids = torch.cat([input_ids, generated_ids.unsqueeze(0)], dim=1)  # [1, prompt_len + gen_len]
    full_len = full_ids.shape[1]
    print(f"  [Forward] 原始序列长度: {full_len} (prompt={prompt_len}, gen={gen_len})")

    # 2. 外部插入 LMK token（和 eval_ppl_hf.py 一致）
    full_ids_with_lmk = insert_special_tokens(full_ids, lmk_id, chunk_size)
    position_ids = create_position_ids_with_landmarks(None, full_len, chunk_size, device)
    new_seq_len = full_ids_with_lmk.shape[1]
    print(f"  [Forward] 插入 LMK 后序列长度: {new_seq_len}")

    # 3. 构建 LMK 位置的 mask（用于后续剔除 logits）
    #    insert_special_tokens 在每 (chunk_size-1) 个 real token 后插入一个 LMK
    #    LMK 位于 index % chunk_size == chunk_size - 1 的位置
    #    最后不完整 chunk 没有 LMK（remainder < chunk_size，不会触发该条件）
    pos_indices = torch.arange(new_seq_len, device=device)
    is_lmk = (pos_indices % chunk_size == chunk_size - 1)
    non_lmk_mask = ~is_lmk
    num_lmk = is_lmk.sum().item()
    num_non_lmk = non_lmk_mask.sum().item()
    print(f"  [Forward] LMK 数量: {num_lmk}, 非 LMK 数量: {num_non_lmk}")
    assert num_non_lmk == full_len, f"非 LMK 数量 {num_non_lmk} != 原始序列长度 {full_len}"

    # 4. 关闭 auto_insert_lmk 和 _gen_state.active，防止模型内部重复处理
    saved_auto_insert_lmk = model.auto_insert_lmk
    saved_gen_state_active = model._gen_state.active
    saved_lmk_positions = model._gen_state.lmk_positions_in_input
    model.auto_insert_lmk = False
    model._gen_state.active = False
    model._gen_state.lmk_positions_in_input = None

    # 使用 use_cache=True，和 eval_ppl_hf.py 保持一致
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids_with_lmk,
            position_ids=position_ids,
            use_cache=True,
            attention_mask=None,
        )

    # 恢复状态
    model.auto_insert_lmk = saved_auto_insert_lmk
    model._gen_state.active = saved_gen_state_active
    model._gen_state.lmk_positions_in_input = saved_lmk_positions

    # 5. 从 logits 中外部剔除 LMK 位置（和 eval_ppl_hf.py 中处理 label 的方式对应）
    logits = outputs.logits  # [1, new_seq_len, vocab_size]
    logits_no_lmk = logits[:, non_lmk_mask, :]  # [1, full_len, vocab_size]
    print(f"  [Forward] 剔除 LMK 后 logits 形状: {logits_no_lmk.shape}")

    # 6. Causal LM shift: logits[i] 预测 token[i+1]
    #    要预测 generated tokens，需要取 logits[prompt_len-1 : prompt_len+gen_len-1]
    pred_logits = logits_no_lmk[:, prompt_len - 1 : prompt_len + gen_len - 1, :]  # [1, gen_len, vocab_size]
    pred_tokens = pred_logits.argmax(dim=-1).squeeze(0)  # [gen_len]

    # 同时返回 pred_logits 用于计算 PPL
    return pred_tokens, pred_logits.squeeze(0)  # pred_logits: [gen_len, vocab_size]


def run_forward_baseline_batch(model, input_ids, attention_mask, generated_ids, device, drop_attention_mask=False):
    if model.insert_landmarks:
        raise NotImplementedError(
            "batch_size > 1 的训推一致测试目前只支持 insert_landmarks=false "
            "(param_reuse/base OLMo)。LMK 外插入的 padding/position_ids 需要单独实现。"
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
    对比 generate 输出和 forward argmax 输出。
    如果提供了 gen_scores 和 fwd_logits，会在不匹配位置打印 top1/top2 的 logits，
    以判断是否因为 logits 接近导致 argmax 的随机性。
    
    Args:
        gen_scores: [gen_len, vocab_size] generate 每步的 logits
        fwd_logits: [gen_len, vocab_size] forward 对应位置的 logits
    """
    gen_len = generated_ids.shape[0]
    pred_len = pred_tokens.shape[0]
    compare_len = min(gen_len, pred_len)

    match = (generated_ids[:compare_len] == pred_tokens[:compare_len])
    match_count = match.sum().item()
    total = compare_len

    print(f"\n  [对比结果] 总 token 数: {total}, 匹配数: {match_count}, 匹配率: {match_count/total*100:.2f}%")

    tie_like_mismatches = 0

    if match_count == total:
        print("  ✅ 完全匹配！Generate 和 Forward 的输出一致。")
    else:
        print("  ❌ 存在不匹配！")
        mismatch_indices = torch.where(~match)[0]
        print(f"  不匹配位置 (最多显示前10个): {mismatch_indices[:10].tolist()}")
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
            print(f"    位置 {idx}: generate={gen_tok}('{gen_text}') vs forward={pred_tok}('{pred_text}')")

            # 打印 top1/top2 logits 分析是否因为 logits 接近导致 argmax 随机性
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

            # 如果两侧都有 logits，额外打印对方 top1 token 在自己侧的 logit 值
            if gen_scores is not None and fwd_logits is not None:
                # generate 侧：forward 的 top1 token 在 generate logits 中的值
                g_logit_for_fwd_top1 = gen_scores[idx][pred_tok].item()
                # forward 侧：generate 的 top1 token 在 forward logits 中的值
                f_logit_for_gen_top1 = fwd_logits[idx][gen_tok].item()
                print(f"      [交叉对比] gen侧对fwd_top1('{pred_text}')的logit={g_logit_for_fwd_top1:.4f}, "
                      f"fwd侧对gen_top1('{gen_text}')的logit={f_logit_for_gen_top1:.4f}")

        if gen_scores is not None and fwd_logits is not None and tie_like_mismatches == mismatch_indices.numel():
            print(
                f"  ⚠️ 所有 {tie_like_mismatches} 个 token 不匹配都在 logit 容差 "
                f"{logit_tie_tolerance} 内，视为 bf16/不同 kernel 路径下的 argmax tie。"
            )
            return True

    return match_count == total


def compare_ppl(gen_ppl, gen_nll, fwd_ppl, fwd_nll, nll_tolerance):
    """
    对比 generate 和 forward 的 PPL。
    """
    print(f"\n  [PPL 对比]")
    print(f"    Generate PPL: {gen_ppl:.6f} (avg NLL: {gen_nll:.6f})")
    print(f"    Forward  PPL: {fwd_ppl:.6f} (avg NLL: {fwd_nll:.6f})")
    
    ppl_diff = abs(gen_ppl - fwd_ppl)
    nll_diff = abs(gen_nll - fwd_nll)
    # 使用相对误差判断，因为 bf16 精度下可能有微小差异
    rel_diff = ppl_diff / max(gen_ppl, fwd_ppl, 1e-8)
    
    print(f"    PPL 绝对差: {ppl_diff:.6f}, 相对差: {rel_diff:.6f}")
    print(f"    NLL 绝对差: {nll_diff:.6f}")
    
    # PPL is exponentially sensitive for very short generations (e.g. EOS-only),
    # so use average-NLL tolerance as the more stable bf16 diagnostic.
    ppl_threshold = 0.01
    if rel_diff < ppl_threshold:
        print(f"    ✅ PPL 一致！(相对差 {rel_diff:.6f} < 阈值 {ppl_threshold})")
        return True
    if nll_diff <= nll_tolerance:
        print(
            f"    ⚠️ PPL 相对差超过阈值，但 NLL 绝对差 {nll_diff:.6f} "
            f"<= {nll_tolerance}，视为短序列/bf16 数值误差。"
        )
        return True

    print(
        f"    ❌ PPL 不一致！相对差 {rel_diff:.6f} >= {ppl_threshold} "
        f"且 NLL 绝对差 {nll_diff:.6f} > {nll_tolerance}"
    )
    return False


def compare_prefill_next_logits(model, tokenizer, prompts, device, padding_side, tolerance):
    print("\n[Step 0] 对比 batched prefill next-token logits 与逐条 bsz=1 prefill...")
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
                f"  [Row {row}] ✅ prefill top1 一致: {batch_top}('{tokenizer.decode([batch_top])}'), "
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
                f"  [Row {row}] ⚠️ prefill top1 不同但属于近 tie: "
                f"batch={batch_top}('{tokenizer.decode([batch_top])}') vs "
                f"single={single_top}('{tokenizer.decode([single_top])}'), "
                f"gap_batch={batch_gap:.6f}, gap_single={single_gap:.6f}, "
                f"max_abs_diff={max_abs_diff:.6f}"
            )
        else:
            all_passed = False
            print(
                f"  [Row {row}] ❌ prefill top1 不一致: "
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
        print(f"  [Row {row}] ✅ batched 与 bsz=1 生成 token 完全一致。")
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
                f"  [Row {row}] ⚠️ 首个生成分叉是近 tie，后续自回归序列不再强行判错: "
                f"pos={idx}, batch={batch_tok}('{tokenizer.decode([batch_tok])}') vs "
                f"single={single_tok}('{tokenizer.decode([single_tok])}'), "
                f"gap_batch={batch_gap:.6f}, gap_single={single_gap:.6f}"
            )
            return True
        print(
            f"  [Row {row}] ❌ batched 与 bsz=1 首个非 tie 分叉: "
            f"pos={idx}, batch={batch_tok}('{tokenizer.decode([batch_tok])}') vs "
            f"single={single_tok}('{tokenizer.decode([single_tok])}'), "
            f"gap_batch={batch_gap:.6f}, gap_single={single_gap:.6f}"
        )
        return False

    print(
        f"  [Row {row}] ❌ batched 与 bsz=1 长度不一致: "
        f"batch_len={batch_trimmed.numel()}, single_len={single_trimmed.numel()}"
    )
    return False


def run_batched_consistency(model, tokenizer, prompts, device, args):
    all_passed = True
    model_core = getattr(model, "model", model)
    saved_debug_drop_attention_mask = getattr(model_core.config, "debug_drop_attention_mask", False)
    model_core.config.debug_drop_attention_mask = bool(args.drop_attention_mask)
    if args.drop_attention_mask:
        print("[Repro] 已开启 debug_drop_attention_mask：HSA forward 会忽略传入的 attention_mask。")

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

        print("\n[Step 1] 运行 batched model.generate() (greedy decode)...")
        input_ids, attention_mask, generated_ids, gen_scores = run_generate_batch(
            model,
            tokenizer,
            batch_prompts,
            device,
            args.max_new_tokens,
            args.padding_side,
            drop_attention_mask=args.drop_attention_mask,
        )
        print(f"  padded 输入长度: {input_ids.shape[1]}")
        print(f"  生成步数: {generated_ids.shape[1]}")
        for row in range(generated_ids.shape[0]):
            gen_text = tokenizer.decode(generated_ids[row], skip_special_tokens=True)
            real_prompt_len = int(attention_mask[row].sum().item())
            print(f"  [Row {row}] real_prompt_tokens={real_prompt_len}, 生成文本: {gen_text}")

        if args.compare_batch_single:
            print("\n[Step 1.5] 对比 batched generate 与逐条 bsz=1 generate...")
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

        print("\n[Step 2] 运行 batched model.forward() (全量 prefill 对照组)...")
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
                print(f"\n[Step 3] Row {row} / Prompt {global_idx}: 生成结果为空，跳过 logits/PPL 对比并判失败。")
                all_passed = False
                continue
            print(f"\n[Step 3] Row {row} / Prompt {global_idx}: 对比 generate vs forward argmax...")
            argmax_passed = compare_results(
                compare_generated_ids,
                pred_tokens[row][:compare_len],
                tokenizer,
                gen_scores=gen_scores[row][:compare_len],
                fwd_logits=fwd_logits[row][:compare_len],
                logit_tie_tolerance=args.logit_tie_tolerance,
            )

            print(f"\n[Step 4] Row {row} / Prompt {global_idx}: 对比 generate vs forward PPL...")
            gen_ppl, gen_nll = compute_ppl_from_logits(gen_scores[row][:compare_len], compare_generated_ids)
            fwd_ppl, fwd_nll = compute_ppl_from_logits(fwd_logits[row][:compare_len], compare_generated_ids)
            ppl_passed = compare_ppl(gen_ppl, gen_nll, fwd_ppl, fwd_nll, args.nll_tolerance)

            if not argmax_passed or not ppl_passed:
                all_passed = False

    model_core.config.debug_drop_attention_mask = saved_debug_drop_attention_mask
    return all_passed


def main(args):
    device = torch.device(args.device)
    print(f"[INFO] 使用设备: {device}")

    # ---- 加载 tokenizer ----
    tokenizer_path = args.tokenizer_path or args.checkpoint_path
    assert tokenizer_path, "请提供 --tokenizer_path 或 --checkpoint_path（用于加载 tokenizer）"

    print(f"[INFO] 加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 加载模型 ----
    # 1) 仅 checkpoint：from_pretrained，config 来自 ckpt 目录
    # 2) 仅 config_path：from_config 随机初始化（同 eval/eval_ppl_hf.py）
    # 3) config_path + checkpoint：用 config_path 构图，权重从 checkpoint 加载（HF: from_pretrained(ckpt, config=...)）
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
    }
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl

    if args.checkpoint_path and args.config_path:
        print(f"[INFO] 加载模型: 按 config 构图 + 从 checkpoint 加载权重")
        print(f"       config_path: {args.config_path}")
        print(f"       checkpoint_path: {args.checkpoint_path}")
        config = AutoConfig.from_pretrained(args.config_path)
        config.auto_insert_lmk = args.auto_insert_lmk
        load_kwargs = {**model_kwargs, "config": config, "device_map": device}
        if args.auto_insert_lmk:
            load_kwargs["auto_insert_lmk"] = True
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, **load_kwargs)
    elif args.checkpoint_path:
        print(f"[INFO] 加载模型(from_pretrained): {args.checkpoint_path}")
        load_kwargs = {**model_kwargs, "device_map": device}
        if args.auto_insert_lmk:
            load_kwargs["auto_insert_lmk"] = True
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, **load_kwargs)
    else:
        assert args.config_path is not None, "未指定 --checkpoint_path 时必须提供 --config_path（from_config 随机初始化）"
        print(f"[INFO] 加载模型(from_config): {args.config_path}")
        config = AutoConfig.from_pretrained(args.config_path)
        config.auto_insert_lmk = args.auto_insert_lmk
        model = AutoModelForCausalLM.from_config(config, **model_kwargs).to(device)

    model.eval()
    print(f"[INFO] 模型加载完成，chunk_size={model.chunk_size}, lmk_id={model.lmk_id}")
    print(f"[INFO] auto_insert_lmk={model.auto_insert_lmk}")
    print(f"[INFO] 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # ---- 准备测试 prompts ----
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
            raise ValueError("batch_size > 1 至少需要 2 个 prompt；请通过 --prompts 传入多条。")
        all_passed = run_batched_consistency(model, tokenizer, prompts, device, args)
    else:
        for i, prompt in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"[Prompt {i+1}] {prompt}")
            print(f"{'='*60}")

            # ---- Step 1: Generate (greedy decode) ----
            print("\n[Step 1] 运行 model.generate() (greedy decode)...")
            input_ids, generated_ids, gen_scores = run_generate(model, tokenizer, prompt, device, args.max_new_tokens)
            gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"  输入 token 数: {input_ids.shape[1]}")
            print(f"  生成 token 数: {generated_ids.shape[0]}")
            print(f"  生成文本: {gen_text}")

            # 重置 generate 模式状态
            model._gen_state.reset()

            # ---- Step 2: Forward baseline (全量 prefill) ----
            print("\n[Step 2] 运行 model.forward() (全量 prefill 对照组)...")
            pred_tokens, fwd_logits = run_forward_baseline(model, tokenizer, input_ids, generated_ids, device)

            # ---- Step 3: 对比 argmax ----
            print("\n[Step 3] 对比 generate vs forward argmax 结果...")
            argmax_passed = compare_results(generated_ids, pred_tokens, tokenizer, gen_scores=gen_scores, fwd_logits=fwd_logits)

            # ---- Step 4: 对比 PPL ----
            print("\n[Step 4] 对比 generate vs forward PPL...")
            # Generate 侧 PPL：用 generate 每步的 scores 和实际生成的 token 计算
            gen_ppl, gen_nll = compute_ppl_from_logits(gen_scores, generated_ids)
            # Forward 侧 PPL：用全量 forward 的 logits 和实际生成的 token 计算
            fwd_ppl, fwd_nll = compute_ppl_from_logits(fwd_logits, generated_ids)
            ppl_passed = compare_ppl(gen_ppl, gen_nll, fwd_ppl, fwd_nll, args.nll_tolerance)

            if not argmax_passed or not ppl_passed:
                all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("[INFO] ✅ 所有测试通过！Generate 和 Forward 输出完全一致。")
    else:
        print("[INFO] ❌ 部分测试未通过，请检查上面的不匹配详情。")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 HiLSForCausalLM generate vs forward 一致性")
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="HF 权重目录；可与 --config_path 联用（先按 config 构图再加载此处权重）；若二者皆未指定则使用内置默认",
    )
    parser.add_argument(
        "--config_path", type=str, default=None,
        help=(
            "config 目录或 config.json。"
            "仅指定此项：from_config 随机初始化；"
            "与 --checkpoint_path 同时指定：以此 config 构图，权重从 checkpoint 加载"
        ),
    )
    parser.add_argument(
        "--auto_insert_lmk", action="store_true",
        help="与 eval/eval_ppl_hf.py 一致：from_pretrained 时传入构造参数；from_config 时写入 config.auto_insert_lmk",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None,
        help="Tokenizer 路径；默认与 checkpoint_path 相同；无 checkpoint 时须单独指定",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="运行设备，如 cuda:0 或 cpu"
    )
    parser.add_argument(
        "--attn_impl", type=str, default="flash_attention_3",
        help="注意力实现方式，如 flash_attention_3, sdpa, eager 等"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=100,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--prompts", nargs="+", type=str, default=None,
        help="自定义 prompt 列表"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="训推一致测试的 batch size；>1 时会走 batched generate + batched forward 对照",
    )
    parser.add_argument(
        "--padding_side", choices=["left", "right"], default="left",
        help="batch_size > 1 时 tokenizer padding 方向；decoder-only generate 通常使用 left",
    )
    parser.add_argument(
        "--compare_batch_single", action="store_true",
        help="batch_size > 1 时额外逐条运行 bsz=1 generate，并与 batched generate 每行输出对齐",
    )
    parser.add_argument(
        "--drop_attention_mask", action="store_true",
        help="复现用：batch_size > 1 时让 HSA forward 在 decoder mask 构造前故意丢弃 attention_mask",
    )
    parser.add_argument(
        "--logit_tie_tolerance", type=float, default=0.5,
        help="batch/bsz=1 或 generate/forward top1 不同但互相 logit 差距低于该阈值时，视为 bf16 tie",
    )
    parser.add_argument(
        "--nll_tolerance", type=float, default=0.03,
        help="generate/forward PPL 相对差较大时，若平均 NLL 绝对差低于该阈值则视为短序列/bf16 数值误差",
    )

    args = parser.parse_args()
    if args.checkpoint_path is None and args.config_path is None:
        args.checkpoint_path = DEFAULT_CKPT_PATH

    print(f"[INFO] 参数: {args}")
    main(args)


# python unittests/test_generate_olmo.py
