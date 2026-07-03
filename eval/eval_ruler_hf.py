import os
import sys
import torch
import argparse
import random
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM
from torch.utils import data
from torch.utils.data import SequentialSampler
from data import build_numpy_dataset
from data import RulerSynthesizer
import time
import json
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# 误差工具
def get_abs_err(x, y):
    return (x - y).flatten().abs().max().item()

def get_err_ratio(x, y):
    err = (x - y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base

def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg


from models.FlashHiLS.configuration_hsa import HSAConfig
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks


def resolve_hsa_class(config_path=None, checkpoint_path=None):
    """根据 config 中的 model_type 动态选择 HiLSForCausalLM 实现"""
    model_type = ""
    # 优先从 config_path 读取，其次从 checkpoint_path 下的 config.json 读取
    path = config_path or (os.path.join(checkpoint_path, "config.json") if checkpoint_path else None)
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            model_type = json.load(f).get("model_type", "")
    if "olmo" in model_type:
        from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM
        print("Using OLMo LHSA implementation")
    else:
        from models.FlashHiLS.modeling_qwen_hils import HiLSForCausalLM
        print("Using Qwen LHSA implementation")
    return HiLSForCausalLM


def main(args):
    """
    RULER 任务评测主函数
    
    评测逻辑：
    1. 加载数据，通过 RulerSynthesizer 生成 (prompt + answer) 拼接后的 input_ids
    2. 对 input_ids 插入 lmk token，并处理对应的 label_ids
    3. 使用 chunk prefill 遍历整个序列，收集 answer 位置的 logits
    4. 对 logits 取 argmax 得到预测 token id，与 label 对比计算准确率
    
    支持两种模式：
    - tp_size=1: 单卡推理模式
    - tp_size>1: Tensor Parallel 多卡推理模式
    """
    
    # ==================== 模型加载（HF方式，和eval_ppl_hf一致） ====================
    HiLSForCausalLM = resolve_hsa_class(args.config_path, args.checkpoint_path)
    AutoConfig.register("olmo_lhsa", HSAConfig)
    HiLSForCausalLM.config_class = HSAConfig
    AutoModelForCausalLM.register(HSAConfig, HiLSForCausalLM)

    device = torch.device('cuda:0')

    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'attn_implementation': 'flash_attention_3',
        'device_map': device,
    }

    if args.checkpoint_path:
        if args.auto_insert_lmk:
            model_kwargs['auto_insert_lmk'] = True
        # 如果指定了 config_path，则用它覆盖 checkpoint 目录自带的 config
        if args.config_path:
            config = AutoConfig.from_pretrained(args.config_path)
            if args.auto_insert_lmk:
                config.auto_insert_lmk = True
            model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, config=config, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, **model_kwargs)


    model.eval()
    
    # ==================== 数据准备 ====================
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir, trust_remote_code=True)
    
    # 构建数据集
    dataset = build_numpy_dataset(args.corpus_path, args.max_seq_len, namespace='test')
    
    # 初始化 RulerSynthesizer
    task_kwargs = {}
    if args.needle_len > 0:
        task_kwargs['length'] = args.needle_len
    if args.total_var > 0:
        task_kwargs['total_var'] = args.total_var
    if args.num_queries > 0:
        task_kwargs['num_queries'] = args.num_queries
        
    ruler_synthesizer = RulerSynthesizer(tokenizer, task_id=args.task_id, **task_kwargs)
    
    # 构建 DataLoader
    dataloader = data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=ruler_synthesizer.single_token_eval_collate_fn,
        sampler=SequentialSampler(dataset),
        num_workers=4,           # 仍然是 4~8，不用太多
    )


    # 获取模型配置
    chunk_size = getattr(model.config, 'chunk_size', 64)
    lmk_id = tokenizer.vocab_size
    # segment_size = args.segment_size if args.segment_size > 0 else 4096
    # segment_size <= 0 表示不切分，全量推理
    use_chunk_prefill = args.segment_size > 0
    segment_size = args.segment_size if args.segment_size > 0 else args.max_seq_len
    task_names = {0: "Single NIAH", 1: "Multi Query", 2: "Variable Tracking", 3: "FWE"}
    task_name = task_names.get(args.task_id, f"Task {args.task_id}")

    print(f"\n{'='*60}")
    print(f"Task: {task_name}, Max Seq Len: {args.max_seq_len}")
    print(f"Segment Size: {segment_size}, Insert LMK: {args.insert_lmk}")
    print(f"Chunk Prefill: {'Enabled' if use_chunk_prefill else 'Disabled (Full Inference)'}")
    print(f"Skip HSA Prefill: {getattr(args, 'skip_hsa_prefill', False)}")
    print(f"{'='*60}\n")
    
    # 评测循环
    total_samples = 0
    total_correct_tokens = 0
    total_tokens = 0
    exact_match_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        batch_start_time = time.time()  # 记录整个 batch 开始时间
        if args.max_samples > 0 and batch_idx >= args.max_samples:
            break
        
        # input_ids: (1, L) = prompt + answer 拼接
        # labels: (1, answer_len) = 只有 answer 部分
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)  # (1, answer_len)
        answer_len = labels.shape[1]
        # 打印样本信息
        if args.verbose:
            prompt_text = tokenizer.decode(input_ids[0, :-answer_len].tolist(), skip_special_tokens=True)
            answer_text = tokenizer.decode(labels[0].tolist(), skip_special_tokens=True)
            print(f"Sample {batch_idx + 1}:")
            print(f"  Prompt: {prompt_text[:200]}")  # 只打印前200字符
            print(f"  Answer token ids: {labels[0].tolist()}")
            print(f"  Answer text: {answer_text}\n")
        if args.insert_lmk:
            pos_ids = None
            
            # 记录原始序列长度和 answer 起始位置
            orig_seq_len = input_ids.shape[1]
            orig_answer_start = orig_seq_len - answer_len  # answer 在原始序列中的起始位置
            
            label_ids = input_ids.clone()
            input_ids = insert_special_tokens(input_ids, fill_id=lmk_id, chunk_size=chunk_size)
            # 与 eval_ppl.py 保持一致的 label 构造方式
            label_ids = torch.roll(label_ids, shifts=-1, dims=-1)  # 先左移一位
            label_ids[:, -1] = -100                                 # 最后一位设为 -100
            label_ids = insert_special_tokens(label_ids, fill_id=-100, chunk_size=chunk_size)  # 插入 lmk 位置用 -100 填充
            label_ids = torch.roll(label_ids, shifts=1, dims=-1)   # 再右移一位

            if args.adjust_lmk_pos:
                pos_ids = create_position_ids_with_landmarks(None, orig_seq_len, chunk_size=chunk_size, device=device)
            
            # 计算 answer 起始位置在插入 lmk 后的新位置
            # 每 (chunk_size-1) 个 token 后会插入 1 个 lmk
            # 所以位置 i 之前插入了 i // (chunk_size - 1) 个 lmk
            answer_start_with_lmk = orig_answer_start + (orig_answer_start // (chunk_size - 1))
            new_seq_len = input_ids.shape[1]
            answer_len_with_lmk = new_seq_len - answer_start_with_lmk
        else:
            label_ids = input_ids.clone()
            answer_start_with_lmk = input_ids.shape[1] - answer_len
            answer_len_with_lmk = answer_len
        
        seq_len = input_ids.shape[1]
        num_segments = (seq_len + segment_size - 1) // segment_size
        # # [DEBUG] 打印输入序列末尾：答案前3个token + 答案
        # if batch_idx < 3:
        #     debug_start = max(0, seq_len - answer_len_with_lmk - 3)
        #     debug_ids = input_ids[0, debug_start:].tolist()
        #     print(f"\n[INPUT DEBUG] Sample {batch_idx + 1}: answer前3 + answer")
        #     for i, tid in enumerate(debug_ids):
        #         marker = " <-- answer start" if i == 3 else ""
        #         print(f"  {tid:>8} | {repr(tokenizer.decode([tid]))}{marker}")
        
        # 计算 answer 部分的起始位置（用于判断哪些 segment 需要保留 logits）
        # logits[i] 预测 position i+1，所以预测 answer 需要 logits[answer_start-1:answer_end-1]
        # 即需要保留从 position (answer_start_with_lmk - 1) 开始的 logits
        answer_logits_start = seq_len - answer_len_with_lmk - 1  # 需要的 logits 起始位置
        first_answer_segment = answer_logits_start // segment_size  # 第一个包含 answer logits 的 segment
        
        past_key_values = None
        answer_logits_cpu = None
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            if not use_chunk_prefill:
                # 全量推理模式：一次性处理整个序列
                cache_pos = torch.arange(0, seq_len, device=device)
                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                out = model(
                    input_ids=input_ids,
                    cache_position=cache_pos,
                    use_cache=False,
                    logits_to_keep=answer_len_with_lmk + 1,
                    position_ids=pos_ids if args.adjust_lmk_pos else None,
                )
                end_time.record()
                torch.cuda.synchronize()
                elapsed_ms = start_time.elapsed_time(end_time)
                if batch_idx==0:
                    print(f"[Full] seq_len={seq_len}, time={elapsed_ms:.2f}ms")
                # 只保留 answer 部分的 logits 并立即移到 CPU，释放 GPU 显存
                # logits[i] 预测 position i+1 的 token，所以取 -(answer_len_with_lmk+1):-1
                answer_logits_cpu = out.logits[:, :-1, :].cpu()
                del out
                torch.cuda.empty_cache()
            else:
                # 分段 prefill 模式：只收集覆盖 answer 部分的 segment 的 logits
                answer_logits_list = []
                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                
                for i in range(num_segments):
                    start_idx = i * segment_size
                    end_idx = min((i + 1) * segment_size, seq_len)
                    
                    seg_input_ids = input_ids[:, start_idx:end_idx]
                    seg_cache_pos = torch.arange(start_idx, end_idx, device=device)
                    
                    # 计算这个 segment 需要保留多少 logits
                    if i >= first_answer_segment:
                        # 需要 logits 的 segment
                        seg_logits_to_keep = end_idx - start_idx  # 保留全部
                    else:
                        # 不需要 logits 的 segment，只保留 1 个 token 的 logits 以最小化显存占用
                        seg_logits_to_keep = 1
                    
                    # 切片 position_ids 以匹配当前 segment
                    seg_pos_ids = pos_ids[:, start_idx:end_idx] if args.adjust_lmk_pos else None

                    # 非 answer segment 时跳过上半 HSA 层以加速 prefill
                    # 让 first_answer_segment 及其前一个 segment 都跑全部层，
                    # 使 HSA 层在进入 answer 区域前已有一个 segment 的 KV cache
                    extra_kwargs = {}
                    if getattr(args, 'skip_hsa_prefill', False) and i < first_answer_segment - 1:
                        extra_kwargs["skip_hsa"] = True

                    out = model(
                        input_ids=seg_input_ids,
                        cache_position=seg_cache_pos,
                        use_cache=True,
                        past_key_values=past_key_values,
                        logits_to_keep=seg_logits_to_keep,
                        position_ids=seg_pos_ids,
                        **extra_kwargs,
                    )
                    past_key_values = out.past_key_values
                    
                    # 只保留覆盖 answer 部分的 segment 的 logits
                    if i >= first_answer_segment:
                        answer_logits_list.append(out.logits.cpu())  # 立即移到 CPU
                    
                    del out
                
                end_time.record()
                torch.cuda.synchronize()
                elapsed_ms = start_time.elapsed_time(end_time)
                if batch_idx==0:
                    print(f"[ChunkPrefill] seq_len={seq_len}, segments={num_segments}, "
                          f"answer_segments={num_segments - first_answer_segment}, time={elapsed_ms:.2f}ms")
                
                # 拼接 answer 相关的 logits（已在 CPU 上）
                answer_region_logits = torch.cat(answer_logits_list, dim=1)  # 在 CPU 上拼接
                del answer_logits_list
                
                # 从 answer_region_logits 中提取真正的 answer logits
                # answer_region 起始于 first_answer_segment * segment_size
                # 需要的 logits 起始于 answer_logits_start
                offset_in_region = answer_logits_start - first_answer_segment * segment_size
                answer_logits_cpu = answer_region_logits[:, offset_in_region:offset_in_region + answer_len_with_lmk, :]
                del answer_region_logits
                torch.cuda.empty_cache()
        
        # 提取 answer 部分的 logits 和 labels
        answer_logits = answer_logits_cpu.to(device)
        answer_labels = label_ids[:, -answer_len_with_lmk:]  # (1, answer_len_with_lmk)
        del answer_logits_cpu
        
        pred_tokens = torch.argmax(answer_logits, dim=-1)  # (1, answer_len_with_lmk)
        
        if args.insert_lmk:
            # 过滤掉 label=-100 的位置（lmk 插入位置）
            valid_mask = (answer_labels != -100)
            # print(f"  valid_mask.sum()={valid_mask.sum().item()} (非 -100 的位置数)")
            valid_pred = pred_tokens[valid_mask][:-1]  # 用索引截去最后一个 token (EOS)
            valid_label = answer_labels[valid_mask][:-1]
            # print(f"  valid_pred.shape={valid_pred.shape}, valid_label.shape={valid_label.shape}")
        
        else:
            # 不插入 lmk 时，也用索引截去最后一个 token (EOS)
            valid_pred = pred_tokens.flatten()[:-1]
            valid_label = answer_labels.flatten()[:-1]
        
        # 计算准确率
        correct = (valid_pred == valid_label).sum().item()
        total = valid_label.numel()
        # # [DEBUG] 对比预测和真实答案
        # if batch_idx < 3:
        #     print(f"\n[PRED DEBUG] Sample {batch_idx + 1}:")
        #     print(f"  {'Pred':>8} | {'Label':>8} | Pred Text       | Label Text")
        #     for i, (p, l) in enumerate(zip(valid_pred.tolist(), valid_label.tolist())):
        #         match = "✓" if p == l else "✗"
        #         print(f"  {p:>8} | {l:>8} | {repr(tokenizer.decode([p])):15} | {repr(tokenizer.decode([l]))} {match}")
        total_correct_tokens += correct
        total_tokens += total
        total_samples += 1
        
        # 检查是否完全匹配
        if correct == total:
            exact_match_count += 1
        
        # 打印进度
        if (batch_idx + 1) % args.print_every == 0:
            token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0
            exact_match_rate = exact_match_count / total_samples
            print(f"[{batch_idx + 1}/{args.max_samples if args.max_samples > 0 else 'all'}] "
                  f"Token Acc: {token_acc:.4f}, Exact Match: {exact_match_rate:.4f}")
            
            if args.verbose:
                print(f"  Pred token ids:  {valid_pred.tolist()}")
                print(f"  Label token ids: {valid_label.tolist()}")
                # 逐个对比
                match_status = ['✓' if p == l else '✗' for p, l in zip(valid_pred.tolist(), valid_label.tolist())]
                print(f"  Match status:    {match_status}\n")
    
    # 最终结果
    final_token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0
    final_exact_match = exact_match_count / total_samples if total_samples > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Final Results for {task_name}:")
    print(f"  Total Samples: {total_samples}")
    print(f"  Token Accuracy: {final_token_acc:.4f}")
    print(f"  Exact Match Rate: {final_exact_match:.4f}")
    print(f"{'='*60}")

    if getattr(args, "summary_log", None):
        summary_dir = os.path.dirname(os.path.abspath(args.summary_log))
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_id": int(args.task_id),
            "task_name": task_name,
            "max_seq_len": int(args.max_seq_len),
            "segment_size": int(args.segment_size),
            "insert_lmk": bool(args.insert_lmk),
            "adjust_lmk_pos": bool(args.adjust_lmk_pos),
            "auto_insert_lmk": bool(args.auto_insert_lmk),
            "skip_hsa_prefill": bool(getattr(args, "skip_hsa_prefill", False)),
            "vocab_dir": args.vocab_dir,
            "corpus_path": args.corpus_path,
            "checkpoint_path": args.checkpoint_path,
            "config_path": args.config_path,
            "total_samples": int(total_samples),
            "total_tokens": int(total_tokens),
            "token_accuracy": float(final_token_acc),
            "exact_match_rate": float(final_exact_match),
        }
        with open(args.summary_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    cmd = argparse.ArgumentParser('RULER Evaluation')
    cmd.add_argument('--config_path', required=False, type=str, help='Path to model config')
    cmd.add_argument('--vocab_dir', required=True, type=str, help='Path to tokenizer vocab')
    cmd.add_argument('--corpus_path', required=True, type=str, help='Path to tokenized numpy corpus')
    cmd.add_argument('--checkpoint_path', required=False, type=str, default=None, help='Path to checkpoint')
    cmd.add_argument('--task_id', type=int, default=0, choices=[0, 1, 2, 3],
                     help='Task ID: 0=Single NIAH, 1=Multi Query, 2=Variable Tracking, 3=FWE')
    cmd.add_argument('--max_seq_len', type=int, default=8192, help='Max sequence length')
    cmd.add_argument('--segment_size', type=int, default=4096, help='Segment size for chunk prefill. Set to 0 or negative to disable chunk prefill (full inference)')
    cmd.add_argument('--insert_lmk', action='store_true', help='Insert landmark tokens for HSA model')
    cmd.add_argument('--max_samples', type=int, default=100, help='Max samples to evaluate')
    cmd.add_argument('--print_every', type=int, default=1, help='Print progress every N samples')
    cmd.add_argument('--verbose', action='store_true', help='Print prediction examples')
    cmd.add_argument('--needle_len', type=int, default=-1, help='Needle length for NIAH task')
    cmd.add_argument('--total_var', type=int, default=-1, help='Total variables for VT/MQ tasks')
    cmd.add_argument('--num_queries', type=int, default=-1, help='Number of queries for MQ task')
    cmd.add_argument('--tp_size', type=int, default=1, help='Tensor Parallel size (1=single GPU, >1=multi-GPU TP)')
    cmd.add_argument('--adjust_lmk_pos', action='store_true', help='Adjust position ids for landmarks')
    cmd.add_argument('--auto_insert_lmk', action='store_true', help='Let model automatically insert lmk tokens')
    cmd.add_argument('--skip_hsa_prefill', action='store_true', help='非answer segment跳过HSA层以加速prefill')
    cmd.add_argument('--summary_log', required=False, type=str, default=None, help='Append one-line JSON summary to this file')
    
    
    args = cmd.parse_args()
    main(args)

