import os
import sys
import torch
import argparse
from transformers import AutoTokenizer
import types  
import random
import numpy as np
from data import build_numpy_dataset
import torch.nn as nn
import argparse
import sys
import torch
from transformers import AutoTokenizer
from torch.utils import data
from torch.utils.data import SequentialSampler
from veomni.models import build_foundation_model
from veomni.checkpoint import ckpt_to_state_dict, build_checkpointer
from data import RulerSynthesizer, synthesize_ruler_example
import torch.distributed as dist
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model
from veomni.checkpoint import build_checkpointer
import time 
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

# # 设置路径
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)

# VeOmni Imports
from veomni.models import build_foundation_model
from veomni.checkpoint import build_checkpointer
from models.FlashHiLS.configuration_hsa import HSAConfig
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks


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
    
    # ==================== 模型加载（根据 tp_size 选择分支） ====================
    if args.tp_size > 1:
        # TP 模式：初始化分布式环境
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        # 初始化并行状态
        init_parallel_state(
            dp_size=1,
            tp_size=args.tp_size,
            cp_size=1,
            dp_mode='fsdp2',
        )
        
        # 构建模型
        model = build_foundation_model(
            config_path=args.config_path,
            torch_dtype="bfloat16",
            init_device="cuda",
        )
        
        # 用 TP 包装模型
        model = build_parallelize_model(
            model,
            init_device="cuda",
            dtype="bfloat16",
            weights_path=args.checkpoint_path,
            enable_full_shard=False,
            enable_mixed_precision=True,
            enable_gradient_checkpointing=False,
        )
        
        # 加载 checkpoint
        Checkpointer = build_checkpointer(dist_backend='fsdp2', ckpt_manager='dcp')
        Checkpointer.load(args.checkpoint_path, {"model": model})
        
        model.eval()
    else:
        # 单卡模式
        device = torch.device('cuda:0')
        
        # 构建模型
        model = build_foundation_model(config_path=args.config_path, torch_dtype="bfloat16")
        
        # 加载 checkpoint
        if args.checkpoint_path:
            Checkpointer = build_checkpointer(dist_backend='fsdp2', ckpt_manager='dcp')
            Checkpointer.load(args.checkpoint_path, {"model": model})
        
        model.to(device).eval()
    
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
        
    ruler_synthesizer = RulerSynthesizer(
        tokenizer,
        task_id=args.task_id,
        enable_ruler_plus=args.enable_ruler_plus or args.task_id in (4, 5),
        **task_kwargs,
    )
    
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
    task_names = {
        0: "Single NIAH",
        1: "Multi Query",
        2: "Variable Tracking",
        3: "FWE",
        4: "PMVL",
        5: "PCVL",
    }
    task_name = task_names.get(args.task_id, f"Task {args.task_id}")

    print(f"\n{'='*60}")
    print(f"Task: {task_name}, Max Seq Len: {args.max_seq_len}")
    print(f"Segment Size: {segment_size}, Insert LMK: {args.insert_lmk}")
    print(f"Chunk Prefill: {'Enabled' if use_chunk_prefill else 'Disabled (Full Inference)'}")
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
        original_input_ids = input_ids
        orig_seq_len = input_ids.shape[1]
        orig_answer_start = orig_seq_len - answer_len
        pos_ids = None

        if args.insert_lmk:
            input_ids = insert_special_tokens(input_ids, fill_id=lmk_id, chunk_size=chunk_size)
            if args.adjust_lmk_pos:
                pos_ids = create_position_ids_with_landmarks(None, orig_seq_len, chunk_size=chunk_size, device=device)
        
        seq_len = input_ids.shape[1]
        num_segments = (seq_len + segment_size - 1) // segment_size

        # Each answer token is supervised by the logits at its previous original position.
        # Map those original logit positions to positions after LMK insertion, then gather exactly those logits.
        orig_answer_token_pos = torch.arange(
            orig_answer_start,
            orig_answer_start + answer_len,
            device=device,
        )
        orig_logit_pos = orig_answer_token_pos - 1
        if torch.any(orig_logit_pos < 0):
            raise ValueError(f"Invalid answer start: orig_answer_start={orig_answer_start}")
        if args.insert_lmk:
            answer_logit_pos = orig_logit_pos + (orig_logit_pos // (chunk_size - 1))
        else:
            answer_logit_pos = orig_logit_pos
        logits_start_pos = int(answer_logit_pos.min().item())
        logits_to_keep = seq_len - logits_start_pos
        first_answer_segment = logits_start_pos // segment_size
        
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
                    logits_to_keep=logits_to_keep,
                    position_ids=pos_ids,
                )
                end_time.record()
                torch.cuda.synchronize()
                elapsed_ms = start_time.elapsed_time(end_time)
                if batch_idx==0:
                    print(f"[Full] seq_len={seq_len}, time={elapsed_ms:.2f}ms")
                # 只保留 answer 部分对应位置的 logits 并立即移到 CPU，释放 GPU 显存
                offsets = answer_logit_pos - logits_start_pos
                answer_logits_cpu = out.logits[:, offsets, :].cpu()
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
                    seg_pos_ids = pos_ids[:, start_idx:end_idx] if pos_ids is not None else None
                    
                    out = model(
                        input_ids=seg_input_ids,
                        cache_position=seg_cache_pos,
                        use_cache=True,
                        past_key_values=past_key_values,
                        logits_to_keep=seg_logits_to_keep,
                        position_ids=seg_pos_ids,
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
                answer_region_start = first_answer_segment * segment_size
                offsets = (answer_logit_pos - answer_region_start).cpu()
                answer_logits_cpu = answer_region_logits[:, offsets, :]
                del answer_region_logits
                torch.cuda.empty_cache()
        
        # 提取 answer 部分的 logits，并直接使用原始 labels 作为 ground truth
        answer_logits = answer_logits_cpu.to(device)
        answer_labels = labels
        del answer_logits_cpu
        
        pred_tokens = torch.argmax(answer_logits, dim=-1)  # (1, answer_len)
        if pred_tokens.shape != answer_labels.shape:
            raise RuntimeError(f"Shape mismatch: pred={pred_tokens.shape}, label={answer_labels.shape}")
        
        # 截去最后一个 token (EOS)
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

                try:
                    pred_text = tokenizer.decode(valid_pred.tolist(), skip_special_tokens=True)
                    label_text = tokenizer.decode(valid_label.tolist(), skip_special_tokens=True)
                    prompt_token_ids = original_input_ids[0, :-answer_len].tolist()
                    prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=True)

                    pred_text_clean = pred_text.strip().rstrip(".").strip()
                    label_text_clean = label_text.strip().rstrip(".").strip()

                    pred_in_prompt = pred_text_clean and pred_text_clean in prompt_text
                    label_in_prompt = label_text_clean and label_text_clean in prompt_text
                    pred_first_in_prompt = False
                    if pred_text_clean:
                        pred_first_token = pred_text_clean[: max(1, len(label_text_clean) // 2)] if label_text_clean else pred_text_clean
                        pred_first_in_prompt = pred_first_token in prompt_text

                    print(f"  Pred text:       {pred_text!r}")
                    print(f"  Label text:      {label_text!r}")
                    print(f"  Pred in prompt?  {bool(pred_in_prompt)} (full string)")
                    print(f"  Label in prompt? {bool(label_in_prompt)} (sanity check)")
                    print(f"  Pred prefix in prompt? {bool(pred_first_in_prompt)}\n")
                except Exception as exc:
                    print(f"  [debug decode error] {type(exc).__name__}: {exc}\n")
    
    # 最终结果
    final_token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0
    final_exact_match = exact_match_count / total_samples if total_samples > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Final Results for {task_name}:")
    print(f"  Total Samples: {total_samples}")
    print(f"  Token Accuracy: {final_token_acc:.4f}")
    print(f"  Exact Match Rate: {final_exact_match:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    cmd = argparse.ArgumentParser('RULER Evaluation')
    cmd.add_argument('--config_path', required=True, type=str, help='Path to model config')
    cmd.add_argument('--vocab_dir', required=True, type=str, help='Path to tokenizer vocab')
    cmd.add_argument('--corpus_path', required=True, type=str, help='Path to tokenized numpy corpus')
    cmd.add_argument('--checkpoint_path', required=False, type=str, default=None, help='Path to checkpoint')
    cmd.add_argument('--task_id', type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                     help='Task ID: 0=Single NIAH, 1=Multi Query, 2=Variable Tracking, 3=FWE, 4=PMVL, 5=PCVL')
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
    cmd.add_argument('--enable_ruler_plus', action='store_true', help='Enable Ruler-Plus tasks')
    
    
    args = cmd.parse_args()
    main(args)
