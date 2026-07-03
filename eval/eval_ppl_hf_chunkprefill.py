import models
from data import build_numpy_dataset
import torch.nn as nn
import argparse
import sys
import os
import json
import glob
import math
import torch
import time
from transformers import AutoTokenizer
from torch.utils import data
from torch.utils.data import SequentialSampler, Subset

from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks


from transformers import AutoConfig, AutoModelForCausalLM
from models.FlashHiLS.configuration_hsa import HSAConfig
import json

class KahanSum:
    """Kahan 求和算法，减少浮点累加误差"""
    def __init__(self):
        self.sum = 0.0
        self.c = 0.0  # 误差补偿
        
    def add(self, value):
        y = value - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t
        
    def get(self):
        return self.sum
    
def resolve_hsa_class(config_path=None):
    """根据 config_path 中的 model_type 动态选择 HiLSForCausalLM 实现"""
    model_type = ""
    if config_path:
        with open(config_path, 'r') as f:
            model_type = json.load(f).get("model_type", "")
    if "olmo" in model_type:
        from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM
        print("Using OLMo LHSA implementation")
    else:
        from models.FlashHiLS.modeling_qwen_hils import HiLSForCausalLM
        print("Using Qwen LHSA implementation")
    return HiLSForCausalLM
def aggregate_results(result_dir):
    """读取所有 shard 结果文件并聚合计算最终 PPL"""
    pattern = os.path.join(result_dir, "shard_result_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[聚合] 在 {result_dir} 下未找到 shard_result_*.json 文件")
        return
    
    total_loss = 0.0
    total_count = 0
    for f in files:
        with open(f, 'r') as fp:
            d = json.load(fp)
        print(f"[聚合] 读取 {os.path.basename(f)}: shard_id={d['shard_id']}, count={d['count']}, loss_sum={d['loss_sum']:.6f}")
        total_loss += d['loss_sum']
        total_count += d['count']
    
    if total_count == 0:
        print("[聚合] 总样本数为0，无法计算PPL")
        return
    
    final_mean_loss = total_loss / total_count
    ppl = math.exp(final_mean_loss)
    print(f"[聚合] 总样本数: {total_count}, Final Mean Loss: {final_mean_loss:.4f}, PPL: {ppl:.4f}")


# def resolve_hsa_class(config_path=None):
#     """根据 config_path 中的 model_type 动态选择 HiLSForCausalLM 实现"""
#     model_type = ""
#     if config_path:
#         with open(config_path, 'r') as f:
#             model_type = json.load(f).get("model_type", "")
#     if "olmo" in model_type:
#         from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM
#         print("Using OLMo LHSA implementation")
#     else:
#         from models.FlashHiLS.modeling_qwen_hils import HiLSForCausalLM
#         print("Using Qwen LHSA implementation")
#     return HiLSForCausalLM


def main(args):

    device = torch.device('cuda:0')

    HiLSForCausalLM = resolve_hsa_class(args.config_path)
    AutoConfig.register("flash_hsa", HSAConfig)
    HiLSForCausalLM.config_class = HSAConfig
    AutoModelForCausalLM.register(HSAConfig, HiLSForCausalLM)

    # 分片逻辑：计算当前 shard 负责的样本范围
    use_sharding = hasattr(args, 'shard_id') and args.shard_id is not None and args.num_shards is not None and args.num_shards > 1
    if use_sharding:
        total_samples = args.max_samples
        shard_size = total_samples // args.num_shards
        remainder = total_samples % args.num_shards
        if args.shard_id < remainder:
            shard_start = args.shard_id * (shard_size + 1)
            shard_end = shard_start + shard_size + 1
        else:
            shard_start = remainder * (shard_size + 1) + (args.shard_id - remainder) * shard_size
            shard_end = shard_start + shard_size
        print(f'[Shard {args.shard_id}/{args.num_shards}] 负责样本范围: [{shard_start}, {shard_end}), 共 {shard_end - shard_start} 个样本')

    dataset = build_numpy_dataset(args.data_path, args.max_seq_len, namespace='test')
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    # 使用 Subset 直接切片，避免遍历跳过前面样本的巨大 I/O 开销
    if use_sharding:
        shard_indices = list(range(shard_start, shard_end))
        shard_dataset = Subset(dataset, shard_indices)
        print(f'[Shard {args.shard_id}/{args.num_shards}] Subset 已创建，共 {len(shard_dataset)} 个样本')
    else:
        shard_dataset = dataset

    def vanilla_collate_fn(examples):
        return {
            'input_ids': torch.tensor(examples),
            'labels': torch.tensor(examples)
        }

    dataloader = data.DataLoader(
        shard_dataset,
        batch_size=1,
        collate_fn=vanilla_collate_fn,
        sampler=SequentialSampler(shard_dataset),
        num_workers=1
    )

    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'attn_implementation': 'flash_attention_3',
        'device_map': device,
    }

    if args.checkpoint_path:
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, **model_kwargs)
    else:
        assert args.config_path is not None, "必须提供 --config_path 或 --checkpoint_path"
        config = AutoConfig.from_pretrained(args.config_path)
        model = AutoModelForCausalLM.from_config(config, **model_kwargs).to(device)

    model.eval()
    if args.insert_lmk:
        chunk_size = model.config.chunk_size
        lmk_id = tokenizer.vocab_size
    else:
        chunk_size = None
        lmk_id = None
    use_chunk_prefill = args.segment_size > 0
    segment_size = args.segment_size if use_chunk_prefill else args.max_seq_len

    if use_chunk_prefill and args.last_k_tokens > 0:
        assert segment_size >= args.last_k_tokens

    print(f"\n{'='*60}")
    print(f"Max Seq Len: {args.max_seq_len}, Segment Size: {segment_size}")
    print(f"Insert LMK: {args.insert_lmk}, Adjust LMK Pos: {args.adjust_lmk_pos}")
    print(f"Chunk Prefill: {'Enabled' if use_chunk_prefill else 'Disabled (Full Inference)'}")
    print(f"Skip HSA Prefill: {getattr(args, 'skip_hsa_prefill', False)}")
    print(f"{'='*60}\n")

    loss_accum = KahanSum()
    steps = 0
    ce_fct = nn.CrossEntropyLoss()
    eval_start_time = time.time()
    for inputs in dataloader:
        steps += 1
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        input_ids = inputs['input_ids']
        label_ids = input_ids.clone()
        pos_ids = None

        if args.insert_lmk:
            orig_seq_len = input_ids.shape[1]
            input_ids = insert_special_tokens(input_ids, fill_id=lmk_id, chunk_size=chunk_size)
            label_ids = torch.roll(label_ids, shifts=-1, dims=-1)
            label_ids[:, -1] = -100
            label_ids = insert_special_tokens(label_ids, fill_id=-100, chunk_size=chunk_size)
            label_ids = torch.roll(label_ids, shifts=1, dims=-1)

            if args.adjust_lmk_pos:
                pos_ids = create_position_ids_with_landmarks(orig_seq_len, chunk_size=chunk_size, device=device)

        seq_len = input_ids.shape[1]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            if not use_chunk_prefill:
                # ==================== 全量推理模式 ====================
                kwargs = {}
                if args.last_k_tokens > 0:
                    kwargs['logits_to_keep'] = args.last_k_tokens + 1

                result = model(input_ids, position_ids=pos_ids, use_cache=True, **kwargs)

                out_len = result.logits.shape[1]
                if args.last_k_tokens > 0:
                    out_len = min(out_len, args.last_k_tokens + 1)

                if args.insert_lmk:
                    # 取最后 out_len 的 logits 和 labels，过滤掉 lmk 位置
                    answer_logits = result.logits[:, -out_len:-1, :]
                    answer_labels = label_ids[:, -out_len + 1:]
                    valid_mask = (answer_labels != -100).squeeze(0)
                    loss = ce_fct(
                        answer_logits[0, valid_mask, :],
                        answer_labels[0, valid_mask].to(torch.long)
                    )
                else:
                    loss = ce_fct(
                        result.logits[:, -out_len:-1, :].view(-1, result.logits.shape[-1]),
                        label_ids[:, -out_len + 1:].view(-1).to(torch.long)
                    )
                del result
            else:
                # ==================== Chunk Prefill 模式 ====================
                num_segments = (seq_len + segment_size - 1) // segment_size

                # 确定需要保留 logits 的范围
                if args.last_k_tokens > 0:
                    if args.insert_lmk:
                        orig_answer_start = orig_seq_len - args.last_k_tokens
                        answer_start_with_lmk = orig_answer_start + (orig_answer_start // (chunk_size - 1))
                        answer_len_with_lmk = seq_len - answer_start_with_lmk
                    else:
                        answer_len_with_lmk = args.last_k_tokens
                else:
                    answer_len_with_lmk = seq_len

                answer_logits_start = max(0, seq_len - answer_len_with_lmk - 1)
                first_answer_segment = max(0, answer_logits_start // segment_size)

                past_key_values = None
                answer_logits_list = []

                for i in range(num_segments):
                    start_idx = i * segment_size
                    end_idx = min((i + 1) * segment_size, seq_len)

                    seg_input_ids = input_ids[:, start_idx:end_idx]
                    seg_cache_pos = torch.arange(start_idx, end_idx, device=device)
                    seg_pos_ids = pos_ids[:, start_idx:end_idx] if pos_ids is not None else None

                    if i >= first_answer_segment:
                        seg_logits_to_keep = end_idx - start_idx
                    else:
                        seg_logits_to_keep = 1

                    # 非 answer segment 时跳过上半 HSA 层以加速 prefill
                    # 让 first_answer_segment 及其前一个 segment 都跑全部层，
                    # 使 HSA 层在进入 answer 区域前已有一个 segment 的 KV cache
                    extra_kwargs = {}
                    if getattr(args, 'skip_hsa_prefill', False) and i < first_answer_segment - 1:
                        extra_kwargs["skip_hsa"] = True

                    out = model(
                        input_ids=seg_input_ids,
                        position_ids=seg_pos_ids,
                        cache_position=seg_cache_pos,
                        use_cache=True,
                        past_key_values=past_key_values,
                        logits_to_keep=seg_logits_to_keep,
                        **extra_kwargs,
                    )
                    past_key_values = out.past_key_values

                    if i >= first_answer_segment:
                        answer_logits_list.append(out.logits.cpu())

                    del out

                # 拼接 answer 相关的 logits（在 CPU 上拼接以节省显存）
                answer_region_logits = torch.cat(answer_logits_list, dim=1)
                del answer_logits_list
                torch.cuda.empty_cache()

                # 提取真正需要的 logits
                offset_in_region = answer_logits_start - first_answer_segment * segment_size
                answer_logits = answer_region_logits[:, offset_in_region:offset_in_region + answer_len_with_lmk, :]
                del answer_region_logits

                answer_logits = answer_logits.to(device)
                answer_labels = label_ids[:, -answer_len_with_lmk:]

                if args.insert_lmk:
                    if args.last_k_tokens > 0:
                        answer_logits = answer_logits[:, -args.last_k_tokens:, :]
                        answer_labels = answer_labels[:, -args.last_k_tokens:]
                    valid_mask = (answer_labels != -100).squeeze(0)
                    loss = ce_fct(
                        answer_logits[0, valid_mask, :],
                        answer_labels[0, valid_mask].to(torch.long)
                    )
                else:
                    if args.last_k_tokens > 0:
                        answer_logits = answer_logits[:, -args.last_k_tokens:, :]
                        answer_labels = answer_labels[:, -args.last_k_tokens:]
                    loss = ce_fct(
                        answer_logits.reshape(-1, answer_logits.shape[-1]),
                        answer_labels.reshape(-1).to(torch.long)
                    )

        loss_accum.add(loss.item())
        if steps % 10 == 0:
            print(f'step: {steps}, mean_loss: {loss_accum.get() / steps}')

        if args.max_samples > 0 and steps >= args.max_samples:
            break

    # 最终结果
    final_mean_loss = loss_accum.get() / steps if steps > 0 else 0
    ppl = math.exp(final_mean_loss)
    print(f'Test Length: {args.max_seq_len}, Final Mean Loss: {final_mean_loss:.4f}, PPL: {ppl:.4f}')

    eval_elapsed = time.time() - eval_start_time
    print(f'Total eval time: {eval_elapsed:.2f}s ({eval_elapsed/60:.2f}min), {steps} samples, {eval_elapsed/steps:.2f}s/sample')

    # 分片模式下将结果写入文件供聚合
    if use_sharding and args.result_dir:
        result_data = {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "shard_start": shard_start,
            "shard_end": shard_end,
            "count": steps,
            "loss_sum": loss_accum.get(),
        }
        os.makedirs(args.result_dir, exist_ok=True)
        result_file = os.path.join(args.result_dir, f"shard_result_{args.shard_id}.json")
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f'[Shard {args.shard_id}] 完成! 共 {steps} 个样本, loss_sum={loss_accum.get():.6f}, mean_loss={final_mean_loss:.4f}')
        print(f'[Shard {args.shard_id}] 结果已写入: {result_file}')


if __name__ == "__main__":
    cmd = argparse.ArgumentParser('Chunk Prefill PPL Test')
    cmd.add_argument('--config_path', required=False, type=str, default=None)
    cmd.add_argument('--vocab_dir', required=True, type=str)
    cmd.add_argument('--data_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--max_seq_len', default=16384, type=int)
    cmd.add_argument('--chunk_size', default=64, type=int)
    cmd.add_argument('--checkpoint_path', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--insert_lmk', action='store_true', help='在外部对数据插入 LMK token')
    cmd.add_argument('--adjust_lmk_pos', action='store_true', help='调整 LMK 位置的 position ids')
    cmd.add_argument('--last_k_tokens', type=int, default=-1, help='只用最后 k 个 token 计算 loss')
    cmd.add_argument('--segment_size', type=int, default=-1, help='Chunk prefill 的分段大小，<=0 表示全量推理')
    cmd.add_argument('--max_samples', default=-1, type=int, help='max samples to eval')
    cmd.add_argument('--skip_hsa_prefill', action='store_true', help='非answer segment跳过HSA层以加速prefill')
    # 分片参数（用于多卡并行测试同一个 ckpt）
    cmd.add_argument('--shard_id', type=int, default=None, help='当前进程的分片ID (从0开始)')
    cmd.add_argument('--num_shards', type=int, default=None, help='总分片数 (等于GPU数量)')
    cmd.add_argument('--result_dir', type=str, default=None, help='存放各 shard 结果文件的目录')
    cmd.add_argument('--aggregate', action='store_true', help='聚合所有 shard 结果并计算最终 PPL')
    
    args = cmd.parse_args(sys.argv[1:])
    print(args)

    if args.aggregate:
        aggregate_results(args.result_dir)
    else:
        main(args)

