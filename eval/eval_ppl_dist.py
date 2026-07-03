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
from transformers import AutoTokenizer
from torch.utils import data
from torch.utils.data import SequentialSampler, Subset
from veomni.models import build_foundation_model
from veomni.checkpoint import ckpt_to_state_dict, build_checkpointer
from utils.misc import get_model_fingerprint
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks

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


def main(args):
    device = torch.device('cuda:0')

    # 打印max_seq_len
    print(f'[Shard {args.shard_id}/{args.num_shards}] Max Sequence Length for Evaluation: {args.max_seq_len}')

    # 计算当前 shard 负责的样本范围（提前计算，用于构造 Subset）
    total_samples = args.max_samples
    shard_size = total_samples // args.num_shards
    remainder = total_samples % args.num_shards
    # 前 remainder 个 shard 各多分 1 个样本
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
    shard_indices = list(range(shard_start, shard_end))
    shard_dataset = Subset(dataset, shard_indices)
    print(f'[Shard {args.shard_id}/{args.num_shards}] Subset 已创建，共 {len(shard_dataset)} 个样本，无需跳过')

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

    Checkpointer = build_checkpointer(dist_backend='fsdp2', ckpt_manager='dcp')
    model = build_foundation_model(
        config_path=args.config_path,
        torch_dtype="bfloat16",
    )

    state = {"model": model}
    Checkpointer.load(args.checkpoint_path, state)

    model.eval()

    loss_accum = KahanSum()
    shard_count = 0

    for inputs in dataloader:
        shard_count += 1

        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        input_ids = inputs['input_ids']
        label_ids = input_ids
        pos_ids = None
        if args.insert_lmk:
            input_ids = insert_special_tokens(input_ids, fill_id=tokenizer.vocab_size, chunk_size=64)
            label_ids = torch.roll(label_ids, shifts=-1, dims=-1)
            label_ids[:, -1] = -100
            label_ids = insert_special_tokens(label_ids, fill_id=-100, chunk_size=64)
            label_ids = torch.roll(label_ids, shifts=1, dims=-1)

            if args.adjust_lmk_pos:
                pos_ids = create_position_ids_with_landmarks(None, args.max_seq_len, chunk_size=args.chunk_size, device=device)

        # automatic logits_to_keep
        kwargs = {}
        if args.last_k_tokens > 0:
            kwargs['logits_to_keep'] = args.last_k_tokens + 1

        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            result = model(input_ids, position_ids=pos_ids, use_cache=False, **kwargs)

        ce_fct = nn.CrossEntropyLoss()
        out_len = result.logits.shape[1]
        
        if args.last_k_tokens > 0:
            out_len = min(out_len, args.last_k_tokens + 1)
            
        loss = ce_fct(result.logits[:, -out_len :-1, :].view(-1, result.logits.shape[-1]), label_ids[:, -out_len + 1:].view(-1).to(torch.long))

        loss_accum.add(loss.item())
        if shard_count % 100 == 0:
            print(f'[Shard {args.shard_id}] step: {shard_count}, mean_loss: {loss_accum.get() / shard_count}')

    # 将结果写入文件
    result_data = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "shard_start": shard_start,
        "shard_end": shard_end,
        "count": shard_count,
        "loss_sum": loss_accum.get(),
    }

    os.makedirs(args.result_dir, exist_ok=True)
    result_file = os.path.join(args.result_dir, f"shard_result_{args.shard_id}.json")
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    mean_loss = loss_accum.get() / shard_count if shard_count > 0 else 0
    print(f'[Shard {args.shard_id}] 完成! 共 {shard_count} 个样本, loss_sum={loss_accum.get():.6f}, mean_loss={mean_loss:.4f}')
    print(f'[Shard {args.shard_id}] 结果已写入: {result_file}')


if __name__ == "__main__":
    cmd = argparse.ArgumentParser('Distributed PPL Evaluation (multi-process, per-GPU shard)')
    cmd.add_argument('--config_path', required=False, type=str, default=None)
    cmd.add_argument('--vocab_dir', required=True, type=str)
    cmd.add_argument('--data_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--max_seq_len', default=16384, type=int)
    cmd.add_argument('--chunk_size', default=64, type=int)
    cmd.add_argument('--insert_lmk', action='store_true')
    cmd.add_argument('--checkpoint_path', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--use_cache', action='store_true')
    cmd.add_argument('--last_k_tokens', type=int, default=-1)
    cmd.add_argument('--inference_segment', type=int, default=-1)
    cmd.add_argument('--max_samples', default=-1, type=int, help='max samples to eval')
    cmd.add_argument('--parallel_mode', default='fsdp1', type=str)
    cmd.add_argument('--adjust_lmk_pos', action='store_true')
    # 新增的分片参数
    cmd.add_argument('--shard_id', type=int, required=True, help='当前进程的分片ID (从0开始)')
    cmd.add_argument('--num_shards', type=int, required=True, help='总分片数 (等于GPU数量)')
    cmd.add_argument('--result_dir', type=str, required=True, help='存放各 shard 结果文件的目录')
    # 聚合模式
    cmd.add_argument('--aggregate', action='store_true', help='聚合所有 shard 结果并计算最终 PPL')

    args = cmd.parse_args(sys.argv[1:])
    print(args)

    if args.aggregate:
        aggregate_results(args.result_dir)
    else:
        main(args)
