import models
from data import build_numpy_dataset
import torch.nn as nn
import argparse
import sys
import torch
from transformers import AutoTokenizer
from torch.utils import data
from torch.utils.data import SequentialSampler

from utils.misc import get_model_fingerprint
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks


from transformers import AutoConfig, AutoModelForCausalLM
from models.FlashHiLS.configuration_hsa import HSAConfig
# from models.FlashHiLS.modeling_qwen_hils import HiLSForCausalLM
from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM

AutoConfig.register("flash_hsa", HSAConfig)
HiLSForCausalLM.config_class = HSAConfig
AutoModelForCausalLM.register(HSAConfig, HiLSForCausalLM)

import json
import os
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


def main(args):

    # create dataloader 
    # tokenizer = build_tokenizer(args.vocab_dir)
    device = torch.device('cuda:0')

    dataset = build_numpy_dataset(args.data_path, args.max_seq_len, namespace='test')
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    def vanilla_collate_fn(examples):
        return {
            'input_ids': torch.tensor(examples),
            'labels': torch.tensor(examples)
        }

    dataloader = data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=vanilla_collate_fn,
        sampler=SequentialSampler(dataset),
        num_workers=1
    )


    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'attn_implementation': 'flash_attention_3',
    }

    if args.checkpoint_path:
        if args.auto_insert_lmk:
            model_kwargs['auto_insert_lmk'] = True
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, **model_kwargs).to(device)
    else:
        assert args.config_path is not None, "必须提供 --config_path 或 --checkpoint_path"
        config = AutoConfig.from_pretrained(args.config_path)
        config.auto_insert_lmk = args.auto_insert_lmk
        model = AutoModelForCausalLM.from_config(config, **model_kwargs).to(device)


    model.eval()

    loss_accum = KahanSum()
    steps = 0
    for inputs in dataloader:
        steps += 1
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
                pos_ids = create_position_ids_with_landmarks(args.max_seq_len, chunk_size=args.chunk_size, device=device)
            

        kwargs = {}
        if args.last_k_tokens > 0:
            kwargs['logits_to_keep'] = args.last_k_tokens + 1

        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            result = model(input_ids, position_ids=pos_ids, use_cache=True, **kwargs)

        ce_fct = nn.CrossEntropyLoss()
        out_len = result.logits.shape[1]
        
        if args.last_k_tokens > 0:
            out_len = min(out_len, args.last_k_tokens + 1)
            
        loss = ce_fct(result.logits[:, -out_len :-1, :].view(-1, result.logits.shape[-1]), label_ids[:, -out_len + 1:].view(-1).to(torch.long))
        

        # mean_loss += (loss - mean_loss) / steps
        loss_accum.add(loss.item())
        if steps % 10 == 0:
            print(f'step: {steps}, mean_loss: {loss_accum.get() / steps}')

        if args.max_samples > 0 and steps >= args.max_samples:
            break
    # final ppl
    import math
    final_mean_loss = loss_accum.get() / steps
    ppl = math.exp(final_mean_loss)
    print(f'Test Length: {args.max_seq_len}, Final Mean Loss: {final_mean_loss:.4f}, PPL: {ppl:.4f}')


if __name__ == "__main__":
    cmd = argparse.ArgumentParser('NCR pretraining setup')
    cmd.add_argument('--config_path', required=False, type=str, default=None)
    cmd.add_argument('--vocab_dir', required=True, type=str)
    cmd.add_argument('--data_path', required=True, type=str, help='path to the training corpus')
    # cmd.add_argument('--output_dir', default='/root/')
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
    cmd.add_argument('--auto_insert_lmk', action='store_true', help='让模型内部自动插入 LMK，不需要外部插入')
    args = cmd.parse_args(sys.argv[1:])
    print(args)
    main(args)

