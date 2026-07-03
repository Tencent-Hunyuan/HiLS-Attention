"""
分析 OLMo3-HSA MHA 模式下 block_M 个 token 的 chunk 索引重合率 (使用真实模型权重 + 真实数据)

指标 1 — Overlap Ratio (越大越好, 范围 [0, 1]):
    (block_M * topK  -  union_size) / ((block_M - 1) * topK)
    全部重合 → union=topK → overlap=1; 完全不重合 → union=M*topK → overlap=0。

指标 2 — Coverage Ratio (越小越好):
    实际 unique chunk 数量 / prefix 可见 chunk 总量
    衡量实际访问的 chunk 占所有可见 chunk 的比例；越小说明稀疏性越好。

用法:
    python code_exp/test_block_m_overlap.py --mode all
    python code_exp/test_block_m_overlap.py --mode sweep
    python code_exp/test_block_m_overlap.py --mode per_head
    python code_exp/test_block_m_overlap.py --mode per_layer
    python code_exp/test_block_m_overlap.py --max_seq_len 16384 --max_samples 20
"""

import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_ROOT = os.path.join(ROOT, "eval")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

import argparse
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from einops import rearrange
from transformers import AutoConfig, AutoModelForCausalLM

# ──────────────────────────────────────────────────────────────
# 默认路径
# ──────────────────────────────────────────────────────────────
DEFAULT_CONFIG = os.path.join(ROOT, "configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified.json")
DEFAULT_CKPT   = "/Models/lhsa-olmo3-interleave-8KA512-non-unified-64gpu/global_step_10000/hf_ckpt"
DEFAULT_DATA   = "/data/dolma3_mix-6T-1025-partial-tokenized"
DEFAULT_VOCAB  = os.path.join(ROOT, "configs/olmo3_vocab")


# ──────────────────────────────────────────────────────────────
# 模型加载 & 数据准备
# ──────────────────────────────────────────────────────────────

def _register_hsa_classes(checkpoint_path):
    """
    像 eval_opencompass.py 那样注册 HSA config/model class，
    这样 AutoModelForCausalLM.from_pretrained 就能直接加载。
    返回 (model_type, ConfigClass)。
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from models.FlashHiLS.configuration_hsa import HSAConfig

    # 从 checkpoint 的 config.json 读 model_type
    config_json = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_json):
        raise FileNotFoundError(f"No config.json in {checkpoint_path}")

    with open(config_json) as f:
        model_type = json.load(f).get("model_type", "")

    if "olmo" in model_type:
        from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM
    elif "qwen" in model_type:
        from models.FlashHiLS.modeling_qwen_hils import HiLSForCausalLM
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    class _HSAConfig(HSAConfig):
        pass
    _HSAConfig.model_type = model_type

    AutoConfig.register(model_type, _HSAConfig, exist_ok=True)
    HiLSForCausalLM.config_class = _HSAConfig
    AutoModelForCausalLM.register(_HSAConfig, HiLSForCausalLM, exist_ok=True)

    return model_type, _HSAConfig


def _infer_hsa_params_from_ckpt(checkpoint_path, config):
    """
    从 checkpoint 的权重 shape 推断 config 中缺失的 HSA 参数:
      - hsa_heads / hsa_qk_ratio (从 q_proj shape)
    不推断 unified_retrieval / layerwise_qk_norm，这些以 config 为准。
    """
    idx_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        return

    with open(idx_path) as f:
        index = json.load(f)

    from safetensors import safe_open

    hidden = config.hidden_size

    # 找第一个 HSA 层 (full_attn_interleave 模式下最后一层)
    interleave = getattr(config, "full_attn_interleave", 4)
    first_hsa = interleave - 1  # e.g. layer 3

    # 推断 hsa_heads / hsa_qk_ratio
    q_key = f"model.layers.{first_hsa}.self_attn.q_proj.weight"
    if q_key in index["weight_map"]:
        shard_path = os.path.join(checkpoint_path, index["weight_map"][q_key])
        with safe_open(shard_path, framework="pt") as sf:
            q_out_dim = sf.get_tensor(q_key).shape[0]
        if q_out_dim == hidden:
            config.hsa_heads = config.num_attention_heads
            config.hsa_qk_ratio = 1
            print(f"  [Auto-detect] q_proj out={q_out_dim} == hidden={hidden} "
                  f"=> full-HSA: hsa_heads={config.hsa_heads}, hsa_qk_ratio=1")


def load_model(config_path, checkpoint_path, device="cuda"):
    """
    用 config_path 构建 config（满足新 modeling 要求），
    从 checkpoint 权重 shape 自动补齐缺失参数，再加载权重。
    """
    import models  # trigger model registration side-effects
    from transformers import AutoModelForCausalLM

    # 1. 注册 HSA classes
    model_type, ConfigClass = _register_hsa_classes(checkpoint_path)
    print(f"  [Registered] model_type={model_type}")

    # 2. 用用户指定的 config_path 构建 config（字段更完整）
    config = ConfigClass.from_pretrained(config_path)
    print(f"  [Config] from {config_path}")

    # 3. 从 checkpoint 权重 shape 推断并补齐缺失的 HSA 参数
    _infer_hsa_params_from_ckpt(checkpoint_path, config)

    # 4. 用该 config + checkpoint 权重加载模型
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(device).eval()
    return model


def load_data_batch(data_path, vocab_dir, max_seq_len, num_samples=1, device="cuda"):
    """加载评估数据，返回 input_ids"""
    from data import build_numpy_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, SequentialSampler

    dataset = build_numpy_dataset(data_path, max_seq_len, namespace='test')
    tokenizer = AutoTokenizer.from_pretrained(vocab_dir)

    def collate_fn(examples):
        return {'input_ids': torch.tensor(examples)}

    dataloader = DataLoader(
        dataset, batch_size=1, collate_fn=collate_fn,
        sampler=SequentialSampler(dataset), num_workers=0,
    )

    batches = []
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        batches.append(batch['input_ids'].to(device))

    return batches, tokenizer


def prepare_input_with_landmarks(input_ids, tokenizer, chunk_size, adjust_lmk_pos=True):
    """插入 landmark token 并调整 position ids"""
    from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks

    orig_seq_len = input_ids.shape[1]
    lmk_input_ids = insert_special_tokens(input_ids, fill_id=tokenizer.vocab_size, chunk_size=chunk_size)

    pos_ids = None
    if adjust_lmk_pos:
        pos_ids = create_position_ids_with_landmarks(
            None, orig_seq_len, chunk_size=chunk_size, device=input_ids.device
        )

    return lmk_input_ids, pos_ids


# ──────────────────────────────────────────────────────────────
# Hook: 收集每个 HSA 层的 topk indices
# ──────────────────────────────────────────────────────────────

class HSATopkCollector:
    """
    通过 hook 收集每个 HiLSAttention 层在 forward 中调用 topk_func 后的 indices。
    """
    def __init__(self, model):
        self.model = model
        self.collected = {}  # layer_idx -> indices tensor
        self._hooks = []
        self._install_hooks()

    def _install_hooks(self):
        """Monkey-patch topk_func in each HiLSAttention layer to capture indices."""
        from models.FlashHiLS.hils_attention import HiLSAttention

        # 遍历所有 decoder layer
        for layer in self.model.model.layers:
            attn = layer.self_attn
            if isinstance(attn, HiLSAttention):
                layer_idx = attn.layer_idx
                original_topk = attn.topk_func

                def make_wrapper(orig_fn, lid):
                    def wrapper(*args, **kwargs):
                        indices, scores = orig_fn(*args, **kwargs)
                        self.collected[lid] = {
                            'indices': indices.detach().cpu(),
                            'scores': scores.detach().cpu(),
                        }
                        return indices, scores
                    return wrapper

                attn.topk_func = make_wrapper(original_topk, layer_idx)

    def clear(self):
        self.collected.clear()

    def get_hsa_layer_indices(self):
        """返回 {layer_idx: indices_tensor}"""
        return {k: v['indices'] for k, v in self.collected.items()}


# ──────────────────────────────────────────────────────────────
# 分析核心
# ──────────────────────────────────────────────────────────────

def analyze_overlap(indices: torch.Tensor, block_M: int, topk: int,
                    chunk_size: int, hsa_sliding_window: int,
                    seq_len: int = None):
    """
    给定 indices (B, L, H, topk)，按 block_M 分组统计重合率。
    返回 dict 包含汇总统计 + 按位置分段统计。
    """
    B, L, H, K = indices.shape
    if seq_len is None:
        seq_len = L
    num_chunks = seq_len // chunk_size
    indices_cpu = indices.numpy() if isinstance(indices, torch.Tensor) else indices

    num_groups = L // block_M

    overlap_ratios = []
    coverage_ratios = []
    union_sizes = []
    total_valid_counts = []
    position_bins = defaultdict(lambda: {"overlap": [], "coverage": [], "union": []})

    for b in range(B):
        for g in range(num_groups):
            start_t = g * block_M
            end_t = start_t + block_M

            for h in range(H):
                all_selected = set()
                valid_count = 0

                for t in range(start_t, end_t):
                    for k in range(K):
                        idx = int(indices_cpu[b, t, h, k])
                        if idx >= 0:
                            all_selected.add(idx)
                            valid_count += 1

                if valid_count == 0:
                    continue

                union_size = len(all_selected)
                # 分母是 (M-1)*topK: 最大可能的冗余量
                # 全部重合 → union=topK → overlap=1; 完全不重合 → union=M*topK → overlap=0
                max_redundancy = valid_count - (valid_count // block_M)  # = (M-1)*topK when all valid
                overlap_ratio = (valid_count - union_size) / max_redundancy if max_redundancy > 0 else 0.0

                # prefix chunk count: 以组内最后一个 token 为基准
                last_t = end_t - 1
                visible_end = (last_t - hsa_sliding_window + 1) // chunk_size
                num_prefix = min(max(visible_end, 0), num_chunks)
                coverage_ratio = union_size / num_prefix if num_prefix > 0 else 0.0

                overlap_ratios.append(overlap_ratio)
                coverage_ratios.append(coverage_ratio)
                union_sizes.append(union_size)
                total_valid_counts.append(valid_count)

                # bin by position
                rel_pos = (start_t + end_t) / 2 / L
                if rel_pos < 0.33:
                    bin_name = "early (0-33%)"
                elif rel_pos < 0.66:
                    bin_name = "mid (33-66%)"
                else:
                    bin_name = "late (66-100%)"
                position_bins[bin_name]["overlap"].append(overlap_ratio)
                position_bins[bin_name]["coverage"].append(coverage_ratio)
                position_bins[bin_name]["union"].append(union_size)

    if not overlap_ratios:
        return {"num_groups": 0}

    results = {
        "overlap_ratio_mean": np.mean(overlap_ratios),
        "overlap_ratio_std":  np.std(overlap_ratios),
        "coverage_ratio_mean": np.mean(coverage_ratios),
        "coverage_ratio_std":  np.std(coverage_ratios),
        "union_size_mean": np.mean(union_sizes),
        "union_size_std":  np.std(union_sizes),
        "total_topk_mean": np.mean(total_valid_counts),
        "num_groups": len(overlap_ratios),
        "position_bins": {},
    }
    for bin_name, data in sorted(position_bins.items()):
        results["position_bins"][bin_name] = {
            "overlap_mean": np.mean(data["overlap"]),
            "coverage_mean": np.mean(data["coverage"]),
            "union_mean": np.mean(data["union"]),
            "count": len(data["overlap"]),
        }
    return results


def print_results(name, results, block_M, topk):
    max_total = block_M * topk
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    if results["num_groups"] == 0:
        print("  [No valid groups — skipped]")
        return

    print(f"\n  [Overall]  ({results['num_groups']} groups × heads)")
    print(f"    Overlap Ratio   : {results['overlap_ratio_mean']:.4f} +/- {results['overlap_ratio_std']:.4f}  (higher=better)")
    print(f"    Coverage Ratio  : {results['coverage_ratio_mean']:.4f} +/- {results['coverage_ratio_std']:.4f}  (lower=better)")
    print(f"    Union Size      : {results['union_size_mean']:.1f} +/- {results['union_size_std']:.1f}  (max={max_total}, ideal~{topk})")
    print(f"    Avg Valid TopK  : {results['total_topk_mean']:.1f} / {max_total}")

    print(f"\n  [By position]")
    for bin_name, data in results.get("position_bins", {}).items():
        print(f"    {bin_name:20s}  overlap={data['overlap_mean']:.4f}  "
              f"coverage={data['coverage_mean']:.4f}  "
              f"union={data['union_mean']:.1f}  (n={data['count']})")


# ──────────────────────────────────────────────────────────────
# 主分析函数
# ──────────────────────────────────────────────────────────────

def run_model_and_collect(args):
    """加载模型，跑 forward，收集 topk indices"""
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = load_model(args.config_path, args.checkpoint_path)

    # 从 checkpoint 自带的 config.json 读取参数（最权威）
    ckpt_config_path = os.path.join(args.checkpoint_path, "config.json")
    if os.path.exists(ckpt_config_path):
        with open(ckpt_config_path) as f:
            config = json.load(f)
    else:
        with open(args.config_path) as f:
            config = json.load(f)

    chunk_size = config.get("chunk_size", 64)
    hsa_topk = config.get("hsa_topk", 32)
    hsa_sliding_window = config.get("sliding_window", 4096)  # HSA 用的 sliding_window
    print(f"  chunk_size={chunk_size}, hsa_topk={hsa_topk}, hsa_sliding_window={hsa_sliding_window}")

    print(f"\nLoading data from: {args.data_path}")
    batches, tokenizer = load_data_batch(
        args.data_path, args.vocab_dir, args.max_seq_len, num_samples=args.max_samples,
    )
    print(f"  Loaded {len(batches)} samples, seq_len={batches[0].shape[1]}")

    collector = HSATopkCollector(model)

    # 收集所有 sample 的 indices
    all_indices_per_layer = defaultdict(list)  # layer_idx -> list of (B, L, H, topk)

    for i, input_ids in enumerate(batches):
        print(f"  Running sample {i+1}/{len(batches)}...", end=" ", flush=True)

        lmk_input_ids, pos_ids = prepare_input_with_landmarks(
            input_ids, tokenizer, chunk_size, adjust_lmk_pos=True
        )

        collector.clear()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            model(lmk_input_ids, position_ids=pos_ids, use_cache=False)

        layer_indices = collector.get_hsa_layer_indices()
        for lid, idx_tensor in layer_indices.items():
            all_indices_per_layer[lid].append(idx_tensor)
        print(f"collected {len(layer_indices)} HSA layers")

    # 合并: concat along batch dim
    merged = {}
    for lid in sorted(all_indices_per_layer.keys()):
        merged[lid] = torch.cat(all_indices_per_layer[lid], dim=0)  # (N_samples, L, H, topk)

    return merged, config


def main_basic(args):
    """基本分析: 每个 HSA 层的 overlap / coverage"""
    merged, config = run_model_and_collect(args)

    chunk_size = config.get("chunk_size", 64)
    hsa_topk = config.get("hsa_topk", 32)
    hsa_sliding_window = config.get("sliding_window", 4096)
    block_M = args.block_M

    print(f"\n\n{'#'*70}")
    print(f"# Basic Analysis: block_M={block_M}, topk={hsa_topk}")
    print(f"{'#'*70}")

    for lid in sorted(merged.keys()):
        indices = merged[lid]
        B, L, H, K = indices.shape
        results = analyze_overlap(
            indices, block_M, hsa_topk, chunk_size, hsa_sliding_window, seq_len=L * chunk_size
        )
        print_results(f"Layer {lid} (HSA)", results, block_M, hsa_topk)


def main_sweep(args):
    """Sweep block_M 值，观察重合率变化"""
    merged, config = run_model_and_collect(args)

    chunk_size = config.get("chunk_size", 64)
    hsa_topk = config.get("hsa_topk", 32)
    hsa_sliding_window = config.get("sliding_window", 4096)

    block_m_values = [1, 2, 4, 8, 16, 32, 64]

    print(f"\n\n{'#'*70}")
    print(f"# Block_M Sweep")
    print(f"{'#'*70}")

    for lid in sorted(merged.keys()):
        indices = merged[lid]
        B, L, H, K = indices.shape

        print(f"\n{'='*80}")
        print(f"  Layer {lid} — seq_len(with_lmk)={L}, topk={hsa_topk}")
        print(f"{'='*80}")
        print(f"\n{'block_M':>8s} | {'Overlap':>10s} | {'Coverage':>10s} | {'Union':>8s} | {'Max':>8s}")
        print(f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

        for bm in block_m_values:
            if bm > L:
                continue
            results = analyze_overlap(
                indices, bm, hsa_topk, chunk_size, hsa_sliding_window, seq_len=L * chunk_size
            )
            if results["num_groups"] == 0:
                continue
            max_possible = bm * hsa_topk
            print(f"{bm:>8d} | {results['overlap_ratio_mean']:>10.4f} | "
                  f"{results['coverage_ratio_mean']:>10.4f} | "
                  f"{results['union_size_mean']:>8.1f} | {max_possible:>8d}")


def main_per_head(args):
    """逐 head 分析重合率"""
    merged, config = run_model_and_collect(args)

    chunk_size = config.get("chunk_size", 64)
    hsa_topk = config.get("hsa_topk", 32)
    hsa_sliding_window = config.get("sliding_window", 4096)
    block_M = args.block_M

    print(f"\n\n{'#'*70}")
    print(f"# Per-Head Analysis: block_M={block_M}")
    print(f"{'#'*70}")

    for lid in sorted(merged.keys()):
        indices = merged[lid]  # (N, L, H, K)
        B, L, H, K = indices.shape
        indices_cpu = indices.numpy()
        num_groups = L // block_M

        print(f"\n{'='*80}")
        print(f"  Layer {lid} — H={H}")
        print(f"{'='*80}")
        print(f"\n{'Head':>6s} | {'Overlap':>10s} | {'Coverage':>10s} | {'Union':>8s}")
        print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

        num_chunks = (L * chunk_size) // chunk_size  # approx

        for h in range(H):
            olap_list, cov_list, union_list = [], [], []
            for b in range(B):
                for g in range(num_groups):
                    st = g * block_M
                    et = st + block_M
                    sel = set()
                    vc = 0
                    for t in range(st, et):
                        for k in range(K):
                            idx = int(indices_cpu[b, t, h, k])
                            if idx >= 0:
                                sel.add(idx)
                                vc += 1
                    if vc == 0:
                        continue
                    us = len(sel)
                    max_red = vc - (vc // block_M)
                    olap_list.append((vc - us) / max_red if max_red > 0 else 0.0)
                    vis = (et - 1 - hsa_sliding_window + 1) // chunk_size
                    npre = min(max(vis, 0), num_chunks)
                    cov_list.append(us / npre if npre > 0 else 0.0)
                    union_list.append(us)

            if olap_list:
                print(f"{h:>6d} | {np.mean(olap_list):>10.4f} | "
                      f"{np.mean(cov_list):>10.4f} | {np.mean(union_list):>8.1f}")


def main_per_layer(args):
    """逐层对比所有 HSA 层的 overlap (一张表)"""
    merged, config = run_model_and_collect(args)

    chunk_size = config.get("chunk_size", 64)
    hsa_topk = config.get("hsa_topk", 32)
    hsa_sliding_window = config.get("sliding_window", 4096)
    block_M = args.block_M

    print(f"\n\n{'#'*70}")
    print(f"# Per-Layer Comparison: block_M={block_M}, topk={hsa_topk}")
    print(f"{'#'*70}")
    print(f"\n{'Layer':>6s} | {'Overlap':>10s} | {'Coverage':>10s} | {'Union':>8s} | {'Groups':>8s}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    for lid in sorted(merged.keys()):
        indices = merged[lid]
        B, L, H, K = indices.shape
        results = analyze_overlap(
            indices, block_M, hsa_topk, chunk_size, hsa_sliding_window, seq_len=L * chunk_size
        )
        if results["num_groups"] == 0:
            continue
        print(f"{lid:>6d} | {results['overlap_ratio_mean']:>10.4f} | "
              f"{results['coverage_ratio_mean']:>10.4f} | "
              f"{results['union_size_mean']:>8.1f} | {results['num_groups']:>8d}")


def main_inter_block(args):
    """
    分析相邻 block 之间的 chunk overlap（L2 cache 友好度）。

    对连续的 block pair (block_g, block_{g+1})：
      - intra_overlap: block 内部的 overlap（和之前一样）
      - inter_overlap: 两个 block 的 union 的交集占比
        = |union_g ∩ union_{g+1}| / |union_g ∪ union_{g+1}|  (Jaccard)
      - reuse_ratio:  block_{g+1} 中有多少 chunk 已在 block_g 中出现
        = |union_g ∩ union_{g+1}| / |union_{g+1}|
    """
    merged, config = run_model_and_collect(args)

    chunk_size = config.get("chunk_size", 64)
    hsa_topk = config.get("hsa_topk", 32)
    hsa_sliding_window = config.get("sliding_window", 4096)
    block_M = args.block_M

    print(f"\n\n{'#'*70}")
    print(f"# Inter-Block Overlap (L2 reuse): block_M={block_M}, topk={hsa_topk}")
    print(f"{'#'*70}")

    for lid in sorted(merged.keys()):
        indices = merged[lid]
        B, L, H, K = indices.shape
        indices_cpu = indices.numpy()
        num_groups = L // block_M

        jaccard_list = []
        reuse_list = []

        for b in range(B):
            for h in range(H):
                prev_union = None
                for g in range(num_groups):
                    st = g * block_M
                    et = st + block_M
                    cur_union = set()
                    for t in range(st, et):
                        for k in range(K):
                            idx = int(indices_cpu[b, t, h, k])
                            if idx >= 0:
                                cur_union.add(idx)

                    if prev_union is not None and len(cur_union) > 0 and len(prev_union) > 0:
                        inter = prev_union & cur_union
                        union_both = prev_union | cur_union
                        jaccard_list.append(len(inter) / len(union_both))
                        reuse_list.append(len(inter) / len(cur_union))

                    prev_union = cur_union

        if not jaccard_list:
            print(f"  Layer {lid}: no valid inter-block pairs")
            continue

        print(f"\n  Layer {lid}:")
        print(f"    Jaccard (block_g ∩ block_g+1) / (block_g ∪ block_g+1):")
        print(f"      mean={np.mean(jaccard_list):.4f}  std={np.std(jaccard_list):.4f}")
        print(f"    Reuse  (block_g ∩ block_g+1) / |block_g+1|:")
        print(f"      mean={np.mean(reuse_list):.4f}  std={np.std(reuse_list):.4f}")
        print(f"      => {np.mean(reuse_list)*100:.1f}% of block_g+1's chunks were already in block_g (L2 hot)")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main_ppl(args):
    """
    PPL 验证 (模仿 eval_ppl_hf.py)，确认模型加载正确。
    100 samples, last_k_tokens=512, 8K seq_len.
    """
    from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks
    from transformers import AutoTokenizer

    device = torch.device('cuda:0')

    print(f"\n{'#'*70}")
    print(f"# PPL Sanity Check")
    print(f"#   checkpoint : {args.checkpoint_path}")
    print(f"#   config     : {args.config_path}")
    print(f"#   max_seq_len: {args.max_seq_len}")
    print(f"#   last_k     : 512")
    print(f"#   samples    : {args.max_samples}")
    print(f"{'#'*70}")

    # 加载模型
    print("\nLoading model...")
    model = load_model(args.config_path, args.checkpoint_path, device=device)

    # 读 config
    with open(args.config_path) as f:
        config = json.load(f)
    chunk_size = config.get("chunk_size", 64)

    # 加载数据
    print("Loading data...")
    from data import build_numpy_dataset
    from torch.utils.data import DataLoader, SequentialSampler

    dataset = build_numpy_dataset(args.data_path, args.max_seq_len, namespace='test')
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    def collate_fn(examples):
        return {'input_ids': torch.tensor(examples), 'labels': torch.tensor(examples)}

    dataloader = DataLoader(
        dataset, batch_size=1, collate_fn=collate_fn,
        sampler=SequentialSampler(dataset), num_workers=1,
    )

    last_k_tokens = 512
    ce_fct = nn.CrossEntropyLoss()
    loss_sum = 0.0
    loss_c = 0.0  # Kahan compensation
    steps = 0

    for inputs in dataloader:
        steps += 1
        input_ids = inputs['input_ids'].to(device)
        label_ids = input_ids.clone()

        # 插入 landmark
        input_ids = insert_special_tokens(input_ids, fill_id=tokenizer.vocab_size, chunk_size=chunk_size)
        label_ids_shifted = torch.roll(label_ids, shifts=-1, dims=-1)
        label_ids_shifted[:, -1] = -100
        label_ids_shifted = insert_special_tokens(label_ids_shifted, fill_id=-100, chunk_size=chunk_size)
        label_ids_lmk = torch.roll(label_ids_shifted, shifts=1, dims=-1)

        # position ids
        orig_seq_len = inputs['input_ids'].shape[1]
        pos_ids = create_position_ids_with_landmarks(
            None, orig_seq_len, chunk_size=chunk_size, device=device
        )

        kwargs = {}
        if last_k_tokens > 0:
            kwargs['logits_to_keep'] = last_k_tokens + 1

        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            result = model(input_ids, position_ids=pos_ids, use_cache=False, **kwargs)

        out_len = result.logits.shape[1]
        if last_k_tokens > 0:
            out_len = min(out_len, last_k_tokens + 1)

        loss = ce_fct(
            result.logits[:, -out_len:-1, :].reshape(-1, result.logits.shape[-1]),
            label_ids_lmk[:, -out_len+1:].reshape(-1).to(torch.long).to(device),
        )

        # Kahan summation
        y = loss.item() - loss_c
        t = loss_sum + y
        loss_c = (t - loss_sum) - y
        loss_sum = t

        if steps % 10 == 0:
            mean_loss = loss_sum / steps
            print(f"  step {steps:>4d}, mean_loss={mean_loss:.4f}, ppl={math.exp(mean_loss):.4f}")

        if args.max_samples > 0 and steps >= args.max_samples:
            break

    final_mean_loss = loss_sum / steps
    ppl = math.exp(final_mean_loss)
    print(f"\n  Final: {steps} samples, mean_loss={final_mean_loss:.4f}, PPL={ppl:.4f}")
    return ppl

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze block_M chunk overlap in OLMo3-HSA using real model + data"
    )
    parser.add_argument("--mode", choices=["basic", "sweep", "per_head", "per_layer", "inter_block", "ppl", "all"],
                        default="all", help="Analysis mode")
    parser.add_argument("--config_path", default=DEFAULT_CONFIG,
                        help="Model config JSON")
    parser.add_argument("--checkpoint_path", default=DEFAULT_CKPT,
                        help="Checkpoint path")
    parser.add_argument("--data_path", default=DEFAULT_DATA,
                        help="Eval data path")
    parser.add_argument("--vocab_dir", default=DEFAULT_VOCAB,
                        help="Tokenizer path")
    parser.add_argument("--max_seq_len", type=int, default=8192,
                        help="Max sequence length (before landmark insertion)")
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Number of samples to run")
    parser.add_argument("--block_M", type=int, default=16,
                        help="Block M size for grouping tokens")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"{'='*70}")
    print(f"  OLMo3-HSA Block_M Overlap Analysis (Real Model + Data)")
    print(f"{'='*70}")
    print(f"  Mode          : {args.mode}")
    print(f"  Config        : {args.config_path}")
    print(f"  Checkpoint    : {args.checkpoint_path}")
    print(f"  Data          : {args.data_path}")
    print(f"  max_seq_len   : {args.max_seq_len}")
    print(f"  max_samples   : {args.max_samples}")
    print(f"  block_M       : {args.block_M}")
    print(f"{'='*70}")
    print(f"\nMetrics:")
    print(f"  Overlap Ratio  = (M*topK - union) / ((M-1)*topK)   higher=better, 1=perfect overlap")
    print(f"  Coverage Ratio = union_size / prefix_chunks                     lower=better")

    if args.mode == "ppl":
        main_ppl(args)
        sys.exit(0)

    if args.mode in ("basic", "all"):
        main_basic(args)
    if args.mode in ("sweep", "all"):
        main_sweep(args)
    if args.mode in ("per_head", "all"):
        main_per_head(args)
    if args.mode in ("per_layer", "all"):
        main_per_layer(args)
    if args.mode in ("inter_block", "all"):
        main_inter_block(args)
