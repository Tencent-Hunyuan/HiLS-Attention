import torch
import argparse
import os
import glob
from pathlib import Path
from collections import OrderedDict

try:
    from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def load_ckpt(ckpt_path):
    """
    加载 HF checkpoint 目录或单个文件。
    支持:
      - 目录 (包含 *.safetensors 或 *.bin 文件)
      - 单个 .safetensors / .bin / .pt / .pth 文件
    返回: state_dict (OrderedDict)
    """
    ckpt_path = str(ckpt_path)

    if os.path.isdir(ckpt_path):
        state_dict = OrderedDict()
        # 优先加载 safetensors
        safetensor_files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
        if safetensor_files and HAS_SAFETENSORS:
            for f in safetensor_files:
                print(f"  加载 {os.path.basename(f)}")
                shard = load_safetensors(f, device="cpu")
                state_dict.update(shard)
            return state_dict

        # fallback: pytorch bin 文件
        bin_files = sorted(glob.glob(os.path.join(ckpt_path, "*.bin")))
        if bin_files:
            for f in bin_files:
                print(f"  加载 {os.path.basename(f)}")
                shard = torch.load(f, map_location="cpu", weights_only=True)
                state_dict.update(shard)
            return state_dict

        raise FileNotFoundError(f"目录中未找到 safetensors 或 bin 文件: {ckpt_path}")

    else:
        # 单个文件
        if ckpt_path.endswith(".safetensors") and HAS_SAFETENSORS:
            return OrderedDict(load_safetensors(ckpt_path, device="cpu"))
        else:
            return torch.load(ckpt_path, map_location="cpu", weights_only=True)


def merge_checkpoints(checkpoint_paths, output_path, merge_method='average', weights=None):
    """
    合并多个模型检查点

    参数:
        checkpoint_paths: 检查点路径列表 (HF ckpt 目录或文件)
        output_path: 合并后模型的保存路径
        merge_method: 合并方法，可选 'average' 或 'weighted_average'
        weights: 当使用加权平均时，各模型的权重列表
    """
    if not checkpoint_paths:
        raise ValueError("至少需要提供一个检查点路径")

    if merge_method == 'weighted_average' and (weights is None or len(weights) != len(checkpoint_paths)):
        raise ValueError("加权平均需要为每个检查点提供对应的权重")

    if weights is None:
        weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)

    # 加载第一个检查点
    print(f"处理检查点 1/{len(checkpoint_paths)}: {checkpoint_paths[0]}")
    base_state_dict = load_ckpt(checkpoint_paths[0])

    # 初始化合并后的状态字典
    merged_state_dict = OrderedDict()
    for key in base_state_dict.keys():
        merged_state_dict[key] = weights[0] * base_state_dict[key].float()

    # 释放内存
    del base_state_dict

    # 处理剩余的检查点
    for i, checkpoint_path in enumerate(checkpoint_paths[1:], 1):
        print(f"处理检查点 {i+1}/{len(checkpoint_paths)}: {checkpoint_path}")
        state_dict = load_ckpt(checkpoint_path)

        # 验证状态字典的键是否匹配
        if set(state_dict.keys()) != set(merged_state_dict.keys()):
            missing = set(merged_state_dict.keys()) - set(state_dict.keys())
            extra = set(state_dict.keys()) - set(merged_state_dict.keys())
            msg = f"检查点 {checkpoint_path} 的键与基础检查点不匹配"
            if missing:
                msg += f"\n  缺少: {list(missing)[:5]}..."
            if extra:
                msg += f"\n  多余: {list(extra)[:5]}..."
            raise ValueError(msg)

        for key in state_dict.keys():
            merged_state_dict[key] += weights[i] * state_dict[key].float()

        del state_dict

    # 转回 bfloat16 保存
    for key in merged_state_dict:
        merged_state_dict[key] = merged_state_dict[key].bfloat16()

    # 保存合并后的检查点
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if HAS_SAFETENSORS:
        # 保存为 safetensors 格式
        save_safetensors(merged_state_dict, str(output_path / "model.safetensors"))
        print(f"合并后的模型已保存到: {output_path / 'model.safetensors'}")
    else:
        torch.save(merged_state_dict, output_path / "model.bin")
        print(f"合并后的模型已保存到: {output_path / 'model.bin'}")

    # 复制第一个 checkpoint 中的配置文件
    first_ckpt = Path(checkpoint_paths[0])
    if first_ckpt.is_dir():
        import shutil
        for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json",
                            "special_tokens_map.json", "generation_config.json",
                            "preprocessor_config.json", "processor_config.json"]:
            src = first_ckpt / config_file
            if src.exists():
                shutil.copy2(str(src), str(output_path / config_file))
                print(f"  复制配置: {config_file}")

    print(f"\n合并完成! 输出目录: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='合并多个模型检查点')
    parser.add_argument('checkpoints', nargs='+', help='要合并的检查点路径 (HF ckpt 目录或文件)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录路径 (默认: 第一个 ckpt 的父目录/merged)')
    parser.add_argument('--memo', type=str, default='merged_model',
                        help='合并后模型的别名 (用于输出目录名)')
    parser.add_argument('--method', type=str, default='average',
                        choices=['average', 'weighted_average'],
                        help='合并方法')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                        help='加权平均的权重列表')

    args = parser.parse_args()

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        first_parent = Path(args.checkpoints[0]).parent
        output_path = str(first_parent / args.memo)

    merge_checkpoints(
        checkpoint_paths=args.checkpoints,
        output_path=output_path,
        merge_method=args.method,
        weights=args.weights,
    )


if __name__ == '__main__':
    main()
