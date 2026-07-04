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
    ckpt_path = str(ckpt_path)

    if os.path.isdir(ckpt_path):
        state_dict = OrderedDict()
        safetensor_files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
        if safetensor_files and HAS_SAFETENSORS:
            for f in safetensor_files:
                print(f"  Loading {os.path.basename(f)}")
                shard = load_safetensors(f, device="cpu")
                state_dict.update(shard)
            return state_dict

        bin_files = sorted(glob.glob(os.path.join(ckpt_path, "*.bin")))
        if bin_files:
            for f in bin_files:
                print(f"  Loading {os.path.basename(f)}")
                shard = torch.load(f, map_location="cpu", weights_only=True)
                state_dict.update(shard)
            return state_dict

        raise FileNotFoundError(f"No safetensors or bin files found in directory: {ckpt_path}")

    else:
        if ckpt_path.endswith(".safetensors") and HAS_SAFETENSORS:
            return OrderedDict(load_safetensors(ckpt_path, device="cpu"))
        else:
            return torch.load(ckpt_path, map_location="cpu", weights_only=True)


def merge_checkpoints(checkpoint_paths, output_path, merge_method='average', weights=None):
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path must be provided")

    if merge_method == 'weighted_average' and (weights is None or len(weights) != len(checkpoint_paths)):
        raise ValueError("Weighted average requires a corresponding weight for each checkpoint")

    if weights is None:
        weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)

    print(f"Processing checkpoint 1/{len(checkpoint_paths)}: {checkpoint_paths[0]}")
    base_state_dict = load_ckpt(checkpoint_paths[0])

    merged_state_dict = OrderedDict()
    for key in base_state_dict.keys():
        merged_state_dict[key] = weights[0] * base_state_dict[key].float()

    del base_state_dict

    for i, checkpoint_path in enumerate(checkpoint_paths[1:], 1):
        print(f"Processing checkpoint {i+1}/{len(checkpoint_paths)}: {checkpoint_path}")
        state_dict = load_ckpt(checkpoint_path)

        if set(state_dict.keys()) != set(merged_state_dict.keys()):
            missing = set(merged_state_dict.keys()) - set(state_dict.keys())
            extra = set(state_dict.keys()) - set(merged_state_dict.keys())
            msg = f"Keys of checkpoint {checkpoint_path} do not match the base checkpoint"
            if missing:
                msg += f"\n  Missing: {list(missing)[:5]}..."
            if extra:
                msg += f"\n  Extra: {list(extra)[:5]}..."
            raise ValueError(msg)

        for key in state_dict.keys():
            merged_state_dict[key] += weights[i] * state_dict[key].float()

        del state_dict

    for key in merged_state_dict:
        merged_state_dict[key] = merged_state_dict[key].bfloat16()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if HAS_SAFETENSORS:
        save_safetensors(merged_state_dict, str(output_path / "model.safetensors"))
        print(f"Merged model saved to: {output_path / 'model.safetensors'}")
    else:
        torch.save(merged_state_dict, output_path / "model.bin")
        print(f"Merged model saved to: {output_path / 'model.bin'}")

    first_ckpt = Path(checkpoint_paths[0])
    if first_ckpt.is_dir():
        import shutil
        for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json",
                            "special_tokens_map.json", "generation_config.json",
                            "preprocessor_config.json", "processor_config.json"]:
            src = first_ckpt / config_file
            if src.exists():
                shutil.copy2(str(src), str(output_path / config_file))
                print(f"  Copying config: {config_file}")

    print(f"\nMerge complete! Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Merge multiple model checkpoints')
    parser.add_argument('checkpoints', nargs='+', help='Checkpoint paths to merge (HF ckpt directory or file)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory path (default: parent directory of the first ckpt / merged)')
    parser.add_argument('--memo', type=str, default='merged_model',
                        help='Alias for the merged model (used for the output directory name)')
    parser.add_argument('--method', type=str, default='average',
                        choices=['average', 'weighted_average'],
                        help='Merge method')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                        help='List of weights for weighted average')

    args = parser.parse_args()

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
