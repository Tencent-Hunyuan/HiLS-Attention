from veomni.data.dataset import DATASET_REGISTRY, MappingDataset
from typing import Optional, Literal, Callable
from .lazy_dataset import LazyChunkedLoader, TextDataset
from .data_transform import process_numpy_example, RulerSynthesizer, synthesize_ruler_example
from .chat_template import OLMo3Template 


__all__ = [
    "process_numpy_example",
    "RulerSynthesizer",
    "synthesize_ruler_example",
]


@DATASET_REGISTRY.register('olmo3')
def build_numpy_dataset(
    train_path: str,
    max_seq_len: int,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    sort_files: bool = False,
    **kwargs,
) -> "Dataset":
    ds = LazyChunkedLoader(train_path, split=namespace, sort_files=sort_files)
    dataset = TextDataset(ds, max_seq_len)
    return MappingDataset(data=dataset, transform=transform)


@DATASET_REGISTRY.register('olmo3_mix')
def build_numpy_mix_dataset(
    train_path: str,
    max_seq_len: int,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    **kwargs,
) -> "Dataset":
    """
    Build a weighted mixture dataset from a YAML config.

    The YAML should have:
        data_root: /path/to/tokenized/data
        sources:
          - name: source_name
            dirs: [dir1, dir2, ...]
            weight: 0.1
          ...

    Each source creates a LazyChunkedLoader over all .data files in its dirs.
    The MixedTextDataset samples sources proportionally to their weights,
    then samples a segment from the selected source.
    """
    from .mix_dataset import MixedTextDataset
    import yaml, os

    with open(train_path) as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    sources = cfg["sources"]

    loaders = []
    weights = []
    names = []

    for src in sources:
        dirs = src["dirs"]
        full_dirs = [os.path.join(data_root, d) for d in dirs]

        # Collect all .data files across multiple dirs for this source
        all_files = []
        for d in full_dirs:
            if not os.path.isdir(d):
                print(f"Warning: directory {d} does not exist, skipping source {src['name']}")
                continue
            for root, _, files in os.walk(d, followlinks=True):
                for file in files:
                    if file.endswith(".data"):
                        all_files.append(os.path.join(root, file))

        if not all_files:
            print(f"Warning: no .data files found for source {src['name']}, skipping")
            continue

        loader = LazyChunkedLoader.__new__(LazyChunkedLoader)
        loader.array_data_type = __import__('numpy').uint32
        loader.is_lazy = True
        loader.split = namespace
        loader.val_ratio = 0.1

        import numpy as np
        from tqdm import tqdm
        from itertools import accumulate

        files_ptrs = []
        files_lens = []
        for fpath in tqdm(all_files, desc=f"Loading {src['name']}", ncols=120):
            try:
                fsize = os.path.getsize(fpath)
                if fsize > 0:
                    np_array = np.memmap(fpath, dtype=np.uint32, mode='r')
                    ids_len = np_array.shape[0]
                    offset = 0 if namespace == 'train' else int(0.9 * ids_len)
                    end = int(0.9 * ids_len) if namespace == 'train' else ids_len
                    files_ptrs.append(np_array[offset:end])
                    files_lens.append(files_ptrs[-1].shape[0])
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

        if not files_ptrs:
            print(f"Warning: all files empty for source {src['name']}, skipping")
            continue

        loader.files_ptrs = files_ptrs
        loader.lens = files_lens
        loader.ends = list(accumulate(files_lens))
        loader.total_tokens = loader.ends[-1]
        print(f"Source {src['name']}: {len(files_lens)} files, {loader.total_tokens} tokens")

        loaders.append(loader)
        weights.append(src["weight"])
        names.append(src["name"])

    # Re-normalize weights in case some sources were skipped
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    print(f"\n=== Mix Dataset Summary ===")
    print(f"Active sources: {len(loaders)}")
    for n, w, ld in zip(names, weights, loaders):
        print(f"  {n}: weight={w:.4f}, tokens={ld.total_tokens}")
    total_tokens = sum(ld.total_tokens for ld in loaders)
    print(f"Total tokens: {total_tokens}")
    print(f"===========================\n")

    dataset = MixedTextDataset(loaders, weights, names, max_seq_len)
    return MappingDataset(data=dataset, transform=transform)
