import os
import numpy as np
from tqdm import tqdm
from itertools import accumulate
from torch.utils.data import Dataset
from typing import Literal
import random
from bisect import bisect_right

# from antllm_.tokenization_bailing import BailingTokenizer


class LazyChunkedLoader(object):
    """
    Lazy loader for a two-level directory.

    All files are mmapped during initialization.
    Example of lazy loader directory structure:
        path
            chunk[i]
                *.data
    """

    def __init__(
            self, 
            path, 
            split: Literal['train', 'test']='train', 
            val_ratio=0.1, 
            data_type='data', 
            array_data_type=np.uint32,
            sort_files: bool = False,
        ):
        """
        Args:
            sort_files: if True, sort collected files by their full path in
                ascending order.  This makes the file ordering deterministic
                across different filesystems / OS / ranks, which is important
                for reproducibility and for aligning data across distributed
                workers.  Defaults to False to preserve backward compatibility
                with existing checkpoints that were trained against the native
                ``os.walk`` order (filesystem-dependent).
        """
        self.array_data_type = array_data_type
        self.is_lazy = True
        self.split = split
        self.val_ratio = val_ratio
        self.sort_files = sort_files
        print(f"LazyChunkedLoader: Loading {path}")
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"LazyChunkedLoader: path does not exist or is not a directory: {path}\n"
                f"Hint: if you run inside a container, make sure this path is mounted into the container."
            )
        all_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(f".{data_type}"):
                    all_files.append(os.path.join(root, file))
        if sort_files:
            # Sort by full path (ascending) for deterministic ordering.
            all_files.sort()
            print(f"LazyChunkedLoader: sorted {len(all_files)} files by path (ascending).")
        if len(all_files) == 0:
            raise FileNotFoundError(
                f"LazyChunkedLoader: no '*.{data_type}' files found under: {path}\n"
                f"split={split}, val_ratio={val_ratio}\n"
                f"Hint: verify DATA_PATH points to the tokenized corpus root and is accessible in this runtime."
            )

        files_ptrs = []
        files_lens = []

        # mmap all files and get pointers to each file
        for f in tqdm(all_files, desc="DataLoader: mmaping tokenized files", ncols=120):
            try:
                fsize = os.path.getsize(f)
                if (fsize > 0):
                    np_array = np.memmap(f, dtype=np.uint32, mode='r')
                    ids_len = np_array.shape[0]
                    offset = 0 if split == 'train' else int((1 - val_ratio) * ids_len)
                    end = int((1 - val_ratio) * ids_len) if split == 'train' else ids_len
                    files_ptrs.append(np_array[offset: end])
                    files_lens.append(files_ptrs[-1].shape[0])
                else:
                    print(f"Warning: file {f} is empty, skipping")

            except Exception as e:
                print(f"Error: {e} in file {f}, skipping")

        self.files_ptrs = files_ptrs
        self.lens = files_lens
        self.ends = list(accumulate(self.lens))
        if len(self.ends) == 0:
            raise RuntimeError(
                "LazyChunkedLoader: all candidate files were skipped or produced empty slices.\n"
                f"path={path}, split={split}, val_ratio={val_ratio}, data_type={data_type}\n"
                "Hint: check file permissions, file sizes, and that memmap dtype matches the stored format."
            )

        print(f"total documents: {len(self.lens)}, total tokens:{self.ends[-1]}")
        self.total_tokens = self.ends[-1]

    def __getitem__(self, index):
        """
        return the whole file with the index
        """
        if not isinstance(index, slice):
            rtn = self.files_ptrs[index][:]     # Return the whole file
        else:
            # No need to sample across files. A file already contains many samples.
            raise NotImplementedError(f"Reading slice is not supported: index={index}")
        return rtn


    def __len__(self):
        return len(self.ends)

    def get_text_len(self, idx):
        return self.lens[idx]
        # prev_end = self.ends[idx - 1] if idx > 0 else 0
        # return self.ends[idx] - prev_end

    def file_read(self, start=0, end=None):
        raise NotImplementedError("LazyChunkedLoader: file_read is not supported\nHint: Use LazyLoader instead.")

class TextDataset(Dataset):
    """
    Only support lazy loader for now.
    """
    def __init__(self, ds,
                 segment_len=1024,
                 weighted=True,
                 random_sampling=True,
                 sample_across_doc=True,
                 random_across_doc_sampling=True
    ):
        self.ds = ds
        self.ds_len = len(self.ds)
        self.segment_len = segment_len
        self.weighted = weighted
        self.sample_across_doc = sample_across_doc
        self.random_across_doc_sampling = random_across_doc_sampling
        self.weighting, self.total_len = None, None
        self.is_lazy = True
        self.random_sampling = random_sampling
        # print ("Dataset length: " + str(len(self)))
        # if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
        #     self.is_lazy = True
        self.init_weighting()
        # print (self.weighting)

    def init_weighting(self):
        if self.weighted:
            if self.is_lazy:
                lens = np.array([self.ds.get_text_len(idx) for idx in range(len(self.ds))])
            else:
                lens = np.array([len(d['text']) if isinstance(d, dict)
                                 else len(d) for d in self.ds])
            self.total_len = np.sum(lens)
            # print(f"Dataset document count {len(lens)}, token count {self.total_len}")
            self.weighting = list(accumulate(lens))
        else:
            self.weighting = None


    """
    Ramdomly select a document with length of each documents as weights
    """
    def get_weighted_samples(self, np_rng):
        if self.weighting is not None:
            idx = np_rng.randint(self.total_len)
            return bisect_right(self.weighting, idx)
            # return max(0, bisect_right(self.weighting, idx)-1)
        else:
            return np_rng.randint(self.ds_len)

    def __len__(self):
        return self.ds.total_tokens // self.segment_len

    def __getitem__(self, idx):
        # init rng
        rng = random.Random(idx)
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        # get length weighted random index from dataset
        doc_idx = self.get_weighted_samples(rng)
        doc_len = self.ds.get_text_len(doc_idx)
        tokens = None

        if not self.sample_across_doc:
            tokens_to_skip = doc_len - self.segment_len
            doc_tokens = self.ds[doc_idx]        
            if tokens_to_skip >= 0:
                token_start_idx = rng.randint(tokens_to_skip + 1)
                tokens = doc_tokens[token_start_idx:token_start_idx+self.segment_len]
                # for t in tokens:
                #     assert t >= 0 and t < 50257
            else:
                tokens = np.concatenate((doc_tokens, np.array([-100] * abs(tokens_to_skip), dtype=np.int32)))

            assert len(tokens) == self.segment_len
            return np.array(tokens, dtype=np.int32)
        else:
            if not self.random_across_doc_sampling:
                assert self.segment_len < self.ds.total_tokens
                start = rng.randint(self.ds.total_tokens - self.segment_len - 1)
                assert start+self.segment_len <= self.ds.total_tokens
                tokens = self.ds.file_read(start, start + self.segment_len)
                assert len(tokens) == self.segment_len
                return np.array(tokens, dtype=np.int32)
            else:
                # randomly sample across doc
                tokens = []
                while (len(tokens) < self.segment_len):
                    remaining_tokens = self.segment_len - len(tokens)
                    assert remaining_tokens > 0
                    doc_idx = self.get_weighted_samples(rng)
                    doc_len = self.ds.get_text_len(doc_idx)
                    doc_tokens = self.ds[doc_idx]
                    start = rng.randint(doc_len)
                    end = min(start + remaining_tokens, doc_len)
                    tokens.extend(doc_tokens[start:end])
                    
                assert len(tokens) == self.segment_len
                return np.array(tokens, dtype=np.int32)
