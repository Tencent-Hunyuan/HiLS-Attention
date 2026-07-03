import numpy as np
import random
from bisect import bisect_right
from torch.utils.data import Dataset
from typing import List


class MixedTextDataset(Dataset):
    """
    Weighted mixture of multiple LazyChunkedLoader sources.

    At each __getitem__, first selects a source proportional to weights,
    then samples a segment from that source (same logic as TextDataset).
    """

    def __init__(
        self,
        loaders: list,
        weights: List[float],
        names: List[str],
        segment_len: int = 8192,
    ):
        self.loaders = loaders
        self.weights = weights
        self.names = names
        self.segment_len = segment_len

        # Build per-source cumulative weights for fast sampling
        from itertools import accumulate
        self.cum_weights = list(accumulate(weights))

        # Per-source document-level cumulative lengths (for weighted doc sampling)
        self.source_doc_cum = []
        self.source_total_tokens = []
        for loader in loaders:
            lens = np.array([loader.get_text_len(i) for i in range(len(loader))])
            total = int(np.sum(lens))
            cum = list(accumulate(lens))
            self.source_doc_cum.append(cum)
            self.source_total_tokens.append(total)

        # Total dataset length: sum of weighted token contributions
        self.total_tokens = sum(self.source_total_tokens)

    def __len__(self):
        return self.total_tokens // self.segment_len

    def _pick_source(self, rng: np.random.RandomState) -> int:
        """Pick a source index proportional to weights."""
        r = rng.random()
        return bisect_right(self.cum_weights, r)

    def _sample_from_source(self, source_idx: int, rng: np.random.RandomState) -> np.ndarray:
        """Sample a segment_len chunk from the given source, concatenating across docs if needed."""
        loader = self.loaders[source_idx]
        cum = self.source_doc_cum[source_idx]
        total = self.source_total_tokens[source_idx]

        tokens = []
        while len(tokens) < self.segment_len:
            remaining = self.segment_len - len(tokens)
            # Weighted document sampling
            idx = rng.randint(total)
            doc_idx = bisect_right(cum, idx)
            doc_idx = min(doc_idx, len(loader) - 1)
            doc_len = loader.get_text_len(doc_idx)
            doc_tokens = loader[doc_idx]
            start = rng.randint(doc_len)
            end = min(start + remaining, doc_len)
            tokens.extend(doc_tokens[start:end])

        return np.array(tokens[:self.segment_len], dtype=np.int32)

    def __getitem__(self, idx):
        # Deterministic RNG from idx
        rng = random.Random(idx)
        rng = np.random.RandomState(seed=[rng.randint(0, 2**32 - 1) for _ in range(16)])

        source_idx = self._pick_source(rng)
        return self._sample_from_source(source_idx, rng)
