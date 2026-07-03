"""
Unit tests for MixedTextDataset sampling correctness.

Verifies that the dataset samples from multiple sources according to
a given weight (mix ratio) table, produces correct segment lengths,
and is deterministically reproducible.
"""

import numpy as np
import pytest
from collections import Counter

from data.mix_dataset import MixedTextDataset


# ---------------------------------------------------------------------------
# Mock loader that mimics the LazyChunkedLoader interface
# ---------------------------------------------------------------------------

class MockLoader:
    """
    Lightweight stand-in for LazyChunkedLoader.

    Each "document" is a numpy array filled with a unique token value
    derived from (source_id, doc_index) so that we can trace which source
    a token came from.

    Parameters
    ----------
    source_id : int
        Identifier for this source (used to generate distinguishable tokens).
    num_docs : int
        Number of documents in this source.
    doc_len : int
        Length (in tokens) of every document.
    """

    def __init__(self, source_id: int, num_docs: int = 10, doc_len: int = 2048):
        self.source_id = source_id
        self.num_docs = num_docs
        self.doc_len = doc_len
        self.is_lazy = True

        # Token value for doc i = source_id * 10000 + i
        # This makes it easy to identify which source a token belongs to.
        self.docs = [
            np.full(doc_len, source_id * 10000 + i, dtype=np.int32)
            for i in range(num_docs)
        ]
        self.lens = [doc_len] * num_docs
        self.total_tokens = num_docs * doc_len

    def __len__(self):
        return self.num_docs

    def __getitem__(self, index):
        return self.docs[index]

    def get_text_len(self, idx):
        return self.lens[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identify_source(token_value: int) -> int:
    """Recover the source_id from a token produced by MockLoader."""
    return token_value // 10000


def _build_dataset(
    num_sources: int = 3,
    weights: list | None = None,
    num_docs: int = 10,
    doc_len: int = 2048,
    segment_len: int = 256,
) -> MixedTextDataset:
    """Helper to build a MixedTextDataset backed by MockLoaders."""
    if weights is None:
        weights = [1.0 / num_sources] * num_sources

    loaders = [MockLoader(source_id=i, num_docs=num_docs, doc_len=doc_len) for i in range(num_sources)]
    names = [f"source_{i}" for i in range(num_sources)]
    return MixedTextDataset(loaders=loaders, weights=weights, names=names, segment_len=segment_len)


# ===========================================================================
# Tests
# ===========================================================================


class TestMixedTextDatasetBasic:
    """Basic sanity checks."""

    def test_segment_length(self):
        """Every sample must have exactly `segment_len` tokens."""
        segment_len = 128
        ds = _build_dataset(num_sources=3, segment_len=segment_len)
        for idx in range(min(50, len(ds))):
            sample = ds[idx]
            assert isinstance(sample, np.ndarray), f"Sample {idx} is not ndarray"
            assert sample.shape == (segment_len,), (
                f"Sample {idx}: expected shape ({segment_len},), got {sample.shape}"
            )

    def test_deterministic_reproducibility(self):
        """Same idx must always produce the same output."""
        ds = _build_dataset(num_sources=3, segment_len=256)
        for idx in [0, 1, 42, 99, len(ds) - 1]:
            if idx >= len(ds):
                continue
            a = ds[idx]
            b = ds[idx]
            np.testing.assert_array_equal(a, b, err_msg=f"Non-deterministic at idx={idx}")

    def test_dataset_length(self):
        """__len__ should equal total_tokens // segment_len."""
        segment_len = 256
        ds = _build_dataset(num_sources=2, num_docs=5, doc_len=1024, segment_len=segment_len)
        expected_total = sum(ld.total_tokens for ld in ds.loaders)
        assert len(ds) == expected_total // segment_len


class TestMixedTextDatasetWeightedSampling:
    """Verify that sampling respects the configured weight table."""

    @pytest.mark.parametrize(
        "weights, tolerance",
        [
            ([0.5, 0.3, 0.2], 0.05),
            ([0.8, 0.1, 0.1], 0.05),
            ([0.1, 0.9], 0.05),
            ([1.0 / 3, 1.0 / 3, 1.0 / 3], 0.05),
        ],
        ids=["50-30-20", "80-10-10", "10-90", "uniform-3"],
    )
    def test_sampling_ratio_matches_weights(self, weights, tolerance):
        """
        Draw many samples and check that the fraction of tokens from each
        source is within `tolerance` of the configured weight.
        """
        num_sources = len(weights)
        segment_len = 128
        ds = _build_dataset(
            num_sources=num_sources,
            weights=weights,
            num_docs=20,
            doc_len=4096,
            segment_len=segment_len,
        )

        num_samples = 2000
        source_counter = Counter()

        for idx in range(num_samples):
            sample = ds[idx]
            # All tokens in a single segment come from the same source
            # (because MockLoader fills each doc with a constant value,
            #  and _sample_from_source keeps sampling from the same source).
            src_id = _identify_source(int(sample[0]))
            source_counter[src_id] += 1

        # Normalize weights (MixedTextDataset uses cumulative raw weights,
        # so the caller is responsible for normalization in practice;
        # but _pick_source compares against random() in [0,1) using
        # cumulative *raw* weights — so we normalize here for comparison).
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        for src_id in range(num_sources):
            observed_ratio = source_counter.get(src_id, 0) / num_samples
            expected_ratio = norm_weights[src_id]
            assert abs(observed_ratio - expected_ratio) < tolerance, (
                f"Source {src_id}: expected ratio ~{expected_ratio:.3f}, "
                f"got {observed_ratio:.3f} (tolerance={tolerance})"
            )

    def test_extreme_weight_single_dominant(self):
        """
        When one source has weight ≈1.0 and others ≈0, almost all samples
        should come from the dominant source.
        """
        weights = [0.98, 0.01, 0.01]
        segment_len = 128
        ds = _build_dataset(
            num_sources=3,
            weights=weights,
            num_docs=20,
            doc_len=4096,
            segment_len=segment_len,
        )

        num_samples = 1000
        dominant_count = 0
        for idx in range(num_samples):
            sample = ds[idx]
            src_id = _identify_source(int(sample[0]))
            if src_id == 0:
                dominant_count += 1

        ratio = dominant_count / num_samples
        assert ratio > 0.90, (
            f"Dominant source (weight=0.98) only got {ratio:.3f} of samples"
        )


class TestMixedTextDatasetTokenProvenance:
    """Verify that tokens actually originate from the claimed source."""

    def test_all_tokens_from_same_source_per_segment(self):
        """
        Within a single segment, all tokens should belong to the same source
        (since _sample_from_source draws from one source only).

        Note: tokens may span multiple documents within that source, but
        MockLoader uses source_id * 10000 + doc_idx, so source_id is the
        same for all docs in a source.
        """
        ds = _build_dataset(num_sources=4, segment_len=256, num_docs=10, doc_len=512)
        for idx in range(200):
            sample = ds[idx]
            sources_in_segment = set(_identify_source(int(t)) for t in sample)
            assert len(sources_in_segment) == 1, (
                f"Sample {idx} contains tokens from multiple sources: {sources_in_segment}"
            )

    def test_token_values_are_valid(self):
        """All token values should be recognizable MockLoader tokens."""
        num_sources = 3
        num_docs = 10
        ds = _build_dataset(
            num_sources=num_sources,
            segment_len=128,
            num_docs=num_docs,
            doc_len=1024,
        )
        for idx in range(100):
            sample = ds[idx]
            for t in sample:
                src = _identify_source(int(t))
                doc = int(t) % 10000
                assert 0 <= src < num_sources, f"Invalid source id {src} from token {t}"
                assert 0 <= doc < num_docs, f"Invalid doc id {doc} from token {t}"


class TestMixedTextDatasetEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_source(self):
        """Dataset with a single source should always sample from it."""
        ds = _build_dataset(num_sources=1, weights=[1.0], segment_len=128)
        for idx in range(50):
            sample = ds[idx]
            src_id = _identify_source(int(sample[0]))
            assert src_id == 0, f"Single-source dataset sampled from source {src_id}"

    def test_segment_longer_than_single_doc(self):
        """
        When segment_len > doc_len, the sampler must concatenate tokens
        from multiple documents within the same source.
        """
        doc_len = 64
        segment_len = 256  # 4x doc_len
        ds = _build_dataset(
            num_sources=2,
            weights=[0.5, 0.5],
            num_docs=20,
            doc_len=doc_len,
            segment_len=segment_len,
        )
        for idx in range(50):
            sample = ds[idx]
            assert sample.shape == (segment_len,)
            # Tokens may come from different docs but same source
            sources_in_segment = set(_identify_source(int(t)) for t in sample)
            assert len(sources_in_segment) == 1

    def test_many_sources(self):
        """Stress test with many sources."""
        num_sources = 10
        weights = [1.0 / num_sources] * num_sources
        ds = _build_dataset(
            num_sources=num_sources,
            weights=weights,
            num_docs=5,
            doc_len=1024,
            segment_len=64,
        )
        # Just verify it doesn't crash and produces correct shapes
        for idx in range(100):
            sample = ds[idx]
            assert sample.shape == (64,)

    def test_different_doc_lengths_per_source(self):
        """
        Sources with different document lengths should still produce
        correct segment_len outputs.
        """
        segment_len = 128
        loaders = [
            MockLoader(source_id=0, num_docs=5, doc_len=100),   # short docs
            MockLoader(source_id=1, num_docs=5, doc_len=5000),  # long docs
            MockLoader(source_id=2, num_docs=5, doc_len=500),   # medium docs
        ]
        weights = [0.3, 0.4, 0.3]
        names = ["short", "long", "medium"]
        ds = MixedTextDataset(loaders=loaders, weights=weights, names=names, segment_len=segment_len)

        for idx in range(100):
            sample = ds[idx]
            assert sample.shape == (segment_len,), (
                f"Sample {idx}: expected ({segment_len},), got {sample.shape}"
            )


class TestMixedTextDatasetMixRatioTable:
    """
    End-to-end test with a concrete mix-ratio table, simulating a real
    training configuration.
    """

    def test_realistic_mix_ratio(self):
        """
        Simulate a realistic 5-source mix:
            code:       30%
            web:        25%
            book:       20%
            wiki:       15%
            math:       10%

        Verify observed ratios are within 5% of target after 5000 samples.
        """
        mix_table = {
            "code": 0.30,
            "web":  0.25,
            "book": 0.20,
            "wiki": 0.15,
            "math": 0.10,
        }
        names = list(mix_table.keys())
        weights = list(mix_table.values())
        num_sources = len(names)
        segment_len = 128

        loaders = [
            MockLoader(source_id=i, num_docs=30, doc_len=4096)
            for i in range(num_sources)
        ]
        ds = MixedTextDataset(
            loaders=loaders,
            weights=weights,
            names=names,
            segment_len=segment_len,
        )

        num_samples = 5000
        source_counter = Counter()
        for idx in range(num_samples):
            sample = ds[idx]
            src_id = _identify_source(int(sample[0]))
            source_counter[src_id] += 1

        total_w = sum(weights)
        print("\n=== Realistic Mix Ratio Verification ===")
        for i, name in enumerate(names):
            expected = weights[i] / total_w
            observed = source_counter.get(i, 0) / num_samples
            status = "✓" if abs(observed - expected) < 0.05 else "✗"
            print(f"  {status} {name:>6s}: expected={expected:.3f}, observed={observed:.3f}")
            assert abs(observed - expected) < 0.05, (
                f"{name}: expected ~{expected:.3f}, got {observed:.3f}"
            )
        print("========================================")
