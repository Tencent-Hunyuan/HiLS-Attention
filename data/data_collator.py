"""
Custom collator that truncates/pads a packed batch to a fixed target length.

Designed to be inserted into veomni's CollatePipeline, e.g.:

    from data.data_collator import PadToFixedLengthCollator

    collate_fn = [
        DataCollatorWithPositionIDs(),
        PadToFixedLengthCollator(target_length=max_seq_len),
    ]
    # veomni's build_native_dataloader wraps list into CollatePipeline automatically.
"""

from dataclasses import dataclass
from typing import Dict

import torch

from veomni.data.data_collator import (
    DataCollator,
    add_flash_attention_kwargs_from_position_ids,
)
from veomni.distributed.parallel_state import get_parallel_state


IGNORE_INDEX = -100


@dataclass
class PadToFixedLengthCollator(DataCollator):
    """
    Pipeline-stage collator that truncates or pads an already-packed batch

    Placed **after** DataCollatorWithPositionIDs and **before**
    TextSequenceShardCollator in the pipeline.
    """

    target_length: int = 0

    def __post_init__(self):
        assert self.target_length > 0, "target_length must be > 0"
        self.sp_enabled = get_parallel_state().sp_enabled

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seq_len = batch["input_ids"].shape[-1]
        seq_keys = ("input_ids", "labels", "attention_mask", "position_ids")
        assert seq_len <= self.target_length, (
            f"Packed sequence length ({seq_len}) exceeds target_length ({self.target_length}). "
            f"Check dynamic batching config."
        )

        # --- pad to target_length ---
        pad_len = self.target_length - seq_len
        if pad_len > 0:
            pad_specs = {
                "input_ids": 0,
                "labels": IGNORE_INDEX,
                "attention_mask": 1,
                "position_ids": self.target_length - 1,
            }
            for key, pad_value in pad_specs.items():
                if key not in batch:
                    continue
                tensor = batch[key]
                pad_shape = list(tensor.shape)
                pad_shape[-1] = pad_len
                pad = torch.full(pad_shape, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device)
                batch[key] = torch.cat([tensor, pad], dim=-1)

        # Recompute flash attention kwargs from the (now fixed-length) position_ids
        if not self.sp_enabled and "position_ids" in batch:
            for k in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
                batch.pop(k, None)
            add_flash_attention_kwargs_from_position_ids(batch)

        return batch
