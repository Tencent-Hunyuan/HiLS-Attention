from veomni.utils.arguments import DataArguments, TrainingArguments
from typing import Literal, Optional
from dataclasses import dataclass, field


@dataclass
class OLmo3DataArguments(DataArguments):
    data_type: Literal["plaintext", "conversation", "diffusion", 'numpy'] = field(
        default="conversation",
        metadata={"help": "Type of the training data."},
    )

@dataclass
class RULERDataArguments(DataArguments):
    data_type: str = field(
        default="ruler",
        metadata={"help": "ruler_[ratio]."},
    )
    sort_files: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, sort data files by full path (ascending) when "
                "collecting them in LazyChunkedLoader. Makes file ordering "
                "deterministic across filesystems / ranks. Default False for "
                "backward compatibility with existing checkpoints."
            )
        },
    )

    def __post_init__(self):
        # For olmo3_mix, we handle the YAML ourselves - don't let veomni
        # treat it as a multisource interleave dataset.
        if self.datasets_type in ("olmo3_mix",):
            # Skip the parent __post_init__ which would set enable_multisource=True
            # and override dataset_name to "interleave"
            self.enable_multisource = False
            self.dataset_name = self.datasets_type
        else:
            super().__post_init__()

@dataclass
class ExtendedTrainingArguments(TrainingArguments):
    # Stage2 often wants to resume model/dataloader RNG but rebuild optimizer/scheduler
    # (e.g. stage1 freezes some params and excludes them from optimizer param groups).
    load_optimizer_state: bool = field(
        default=True,
        metadata={"help": "Whether to restore optimizer state from checkpoint when resuming."},
    )
    load_lr_scheduler_state: bool = field(
        default=True,
        metadata={"help": "Whether to restore lr_scheduler state from checkpoint when resuming."},
    )
    load_dataloader_state: bool = field(
        default=True,
        metadata={"help": "Whether to restore dataloader state from checkpoint when resuming."},
    )
    load_environ_meter_state: bool = field(
        default=True,
        metadata={"help": "Whether to restore EnvironMeter state from checkpoint when resuming."},
    )
    load_rng_state: bool = field(
        default=True,
        metadata={"help": "Whether to restore torch RNG state from checkpoint when resuming."},
    )
    include_frozen_params_in_optimizer: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, build optimizer param_groups from all parameters (including requires_grad=False). "
                "This keeps optimizer param_groups stable across freeze/unfreeze stages while frozen params "
                "still do not get gradients or optimizer states."
            )
        },
    )

@dataclass
class CPTArguments(ExtendedTrainingArguments):
    freeze_pattern: Optional[str] = field(
        default=None,
        metadata={"help": "freeze parameters by pattern."},
    )
    hsa_topk_decay_start: Optional[int] = field(
        default=None,
        metadata={"help": "Runtime HSA topk at the beginning of training. Disabled when unset."},
    )
    hsa_topk_decay_end: Optional[int] = field(
        default=None,
        metadata={"help": "Runtime HSA topk after decay. Disabled when unset."},
    )
    hsa_topk_decay_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Number of global optimizer steps used to decay HSA topk. "
                "If <= 0, use the full training horizon."
            )
        },
    )
    hsa_topk_decay_granularity: int = field(
        default=1,
        metadata={
            "help": (
                "Round scheduled runtime HSA topk to this multiple. Larger values reduce "
                "the number of distinct compiled topk kernels."
            )
        },
    )
    hsa_topk_decay_start_step: int = field(
        default=0,
        metadata={"help": "Global step offset before HSA topk decay starts."},
    )


@dataclass
class SFTArguments(TrainingArguments):
    chunk_size: int = field(
        default=64,
        metadata={"help": "chunk size"},
    )
