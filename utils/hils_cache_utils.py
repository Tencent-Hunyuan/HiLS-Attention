import torch
from transformers.cache_utils import CacheLayerMixin

_OFFLOAD_SEQ_THRESHOLD = 128 * 1024
_OFFLOAD_HIDDEN_DIM_THRESHOLD = 4096


class HiLSDynamicLayer(CacheLayerMixin):
    is_sliding = False

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True
        self._hidden_dim = key_states.shape[2] * key_states.shape[3]
        self._offloading = False

    def _should_offload(self, new_seq_len: int) -> bool:
        return (
            self._hidden_dim >= _OFFLOAD_HIDDEN_DIM_THRESHOLD
            and new_seq_len > _OFFLOAD_SEQ_THRESHOLD
        )

    def _move_to_cpu(self) -> None:
        if self.keys.numel() > 0:
            self.keys = self.keys.to("cpu", non_blocking=True).pin_memory()
            self.values = self.values.to("cpu", non_blocking=True).pin_memory()
        else:
            self.keys = torch.tensor([], dtype=self.dtype, device="cpu").pin_memory()
            self.values = torch.tensor([], dtype=self.dtype, device="cpu").pin_memory()
        self._offloading = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        new_total_seq_len = self.get_seq_length() + key_states.shape[1]

        if not self._offloading and self._should_offload(new_total_seq_len):
            self._move_to_cpu()

        if self._offloading:
            key_states_cpu = key_states.to("cpu", non_blocking=True)
            value_states_cpu = value_states.to("cpu", non_blocking=True)
            torch.cuda.current_stream().synchronize()

            if self.keys.numel() > 0:
                self.keys = torch.cat([self.keys, key_states_cpu], dim=1).pin_memory()
                self.values = torch.cat([self.values, value_states_cpu], dim=1).pin_memory()
            else:
                self.keys = key_states_cpu.pin_memory()
                self.values = value_states_cpu.pin_memory()

            keys_gpu = self.keys.to(self.device, non_blocking=True)
            values_gpu = self.values.to(self.device, non_blocking=True)
            return keys_gpu, values_gpu
        else:
            self.keys = torch.cat([self.keys, key_states], dim=1)
            self.values = torch.cat([self.values, value_states], dim=1)
            return self.keys, self.values

    def get_seq_length(self) -> int:
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[1]

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        kv_offset = 0
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_max_cache_shape(self) -> int:
        return -1
