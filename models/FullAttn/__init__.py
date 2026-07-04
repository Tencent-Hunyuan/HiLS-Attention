from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY


@MODELING_REGISTRY.register('fullattn')
def register_fullattn_modeling(architecture: str):
    from .modeling_fullattn import FullAttnForCausalLM, FullAttnModel

    if "ForCausalLM" in architecture:
        return FullAttnForCausalLM
    elif "Model" in architecture:
        return FullAttnModel
    else:
        return FullAttnForCausalLM


@MODEL_CONFIG_REGISTRY.register('fullattn')
def register_fullattn_config():
    from transformers import Qwen3Config
    return Qwen3Config
