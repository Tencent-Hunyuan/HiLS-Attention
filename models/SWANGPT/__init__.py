from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY

@MODELING_REGISTRY.register('swangpt')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_swan_gpt import SWANGPTForCausalLM, SWANGPTModel

    if "ForCausalLM" in architecture:
        return SWANGPTForCausalLM
    elif "Model" in architecture:
        return SWANGPTModel
    else:
        return SWANGPTForCausalLM

@MODEL_CONFIG_REGISTRY.register('swangpt')
def register_swangpt_config():
    from transformers import Qwen3Config
    return Qwen3Config
