from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY


@MODELING_REGISTRY.register('hsa_swa')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_swa_hsa_rope import DRTForCausalLM, DRTModel

    if "ForCausalLM" in architecture:
        return DRTForCausalLM
    elif "Model" in architecture:
        return DRTModel
    else:
        return DRTForCausalLM

@MODEL_CONFIG_REGISTRY.register('hsa_swa')
def register_hsa_swa_config():
    from .configuration_hsa_swa import HSASWAConfig
    return HSASWAConfig
