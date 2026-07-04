from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY


@MODELING_REGISTRY.register('swan_nsa')
def register_swan_nsa_modeling(architecture: str):
    from .modeling_swan_nsa import SWANNSAForCausalLM, SWANNSAModel
    
    if "ForCausalLM" in architecture:
        return SWANNSAForCausalLM
    elif "Model" in architecture:
        return SWANNSAModel
    else:
        return SWANNSAForCausalLM

@MODEL_CONFIG_REGISTRY.register('swan_nsa')
def register_swan_nsa_config():
    from transformers import Qwen3Config
    return Qwen3Config




@MODELING_REGISTRY.register('hybrid_nsa')
def register_hybrid_nsa_modeling(architecture: str):
    from .modeling_nsa_innerx_ultra import HybridNSAForCausalLM, HybridNSAModel
    
    if "ForCausalLM" in architecture:
        return HybridNSAForCausalLM
    elif "Model" in architecture:
        return HybridNSAModel
    else:
        return HybridNSAForCausalLM

@MODEL_CONFIG_REGISTRY.register('hybrid_nsa')
def register_hybrid_nsa_config():
    from transformers import Qwen3Config
    return Qwen3Config
