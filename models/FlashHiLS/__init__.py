from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY


def _select_model_class(architecture: str, model_cls, causal_lm_cls):
    if "ForCausalLM" in architecture:
        return causal_lm_cls
    if "Model" in architecture:
        return model_cls
    return causal_lm_cls


@MODELING_REGISTRY.register('qwen_lhsa')
def register_qwen_lhsa_modeling(architecture: str):
    from .modeling_qwen_hils import HiLSForCausalLM, HiLSModel

    return _select_model_class(architecture, HiLSModel, HiLSForCausalLM)


@MODELING_REGISTRY.register('olmo_lhsa')
def register_olmo_lhsa_modeling(architecture: str):
    from .modeling_olmo_hils import HiLSForCausalLM, HiLSModel

    return _select_model_class(architecture, HiLSModel, HiLSForCausalLM)


@MODEL_CONFIG_REGISTRY.register('qwen_lhsa')
@MODEL_CONFIG_REGISTRY.register('olmo_lhsa')
def register_flash_hsa_config():
    from .configuration_hsa import HSAConfig

    return HSAConfig
