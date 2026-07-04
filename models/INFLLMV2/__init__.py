from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY


def _select_model_class(architecture: str, model_cls, causal_lm_cls):
    if "ForCausalLM" in architecture:
        return causal_lm_cls
    if "Model" in architecture:
        return model_cls
    return causal_lm_cls


@MODELING_REGISTRY.register('infllmv2')
def register_infllmv2_modeling(architecture: str):
    from .modeling_infllmv2 import InfLLMv2ForCausalLM, InfLLMv2Model

    return _select_model_class(architecture, InfLLMv2Model, InfLLMv2ForCausalLM)


@MODEL_CONFIG_REGISTRY.register('infllmv2')
def register_infllmv2_config():
    from transformers import Qwen3Config

    class InfLLMv2Config(Qwen3Config):
        model_type = 'infllmv2'

    return InfLLMv2Config
