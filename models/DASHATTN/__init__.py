from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY


def _select_model_class(architecture: str, model_cls, causal_lm_cls):
    if "ForCausalLM" in architecture:
        return causal_lm_cls
    if "Model" in architecture:
        return model_cls
    return causal_lm_cls


@MODELING_REGISTRY.register('dash_attn')
def register_dash_attn_modeling(architecture: str):
    from .modeling_dash_attn import DashAttnForCausalLM, DashAttnModel

    return _select_model_class(architecture, DashAttnModel, DashAttnForCausalLM)


@MODEL_CONFIG_REGISTRY.register('dash_attn')
def register_dash_attn_config():
    from transformers import Qwen3Config

    class DashAttnConfig(Qwen3Config):
        model_type = 'dash_attn'

    return DashAttnConfig
