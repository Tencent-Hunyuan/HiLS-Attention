import json
import os
import sys
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config
from models.FlashHiLS.configuration_hsa import HSAConfig
def _pop_cli_arg(name: str):
    if name not in sys.argv:
        return None
    i = sys.argv.index(name)
    if i + 1 >= len(sys.argv):
        raise ValueError(f"{name} 缺少参数值")
    value = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    return value


def _peek_cli_arg(name: str):
    if name not in sys.argv:
        return None
    i = sys.argv.index(name)
    if i + 1 >= len(sys.argv):
        raise ValueError(f"{name} 缺少参数值")
    return sys.argv[i + 1]


def _resolve_model_type(config_path=None, checkpoint_path=None):
    model_type = ""
    path = config_path or (os.path.join(checkpoint_path, "config.json") if checkpoint_path else None)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            model_type = json.load(f).get("model_type", "")
    return model_type


def resolve_hsa_class(config_path=None, checkpoint_path=None):
    model_type = _resolve_model_type(config_path, checkpoint_path)
    if "lhsa" in model_type or "hsa" in model_type:
        if "olmo" in model_type:
            from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM
            return HiLSForCausalLM, model_type
        if "qwen" in model_type:
            from models.FlashHiLS.modeling_qwen_hils import HiLSForCausalLM
            return HiLSForCausalLM, model_type
        # Generic HSA fallback (legacy): qwen-based HiLSForCausalLM under the
        # literal "flash_hsa" tag.
        from models.FlashHiLS.modeling_qwen_hils import HiLSForCausalLM
        return HiLSForCausalLM, model_type or "flash_hsa"
    # Non-HSA checkpoint (e.g. stock Olmo3 "olmo3", plain Qwen, etc.).
    # Transformers' builtin AutoModel classes handle it; no HSA registration
    # needed. Return None to signal "skip HSA registration".
    return None, None


hsa_config_path = _pop_cli_arg("--hsa-config")
hf_path = _peek_cli_arg("--hf-path")
_hsa_resolved = resolve_hsa_class(hsa_config_path, hf_path)
if _hsa_resolved[0] is not None:
    HiLSForCausalLM, model_type = _hsa_resolved


    class OpenCompassHSAConfig(HSAConfig):
        model_type = model_type


    AutoConfig.register(model_type, OpenCompassHSAConfig, exist_ok=True)
    HiLSForCausalLM.config_class = OpenCompassHSAConfig
    AutoModelForCausalLM.register(OpenCompassHSAConfig, HiLSForCausalLM, exist_ok=True)

from models.FullAttn.modeling_fullattn import FullAttnForCausalLM


class OpenCompassFullAttnConfig(Qwen3Config):
    model_type = "fullattn"


AutoConfig.register("fullattn", OpenCompassFullAttnConfig, exist_ok=True)
FullAttnForCausalLM.config_class = OpenCompassFullAttnConfig
AutoModelForCausalLM.register(OpenCompassFullAttnConfig, FullAttnForCausalLM, exist_ok=True)

def _import_opencompass_main():
    try:
        from opencompass.cli.main import main as oc_main
        return oc_main
    except ModuleNotFoundError as exc:
        opencompass_path = os.environ.get("OPENCOMPASS_PATH")
        if opencompass_path:
            opencompass_path = os.path.abspath(opencompass_path)
            if os.path.isdir(os.path.join(opencompass_path, "opencompass")) and opencompass_path not in sys.path:
                sys.path.insert(0, opencompass_path)
                from opencompass.cli.main import main as oc_main
                return oc_main

        raise ModuleNotFoundError(
            "Cannot import `opencompass`."
            "Please install OpenCompass, or set "
            "`OPENCOMPASS_PATH=/path/to/opencompass_repo`."
        ) from exc


def _setup_local_dataset_configs():
    """Make local eval/configs/datasets/ usable by opencompass CLI.

    The local dir hosts self-contained `*_gen.py` configs (gsm8k_gen,
    math_gen, cmath_gen, humaneval_plus_gen, mbpp_plus_gen, cruxeval_o_gen,
    ...) that rely on a top-level `custom_datasets` module for custom
    Dataset/Evaluator classes. We:

    1. Put `eval/configs/datasets/` on sys.path so `from custom_datasets
       import ...` works when mmengine lazy-loads the config.
    2. Pre-import `custom_datasets` so that its `@LOAD_DATASET` /
       `@TEXT_POSTPROCESSORS` decorators run BEFORE mmengine parses the
       config (mmengine's LazyObject cannot execute decorator calls).
    3. Auto-inject `--config-dir <eval/configs>` into argv when the user
       hasn't provided one, so opencompass searches our local dataset
       configs first and falls back to the builtin ones otherwise.
    """
    local_datasets_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs", "datasets")
    if not os.path.isdir(local_datasets_root):
        return

    if local_datasets_root not in sys.path:
        sys.path.insert(0, local_datasets_root)

    custom_path = os.path.join(local_datasets_root, "custom_datasets.py")
    if os.path.isfile(custom_path):
        try:
            import custom_datasets  # noqa: F401
            print(f"[eval_opencompass] Pre-imported custom_datasets from {custom_path}")
        except Exception as e:
            print(f"[eval_opencompass] WARNING: failed to pre-import "
                  f"custom_datasets: {e}", file=sys.stderr)

    # Inject --config-dir so opencompass searches eval/configs/ first.
    if "--config-dir" not in sys.argv:
        local_config_root = os.path.dirname(local_datasets_root)
        sys.argv += ["--config-dir", local_config_root]
        print(f"[eval_opencompass] Injected --config-dir {local_config_root}")


_setup_local_dataset_configs()

main = _import_opencompass_main()

if __name__ == '__main__':
    main()
