import sys
from transformers import AutoConfig, AutoModelForCausalLM
from models.FlashHiLS.configuration_hils import HiLSConfig

from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM


def _pop_cli_arg(name: str):
    if name not in sys.argv:
        return None
    i = sys.argv.index(name)
    if i + 1 >= len(sys.argv):
        raise ValueError(f"{name} is missing an argument value")
    value = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    return value


_pop_cli_arg("--hils-config")

HiLSConfig.model_type = "olmo_hils"
AutoConfig.register("olmo_hils", HiLSConfig)
HiLSForCausalLM.config_class = HiLSConfig
AutoModelForCausalLM.register(HiLSConfig, HiLSForCausalLM)

from opencompass.cli.main import main

if __name__ == '__main__':
    main()
