import sys
from transformers import AutoConfig, AutoModelForCausalLM
from models.FlashHiLS.configuration_hils import HSAConfig

from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM


def _pop_cli_arg(name: str):
    """从 sys.argv 中提取并移除指定参数及其值，避免传递给 opencompass argparse。"""
    if name not in sys.argv:
        return None
    i = sys.argv.index(name)
    if i + 1 >= len(sys.argv):
        raise ValueError(f"{name} 缺少参数值")
    value = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    return value


# 剥离 --hsa-config 参数（shell 脚本会传入，但 opencompass 不识别）
_pop_cli_arg("--hsa-config")

HSAConfig.model_type = "olmo_hils"
AutoConfig.register("olmo_hils", HSAConfig)
HiLSForCausalLM.config_class = HSAConfig
AutoModelForCausalLM.register(HSAConfig, HiLSForCausalLM)

from opencompass.cli.main import main

if __name__ == '__main__':
    main()
