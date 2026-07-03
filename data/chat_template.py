"""
OLMo3 Chat Template for VeOmni CHAT_TEMPLATE_REGISTRY.

Implements the same logic as open-instruct's "olmo" Jinja2 chat template,
but as a Python class compatible with VeOmni's ChatTemplate interface.

Handles:
  - system / user / assistant / environment roles
  - assistant messages with content=None (tool-call-only turns)
  - function_calls field in assistant messages
  - functions field in system/user messages
  - loss mask: only assistant message *content* (not role tags) is trainable
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from veomni.data.chat_template import CHAT_TEMPLATE_REGISTRY, ChatTemplate
from veomni.data.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are OLMo, a helpful function-calling AI assistant built by Ai2. "
    "Your date cutoff is November 2024, and your model weights are available "
    "at https://huggingface.co/allenai. "
    "You do not currently have access to any functions. "
    "<functions></functions>"
)


@CHAT_TEMPLATE_REGISTRY.register("olmo3_sft")
class OLMo3Template(ChatTemplate):
    """
    Replicates open-instruct's ``olmo`` chat template.

    Format (per message):
        system:      <|im_start|>system\n{content}[ <functions>{functions}</functions>]<|im_end|>\n
        user:        <|im_start|>user\n{content}[\n<functions>{functions}</functions>]<|im_end|>\n
        assistant:   <|im_start|>assistant\n[{content}][<function_calls>{fc}</function_calls>]<|im_end|>\n
                     (last assistant message ends with eos_token instead of <|im_end|>\n)
        environment: <|im_start|>environment\n{content}<|im_end|>\n

    Loss mask:
        Only assistant message body tokens are trainable (labels = token_ids).
        Everything else (system, user, environment, role tags, eos) is masked (labels = -100).
        Within assistant messages, the ``<|im_start|>assistant\n`` prefix is also masked,
        so only the actual content + function_calls are trained on.
    """

    def encode_messages(
        self,
        messages: Sequence[Dict[str, str]],
        max_seq_len: int = 8192,
    ) -> Dict[str, List[int]]:
        input_ids: List[int] = []
        attention_mask: List[int] = []
        labels: List[int] = []

        has_system = any(m.get("role") == "system" for m in messages)

        # If no system message, prepend a default one (matching open-instruct olmo template)
        if not has_system:
            default_sys = "<|im_start|>system\n" + DEFAULT_SYSTEM_PROMPT + "<|im_end|>\n"
            ids = self.tokenizer.encode(default_sys, add_special_tokens=False)
            input_ids += ids
            attention_mask += [1] * len(ids)
            labels += [IGNORE_INDEX] * len(ids)

        num_messages = len(messages)
        for idx, message in enumerate(messages):
            role = message.get("role", "")
            content = message.get("content", None) or ""
            is_last = idx == num_messages - 1

            if role == "system":
                # <|im_start|>system\n{content}
                prefix = "<|im_start|>system\n" + content
                functions = message.get("functions", None)
                if functions is not None:
                    prefix += " <functions>" + functions + "</functions>"
                else:
                    prefix += " You do not currently have access to any functions. <functions></functions>"
                prefix += "<|im_end|>\n"

                ids = self.tokenizer.encode(prefix, add_special_tokens=False)
                input_ids += ids
                attention_mask += [1] * len(ids)
                labels += [IGNORE_INDEX] * len(ids)

            elif role == "user":
                functions = message.get("functions", None)
                if functions is not None:
                    text = "<|im_start|>user\n" + content + "\n<functions>" + functions + "</functions><|im_end|>\n"
                else:
                    text = "<|im_start|>user\n" + content + "<|im_end|>\n"

                ids = self.tokenizer.encode(text, add_special_tokens=False)
                input_ids += ids
                attention_mask += [1] * len(ids)
                labels += [IGNORE_INDEX] * len(ids)

            elif role == "assistant":
                # --- role prefix (masked) ---
                role_prefix = "<|im_start|>assistant\n"
                prefix_ids = self.tokenizer.encode(role_prefix, add_special_tokens=False)
                input_ids += prefix_ids
                attention_mask += [1] * len(prefix_ids)
                labels += [IGNORE_INDEX] * len(prefix_ids)

                # --- body: content + function_calls (trainable) ---
                body = ""
                if content:
                    body += content
                fc = message.get("function_calls", None)
                if fc is not None:
                    body += "<function_calls>" + fc + "</function_calls>"

                if body:
                    body_ids = self.tokenizer.encode(body, add_special_tokens=False)
                    input_ids += body_ids
                    attention_mask += [1] * len(body_ids)
                    labels += body_ids  # trainable

                # --- suffix (masked) ---
                if not is_last:
                    suffix = "<|im_end|>\n"
                else:
                    suffix = self.tokenizer.eos_token

                suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
                input_ids += suffix_ids
                attention_mask += [1] * len(suffix_ids)
                labels += [IGNORE_INDEX] * len(suffix_ids)

            elif role == "environment":
                text = "<|im_start|>environment\n" + content + "<|im_end|>\n"
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                input_ids += ids
                attention_mask += [1] * len(ids)
                labels += [IGNORE_INDEX] * len(ids)

            else:
                # unknown role — treat as masked
                text = "<|im_start|>" + role + "\n" + content + "<|im_end|>\n"
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                input_ids += ids
                attention_mask += [1] * len(ids)
                labels += [IGNORE_INDEX] * len(ids)

        # Truncate from left to max_seq_len (keep the most recent context)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        model_inputs = {k: v[-max_seq_len:] for k, v in model_inputs.items()}
        return model_inputs

    def get_jinja_template(self) -> str:
        """Return the Jinja2 template string (same as open-instruct's olmo template)."""
        return (
            "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
            "{% if not has_system %}"
            "{{ '<|im_start|>system\\nYou are OLMo, a helpful function-calling AI assistant built by Ai2. "
            "Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. "
            "You do not currently have access to any functions. <functions></functions><|im_end|>\\n' }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|im_start|>system\\n' + message['content'] }}"
            "{% if message.get('functions', none) is not none %}"
            "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\\n' }}"
            "{% else %}"
            "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\\n' }}"
            "{% endif %}"
            "{% elif message['role'] == 'user' %}"
            "{% if message.get('functions', none) is not none %}"
            "{{ '<|im_start|>user\\n' + message['content'] + '\\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\\n' }}"
            "{% else %}"
            "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% endif %}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% if message.get('content', none) is not none %}"
            "{{ message['content'] }}"
            "{% endif %}"
            "{% if message.get('function_calls', none) is not none %}"
            "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
            "{% endif %}"
            "{% if not loop.last %}"
            "{{ '<|im_end|>' + '\\n' }}"
            "{% else %}"
            "{{ eos_token }}"
            "{% endif %}"
            "{% elif message['role'] == 'environment' %}"
            "{{ '<|im_start|>environment\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% endif %}"
            "{% if loop.last and add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )
