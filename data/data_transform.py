from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union


if TYPE_CHECKING:
    from .chat_template import ChatTemplate

import torch
import random
import numpy as np
import string


def process_numpy_example(
    example,
):
    data_dict = {
        "input_ids": torch.tensor(example),
        "attention_mask": torch.tensor([1] * example.shape[0]),
        "labels": torch.tensor(example),
    }
    return [data_dict]


_ORDINAL_WORDS = [
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth",
]


def _ordinal(n: int) -> str:
    """1-based ordinal English word, fallback to '{n}-th'."""
    if 1 <= n <= len(_ORDINAL_WORDS):
        return _ORDINAL_WORDS[n - 1]
    return f"{n}-th"


class RulerSynthesizer:
    def __init__(self, tokenizer, vocab_low = 100, task_id=-1, enable_ruler_plus=False, **kwargs):
        self.tokenizer = tokenizer
        self._low = vocab_low
        self._high = tokenizer.vocab_size
        self._eos_id = tokenizer.eos_token_id

        self._s_niah_needle_ids = \
            self.tokenizer.encode(' |One of the special magic numbers for long-context is:')
        self._s_niah_end = self.tokenizer.encode('|')
        self._s_niah_question = \
            self.tokenizer.encode(' What is the special magic number for long-context mentioned in the provided text? Answer: ')

        self._vt_question = \
            self.tokenizer.encode(' Find all variables that are assigned the value ')
        self._vt_question_answer = self.tokenizer.encode('. Answer: ')

        self._mq_template = '| One of the special magic numbers for {} is {}.|'
        self._mq_question = ' What are all the special magic numbers for {} mentioned in the provided text?'
        self._mq_answer = self.tokenizer.encode('. Answer: ')

        self._fwe_tempalte = "[INST] Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text? [/INST] Answer: According to the coded text above, the three most frequently appeared words are: "
        # self._fwe_wo_prefix_tempalte = "{context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text? [/INST] Answer: According to the coded text above, the three most frequently appeared words are:"

        # ----- Ruler-Plus: positional multi-value lookup tasks -----
        # Same template used for both PMVL (task_id=4) and PCVL (task_id=5):
        #   |KEY: VALUE|
        # The same KEY may appear multiple times with different VALUEs;
        # the model must locate the N-th occurrence (and optionally
        # extract the M-th character of that value).
        self._pml_template = '|{key}: {val}|'
        self._pml_question = (
            ' What is the {ord_n} value for {key} mentioned in the provided text? Answer: '
        )
        self._pcl_question = (
            ' What is the {ord_m} character of the {ord_n} value for {key}'
            ' mentioned in the provided text? Answer: '
        )
        self._pml_answer_end = self.tokenizer.encode('.')

        self.task_id = task_id
        self.enable_ruler_plus = enable_ruler_plus
        self.kwargs = kwargs

    def generate_single_niah(self, inputs, length=7):
        # print(f'needle len: {length}')
        rng = random.Random(int(inputs[0] % 1000) * int(inputs[-1] % 1000))
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
        rand_val = int(rng.randint(10**length, 10**(length+1)))
        passkey_ids = self.tokenizer.encode(f'{rand_val}')
        # passkey_ids = np.array(rng.randint(self._low, self._high, size=length))

        passkey_ids_ = np.concatenate((self._s_niah_needle_ids, passkey_ids, self._s_niah_end))
        # print(passkey_ids_)
        # if self._chunk_win_size == -1:
        passkey_len = len(passkey_ids) + 1
        prompt_ids = np.concatenate((self._s_niah_question, passkey_ids, [self._eos_id]))
        
        total_len = len(inputs)
        start = rng.randint(len(inputs) - len(passkey_ids_) - len(prompt_ids))  # locate at the first sentence
        new_array = np.insert(inputs, start, passkey_ids_)
        new_array = np.insert(new_array, total_len - len(prompt_ids), prompt_ids)
        new_array = new_array[:total_len]

        return new_array, new_array[:-passkey_len], new_array[-passkey_len:]


    def _insert_needles_into_ids(self, input_ids, needles, rng):
        org_len = len(input_ids)
        numbers = list(range(0, org_len))
        rng.shuffle(numbers)
        indices = numbers[:len(needles)]
        for idx in range(len(needles)):
            input_ids = np.insert(input_ids, indices[idx], needles[idx])
            for j in range(idx + 1, len(needles)):
                if indices[j] > indices[idx]:
                    indices[j] += len(needles[idx])
        return input_ids


    def generate_variable_tracking(self, inputs, total_var=6, max_hops=2, varlen=7, **kwargs):
        rng = random.Random(int(inputs[0] % 10000))
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
        def generate_random_variable_name():
            letters = [rng.choice(list(string.ascii_uppercase)) for _ in range(5)]
            return ''.join(letters)

        answer_num = total_var // (max_hops + 1)
        answers = []
        rng = random.Random(int(inputs[-1] % 10000))
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
        while len(answers) < answer_num:
            rand_val = rng.randint(10**varlen, 10**(varlen+1) - 1)
            if f'{rand_val}' not in answers:
                answers.append(f'{rand_val}')

        
        var_names = []
        while len(var_names) < total_var:
            new_var_name = generate_random_variable_name()
            if new_var_name not in var_names:
                var_names.append(new_var_name)

        
        assignments = []
        for i in range(0, total_var, max_hops + 1):
            assignment1 = self.tokenizer.encode(f'|VAR {var_names[i]} = {answers[i // (max_hops + 1)]}|')
            assignment2 = self.tokenizer.encode(f'|VAR {var_names[i + 1]} = {var_names[i]}|')
            assignment3 = self.tokenizer.encode(f'|VAR {var_names[i + 2]} = {var_names[i + 1]}|')
            assignments.append(assignment1)
            assignments.append(assignment2)
            assignments.append(assignment3)

        needle_len = sum([len(ids) for ids in assignments])
        rng.shuffle(assignments)
        
        question_ids = self._vt_question + self.tokenizer.encode(answers[0])
        # answer_ids = self._vt_question_answer + self.tokenizer.encode(var_names[0]) + self.tokenizer
        answer = ', '.join(var_names[:max_hops + 1])
        _answer_ids = self.tokenizer.encode(answer)
        answer_ids = np.concatenate((self._vt_question_answer, _answer_ids, [self._eos_id]))
        input_ids_trunc = inputs[:-len(answer_ids) - len(question_ids)]
        input_ids_trunc = input_ids_trunc[:-needle_len]
        input_with_needle = self._insert_needles_into_ids(input_ids_trunc, assignments, rng)

        new_ids = np.concatenate((input_with_needle, question_ids, answer_ids))

        return new_ids, new_ids[:-len(_answer_ids) - 1], new_ids[-len(_answer_ids) - 1:]


    def generate_multi_query(self, inputs, total_var=6, var_name_len = 5, var_len=5, num_queries=2, **kwargs):
        var_names = []
        var_vals = []
        rng = random.Random(int(inputs[0] % 10000))
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
        while len(var_names) < total_var:
            var_ids = ''.join([rng.choice(list(string.ascii_uppercase)) for _ in range(var_name_len)])
            if var_ids not in var_names:
                var_names.append(var_ids)
        rng = random.Random(int(inputs[-1] % 10000))
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
        while len(var_vals) < total_var:
            rand_val = rng.randint(10**var_len, 10**(var_len+1) - 1)
            if f'{rand_val}' not in var_vals:
                var_vals.append(f'{rand_val}')

        needles = []
        for needle_i in range(total_var):
            needles.append(self.tokenizer.encode(self._mq_template.format(var_names[needle_i], var_vals[needle_i])))
    
        rng.shuffle(needles)
        question_vals = ' and '.join(var_names[:num_queries])
        question = self._mq_question.format(question_vals)
        question_ids = self.tokenizer.encode(question)

        answer = ' '.join(var_vals[:num_queries])
        _answer_ids = self.tokenizer.encode(answer)
        answer_ids = np.concatenate((self._mq_answer, _answer_ids, [self._eos_id]))

        needle_len = sum([len(ids) for ids in needles])
        
        input_ids_trunc = inputs[:-len(answer_ids) - len(question_ids)]
        input_ids_trunc = input_ids_trunc[:-needle_len]
        input_with_needle = self._insert_needles_into_ids(input_ids_trunc, needles, rng)

        new_ids = np.concatenate((input_with_needle, question_ids, answer_ids))

        return new_ids, new_ids[:-len(_answer_ids) - 1], new_ids[-len(_answer_ids) - 1:]

    def generate_frequent_words_extraction(self, inputs, var_len=5, alpha=2.0, vocab_size=2000, num_words=-1, incremental=10, **kwargs):
        # generate vocab
        from scipy.special import zeta
        rng = random.Random(int(inputs[0] % 1000) * int(inputs[-1] % 1000))
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        vocab = [''.join([rng.choice(list(string.ascii_lowercase)) for _ in range(var_len)]) for _ in range(vocab_size)]
        while len(set(vocab)) < vocab_size:
            vocab.append(''.join([rng.choice(list(string.ascii_lowercase)) for _ in range(var_len)]))
        vocab = sorted(list(set(vocab)))
        rng.shuffle(vocab)
        vocab[0] = '...' # treat the top ranked as noise

        # sample words
        template = self._fwe_tempalte
        def gen_text(num_words):
            k = np.arange(1, len(vocab)+1)
            sampled_cnt = num_words*(k**-alpha)/zeta(alpha)
            sampled_words = [[w] * zi for w, zi in zip(vocab, sampled_cnt.astype(int))]
            sampled_words = [x for wlst in sampled_words for x in wlst]
            rng.shuffle(sampled_words)
            return template.format(context=' '.join(sampled_words), query=''), vocab[1:4]
        
        max_len = len(inputs)
        if num_words > 0:
            num_words = num_words
            text, answer = gen_text(num_words)
            while len(self.tokenizer.encode(text)) > max_len:
                num_words -= incremental
                text, answer = gen_text(num_words)
        else:
            num_words = max_len // var_len # init
            text, answer = gen_text(num_words)
            while len(self.tokenizer.encode(text + ' '.join(answer))) < max_len - 1:
                # print(f"num_words: {num_words}, current_len: {len(self.tokenizer.encode(text + ' '.join(answer)))}")
                num_words = int(num_words * 1.1)
                text, answer = gen_text(num_words)
            # num_words -= incremental
            num_words = int(num_words / 1.1)
        text, answer = gen_text(num_words)
        new_ids = self.tokenizer.encode(text + ' '.join(answer)) + [self._eos_id]
        answer_len = len(self.tokenizer.encode(' '.join(answer))) + 1

        return new_ids, new_ids[:-answer_len], new_ids[-answer_len:]

    # ------------------------------------------------------------------
    # Ruler-Plus tasks
    # ------------------------------------------------------------------
    def _build_pml_needles(self, inputs,
                           num_keys=4, vals_per_key=4, key_name_len=5,
                           val_len=6, val_charset='digits',
                           target_n=None, **kwargs):
        """Build the shared corpus of |KEY: VALUE| needles for PMVL/PCVL.

        Same KEY can appear multiple times with different VALUEs. Returns:
            needles      : list[list[int]]  encoded |KEY: VAL| strings
            target_key   : str
            target_vals  : list[str] (in *insertion order* for that key)
            n_idx        : int (1-based) which value of `target_key` to ask about
        """
        rng = random.Random(int(inputs[0] % 10000) ^ int(inputs[-1] % 10000))
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        # generate distinct keys
        keys = []
        while len(keys) < num_keys:
            kname = ''.join(rng.choice(list(string.ascii_uppercase), size=key_name_len))
            if kname not in keys:
                keys.append(kname)

        # generate distinct values for each key
        if val_charset == 'digits':
            charset = list(string.digits)
        elif val_charset == 'lower':
            charset = list(string.ascii_lowercase)
        else:
            charset = list(string.ascii_letters + string.digits)

        all_vals_set = set()
        key_to_vals = {}  # key -> list[str] in *insertion order*
        for k in keys:
            vals_for_k = []
            while len(vals_for_k) < vals_per_key:
                v = ''.join(rng.choice(charset, size=val_len))
                if v not in all_vals_set:
                    vals_for_k.append(v)
                    all_vals_set.add(v)
            key_to_vals[k] = vals_for_k

        # decide which key & which n-th value to query
        target_key = keys[rng.randint(0, num_keys)]
        if target_n is None:
            n_idx = int(rng.randint(0, vals_per_key)) + 1  # 1-based
        else:
            n_idx = max(1, min(int(target_n), vals_per_key))

        # build (key, val) pairs preserving per-key insertion order, then shuffle
        # them globally for placement; ordering of same-key pairs is preserved
        # by indexing them with (key, position-within-key).
        flat_pairs = []
        for k in keys:
            for pos, v in enumerate(key_to_vals[k]):
                flat_pairs.append((k, pos, v))
        # global shuffle of placement order
        order = list(range(len(flat_pairs)))
        rng.shuffle(order)

        # Re-assign pos within target_key according to actual placement order
        # so that "the n-th value for KEY" matches the order they appear in text.
        target_appearance_order = [i for i in order if flat_pairs[i][0] == target_key]
        target_vals_in_order = [flat_pairs[i][2] for i in target_appearance_order]

        needles = []
        for i in order:
            k, _pos, v = flat_pairs[i]
            needles.append(self.tokenizer.encode(self._pml_template.format(key=k, val=v)))

        return needles, target_key, target_vals_in_order, n_idx, rng


    def generate_positional_multi_value_lookup(self, inputs, **kwargs):
        """Task 4 (Ruler-Plus PMVL): find the n-th value for KEY in text.

        Inserts |KEY: VAL| needles where the same KEY may appear multiple
        times. Asks: "What is the {n}-th value for KEY?" Answer is the
        n-th VAL of that KEY in *appearance order*.
        """
        needles, target_key, target_vals, n_idx, rng = self._build_pml_needles(inputs, **kwargs)
        target_val = target_vals[n_idx - 1]

        question = self._pml_question.format(ord_n=_ordinal(n_idx), key=target_key)
        question_ids = self.tokenizer.encode(question)
        _answer_ids = self.tokenizer.encode(target_val)
        answer_ids = np.concatenate((_answer_ids, self._pml_answer_end, [self._eos_id]))

        needle_len = sum(len(ids) for ids in needles)
        input_ids_trunc = inputs[:-len(answer_ids) - len(question_ids)]
        input_ids_trunc = input_ids_trunc[:-needle_len]
        input_with_needle = self._insert_needles_into_ids(input_ids_trunc, needles, rng)

        new_ids = np.concatenate((input_with_needle, question_ids, answer_ids))
        # answer span is [VAL] + '.' + EOS (last len(_answer_ids)+2 tokens)
        ans_span = len(_answer_ids) + 2
        return new_ids, new_ids[:-ans_span], new_ids[-ans_span:]


    def generate_positional_char_in_value_lookup(self, inputs,
                                                  target_m=None, **kwargs):
        """Task 5 (Ruler-Plus PCVL): find the m-th char of the n-th value for KEY.

        Same needle construction as PMVL, but the question further asks for
        a single character at a specific position within the located VAL.
        """
        # default to numeric values so every "character" is a digit
        kwargs.setdefault('val_charset', 'digits')
        kwargs.setdefault('val_len', 7)
        needles, target_key, target_vals, n_idx, rng = self._build_pml_needles(inputs, **kwargs)
        target_val = target_vals[n_idx - 1]

        val_len = len(target_val)
        if target_m is None:
            m_idx = int(rng.randint(0, val_len)) + 1  # 1-based
        else:
            m_idx = max(1, min(int(target_m), val_len))
        target_char = target_val[m_idx - 1]

        question = self._pcl_question.format(
            ord_m=_ordinal(m_idx), ord_n=_ordinal(n_idx), key=target_key,
        )
        question_ids = self.tokenizer.encode(question)
        _answer_ids = self.tokenizer.encode(target_char)
        answer_ids = np.concatenate((_answer_ids, self._pml_answer_end, [self._eos_id]))

        needle_len = sum(len(ids) for ids in needles)
        input_ids_trunc = inputs[:-len(answer_ids) - len(question_ids)]
        input_ids_trunc = input_ids_trunc[:-needle_len]
        input_with_needle = self._insert_needles_into_ids(input_ids_trunc, needles, rng)

        new_ids = np.concatenate((input_with_needle, question_ids, answer_ids))
        ans_span = len(_answer_ids) + 2
        return new_ids, new_ids[:-ans_span], new_ids[-ans_span:]

    def single_token_eval_collate_fn(self, samples):
        chunk_ids_list = []
        ground_truth = []
        for _, ids in enumerate(samples):
            if self.task_id == 0:
                _, q, a = self.generate_single_niah(ids, **self.kwargs)
            elif self.task_id == 1:
                _, q, a = self.generate_multi_query(ids, **self.kwargs)
            elif self.task_id == 2:
                _, q, a = self.generate_variable_tracking(ids, **self.kwargs)
                # 查找分隔符位置，token ID 13 通常是换行符
                # 但不同 tokenizer 可能不同，需要容错处理
                sep_indices = np.argwhere(a == 13).flatten()
                if len(sep_indices) > 0:
                    last_idx = sep_indices[-1]
                    q = np.concatenate([q, a[:last_idx + 1]])
                    a = a[last_idx + 1:]
                # 如果没有找到 token 13，尝试查找其他可能的分隔符（如冒号 58）
                else:
                    pass

                # last_idx = np.argwhere(a == 13).flatten()[-1]
                # q = np.concatenate([q, a[:last_idx + 1]])
                # a = a[last_idx + 1:]
            elif self.task_id == 3:
                _, q, a = self.generate_frequent_words_extraction(ids, **self.kwargs)
            elif self.task_id == 4 and self.enable_ruler_plus:
                _, q, a = self.generate_positional_multi_value_lookup(ids, **self.kwargs)
            elif self.task_id == 5 and self.enable_ruler_plus:
                _, q, a = self.generate_positional_char_in_value_lookup(ids, **self.kwargs)
            
            chunk_ids_list.append(torch.tensor(np.concatenate([q, a])))
            ground_truth.append(torch.tensor(a))
            # chunk_ids_list.append(torch.tensor(q))
            # ground_truth.append(torch.tensor(a[0]))
            # print(self.tokenizer.decode(chunk_ids_list[-1]))
            # print('-' * 20)
            # print(self.tokenizer.decode(a))
            # print('~' * 20)
        return {"input_ids": torch.stack(chunk_ids_list), "labels": torch.stack(ground_truth)}


    def train_collate_fn(self, samples):
        chunk_ids_list = []
        pass_state_ids = []
        final_poses = []
        for group_i, (ids, pass_state) in enumerate(samples):
            # task_id selection: keep legacy semantics, extend if ruler_plus enabled
            #   -1 : random over all enabled tasks (0..3, plus 4..5 if ruler_plus)
            #   -2 : random over 0..2 (legacy)
            #   -3 : ruler_plus-only random over 4..5 (requires enable_ruler_plus)
            #   >=0: explicit task id
            if self.task_id == -1:
                hi = 5 if self.enable_ruler_plus else 3
                task_id = random.randint(0, hi)
            elif self.task_id == -2:
                task_id = random.randint(0, 2)
            elif self.task_id == -3 and self.enable_ruler_plus:
                task_id = random.randint(4, 5)
            else:
                task_id = self.task_id
            # print(f'task_id: {self.task_id}')
            new_ids = None
            if task_id == 0:
                new_ids, q, _ = self.generate_single_niah(ids, **self.kwargs)
            elif task_id == 1:
                new_ids, q, _ = self.generate_multi_query(ids, **self.kwargs)
            elif task_id == 2:
                new_ids, q, _ = self.generate_variable_tracking(ids, **self.kwargs)
            elif task_id == 3:
                new_ids, q, _ = self.generate_frequent_words_extraction(ids, **self.kwargs)
            elif task_id == 4 and self.enable_ruler_plus:
                new_ids, q, _ = self.generate_positional_multi_value_lookup(ids, **self.kwargs)
            elif task_id == 5 and self.enable_ruler_plus:
                new_ids, q, _ = self.generate_positional_char_in_value_lookup(ids, **self.kwargs)
            else:
                raise ValueError(
                    f"Unsupported task_id={task_id} (enable_ruler_plus={self.enable_ruler_plus})"
                )
            final_poses.append(len(q))
            # print(self.tokenizer.decode(new_ids))
            # print('~' * 20)
            # print(self.tokenizer.decode(new_ids[final_poses[-1]:final_poses[-1] + 1])) 
            # print('*' * 20)
            if len(new_ids) < len(ids):
                pad_len = len(ids) - len(new_ids)
                new_ids = np.concatenate((new_ids, [-1] * pad_len))
            chunk_ids_list.append(torch.tensor(new_ids))
            if pass_state:
                pass_state_ids.append(group_i)
        return {"input_ids": torch.stack(chunk_ids_list), 
                "final_pos":final_poses, 
                'pass_init_state': torch.tensor(pass_state_ids, dtype=torch.long)}

def synthesize_ruler_example(
    example: torch.Tensor,
    ruler_synthesizer: RulerSynthesizer,
    params: str
):
    def extract_ratio(s):
        parts = s.split('_')
        if len(parts) == 2:
            return float(parts[1])
        else:
            return 1.0
    ratio = extract_ratio(params)
    # legacy: random over 0..2; if ruler_plus enabled, also include 4 & 5
    if getattr(ruler_synthesizer, 'enable_ruler_plus', False):
        task_id = random.choice([0, 1, 2, 4, 5])
    else:
        task_id = random.randint(0, 2)
    new_ids = None

    rand_val = random.random()
    if rand_val < ratio:
        if task_id == 0:
            new_ids, q, _ = ruler_synthesizer.generate_single_niah(example)
        elif task_id == 1:
            new_ids, q, _ = ruler_synthesizer.generate_multi_query(example)
        elif task_id == 2:
            new_ids, q, _ = ruler_synthesizer.generate_variable_tracking(example)
        elif task_id == 3:
            new_ids, q, _ = ruler_synthesizer.generate_frequent_words_extraction(example)
        elif task_id == 4:
            new_ids, q, _ = ruler_synthesizer.generate_positional_multi_value_lookup(example)
        elif task_id == 5:
            new_ids, q, _ = ruler_synthesizer.generate_positional_char_in_value_lookup(example)
    else:
        new_ids = example
    data_dict = {
        "input_ids": torch.tensor(new_ids),
        "attention_mask": torch.tensor([1] * example.shape[0]),
        "labels": torch.tensor(new_ids),
    }
    return [data_dict]


def process_sft_example_with_lmk(
    example: Dict[str, Any],
    chat_template: "ChatTemplate",
    max_seq_len: int,
    chunk_size: int,
    text_keys: Union[str, List[str]] = "messages",
    source_name: Optional[str] = None,
) -> List[Dict[str, "torch.Tensor"]]:
    if isinstance(text_keys, str):
        text_example = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                text_example = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    # print(text_example)

    tokenized_example = chat_template.encode_messages(text_example, max_seq_len=max_seq_len // chunk_size * (chunk_size - 1))
    tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

    # Pad to multiple of (chunk_size - 1) so that after inserting landmark tokens
    # the sequence length is a multiple of chunk_size.
    # Pad tokens: attention_mask=1 (so rmpad won't strip them), labels=IGNORE_INDEX.
    align = chunk_size - 1  # e.g. 63 when chunk_size=64
    seq_len = tokenized_example["input_ids"].size(0)
    remainder = seq_len % align
    if remainder != 0:
        pad_len = align - remainder
        pad_token_id = chat_template.tokenizer.pad_token_id or 0
        tokenized_example["input_ids"] = torch.cat([
            tokenized_example["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=tokenized_example["input_ids"].dtype),
        ])
        tokenized_example["attention_mask"] = torch.cat([
            tokenized_example["attention_mask"],
            torch.ones(pad_len, dtype=tokenized_example["attention_mask"].dtype),
        ])
        IGNORE_INDEX = -100
        tokenized_example["labels"] = torch.cat([
            tokenized_example["labels"],
            torch.full((pad_len,), IGNORE_INDEX, dtype=tokenized_example["labels"].dtype),
        ])

    return [tokenized_example]
