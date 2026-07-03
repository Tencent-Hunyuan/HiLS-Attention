"""CRUxEval-O (output prediction) 2-shot generation config.

Requires `eval.configs.datasets.custom_datasets` to be imported beforehand
so that CRUxEvalDataset / cruxeval_o_postprocess are registered. See
eval_opencompass_sglang.py which pre-imports this module.
"""
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

from custom_datasets import CRUxEvalDataset, CRUxEvalEvaluator

cruxeval_o_reader_cfg = dict(
    input_columns=['code', 'input'],
    output_column='output',
    train_split='test',
    test_split='test')

cruxeval_o_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=
                     'Based on the given Python code and input, predict the output.\n\n'
                     'Code:\n```python\ndef f(s):\n    return s[::-1]\n```\n'
                     'Input: f("hello")\n'
                     'What is the output?'),
                dict(role='BOT', prompt='The output is "olleh"\n'),
                dict(role='HUMAN', prompt=
                     'Based on the given Python code and input, predict the output.\n\n'
                     'Code:\n```python\ndef f(lst):\n    return sorted(lst, reverse=True)\n```\n'
                     'Input: f([3, 1, 4, 1, 5])\n'
                     'What is the output?'),
                dict(role='BOT', prompt='The output is [5, 4, 3, 1, 1]\n'),
                dict(role='HUMAN', prompt=
                     'Based on the given Python code and input, predict the output.\n\n'
                     'Code:\n```python\n{code}\n```\n'
                     'Input: {input}\n'
                     'What is the output?'),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        max_out_len=128,
        # Force at least a few tokens so base models don't emit EOS at the
        # HUMAN→BOT boundary before producing any answer. GenInferencer
        # forwards min_out_len to model.generate if the signature accepts
        # it; see SGLangModel.generate.
        min_out_len=8,
        stopping_criteria=['\n\n', '\nCode:', '\nInput:', '\nBased on']))

cruxeval_o_eval_cfg = dict(
    evaluator=dict(type=CRUxEvalEvaluator),
    pred_postprocessor=dict(type='cruxeval_o'))

cruxeval_o_datasets = [
    dict(
        abbr='cruxeval_o',
        type=CRUxEvalDataset,
        path='cruxeval-org/cruxeval',
        reader_cfg=cruxeval_o_reader_cfg,
        infer_cfg=cruxeval_o_infer_cfg,
        eval_cfg=cruxeval_o_eval_cfg)
]
