"""HumanEval+ 0-shot generation config.

Uses HumanEvalPlusEvaluatorFixed from custom_datasets.py (which sidesteps
the API mismatch between opencompass.datasets.HumanEvalPlusEvaluator and
recent evalplus versions that causes `assert samples is not None`).
"""
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalDataset

# `custom_datasets` is pre-imported by eval_opencompass_sglang.py
from custom_datasets import HumanEvalPlusEvaluatorFixed

humaneval_plus_reader_cfg = dict(
    input_columns=['prompt'], output_column='task_id', train_split='test')

humaneval_plus_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='Complete the following python code:\n{prompt}'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        max_out_len=512,
        # Cut the completion at top-level boundaries so the base model
        # doesn't emit a second function definition or a test harness.
        stopping_criteria=['\ndef ', '\nclass ', '\nif __name__', '\nassert ', '\nprint(']))

humaneval_plus_eval_cfg = dict(
    evaluator=dict(type=HumanEvalPlusEvaluatorFixed),
    pred_role='BOT',
    k=[1],
    pred_postprocessor=dict(type='humaneval_plus_robust'),
)

humaneval_plus_datasets = [
    dict(
        abbr='humaneval_plus',
        type=HumanevalDataset,
        path='opencompass/humaneval',
        reader_cfg=humaneval_plus_reader_cfg,
        infer_cfg=humaneval_plus_infer_cfg,
        eval_cfg=humaneval_plus_eval_cfg)
]
