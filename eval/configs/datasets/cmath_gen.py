"""CMATH 4-shot CoT generation config.

Requires `eval.configs.datasets.custom_datasets` to be imported beforehand
so that CMATHDataset / cmath_postprocess are registered. See
eval_opencompass_sglang.py which pre-imports this module.
"""
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# `custom_datasets` must be importable at Config-parse time. The runner
# (eval_opencompass_sglang.py) adds eval/configs/datasets/ to sys.path
# and imports it eagerly, triggering registry decorators.
from custom_datasets import CMATHDataset, CMATHEvaluator

cmath_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    # CMATH has only `validation` and `test` splits, no `train`.
    # Point both split aliases at `test` so DatasetReader doesn't crash;
    # our few-shot examples are hardcoded in the prompt template anyway.
    train_split='test',
    test_split='test')

cmath_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='问题：芳芳买了一本书有99页，看了90页，她还剩多少页没有看？\n让我们一步一步思考\n答案：'),
                dict(role='BOT', prompt='芳芳买了99页的书，看了90页，还剩99-90=9页没有看。\n答案是 9\n'),
                dict(role='HUMAN', prompt='问题：商店里有4箱苹果，每箱25个。卖出50个以后，还剩多少个？\n让我们一步一步思考\n答案：'),
                dict(role='BOT', prompt='4箱苹果一共有4×25=100个，卖出50个后还剩100-50=50个。\n答案是 50\n'),
                dict(role='HUMAN', prompt='问题：学校图书室有故事书98本，今天借出46本，还回25本。现在图书室有故事书多少本？\n让我们一步一步思考\n答案：'),
                dict(role='BOT', prompt='图书室原有98本，借出46本后剩98-46=52本，又还回25本，现在有52+25=77本。\n答案是 77\n'),
                dict(role='HUMAN', prompt='问题：一列火车从甲地开往乙地，第一天行了全程的三分之一多60千米，第二天行了全程的一半少30千米，还剩190千米。甲乙两地相距多少千米？\n让我们一步一步思考\n答案：'),
                dict(role='BOT', prompt='设全程为x千米。第一天行了x/3+60千米，第二天行了x/2-30千米，还剩190千米。\n所以 x/3+60 + x/2-30 + 190 = x\n化简得 5x/6 + 220 = x\n解得 x/6 = 220，x = 1320千米。\n答案是 1320\n'),
                dict(role='HUMAN', prompt='问题：{question}\n让我们一步一步思考\n答案：'),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        max_out_len=512,
        # Stop as soon as the model starts a new Q/A pair — base models
        # will otherwise keep hallucinating extra problems indefinitely.
        stopping_criteria=['\n问题：', '\n\n问题', 'Question:', 'Problem:']))

cmath_eval_cfg = dict(
    evaluator=dict(type=CMATHEvaluator),
    pred_postprocessor=dict(type='cmath'),
    dataset_postprocessor=dict(type='cmath_dataset'))

cmath_datasets = [
    dict(
        abbr='cmath',
        type=CMATHDataset,
        path='weitianwen/cmath',
        reader_cfg=cmath_reader_cfg,
        infer_cfg=cmath_infer_cfg,
        eval_cfg=cmath_eval_cfg)
]
