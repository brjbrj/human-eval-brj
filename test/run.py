import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import tempfile
import subprocess
import time
from typing import List, Dict, Tuple, Iterable
import gzip
import customer
import data
import evaluate
import generate

# ---------------------- 可配置参数 ----------------------
config = {
    # 代码生成模型
    "MODEL_PATH" : "/root/brjverl/models/deepseek-math-7b-instruct",
    # 每个任务样本数
    "NUM_SAMPLES_PER_TASK" : 10,
    # 最大输入长度
    "MAX_INPUT_LENGTH" : 512,
    # 最大TOKEN长度
    "MAX_NEW_TOKENS" : 200,
    "TEMPERATURE" : 0.7,
    "TOP_P" : 0.95,
    # 样例的文件路径
    "SAMPLE_FILE" : "/root/brjverl/verl/compiler/sample/test_samples_all.jsonl",
    # 问题文件路径
    "PROBLEM_FILE" : "/root/brjverl/verl/compiler/data/human-eval-v2-20210705.jsonl",
    # 结果文件路径
    "RESULT_FILE" : "/root/brjverl/verl/compiler/result/humaneval_result.jsonl",
    #####适配部分#####
    # 问题提示词适配函数名--传入参数 data_line为问题集的一条数据
    "PROMPT_FUNCTION_NAME" : "prompt_example",
    # 测试代码拼接函数名--传入参数 data_line为问题集的一条数据，completion为生成内容
    "TEST_CODE_FUNCTION_NAME" : "test_code_example",
    # 评测代码拼接函数名--传入参数 data_line为问题集的一条数据
    "TEST_FUNCTION_NAME" : "test_example",
    # 进入点函数名
    "ENTRY_POINT_FUNCTION_NAME" : "entry_point_example",
    #pass@k
    "K" : "1,5,10",
    #n_workers
    "N_WORKERS" : 1,
    #超时限制
    "TIMEOUT" : 3.0,
}

if __name__ == "__main__":
    #generate.generate_sample(config) #注释该句则仅评测
    evaluate.evaluate_sample(config) #注释该句则仅生成
    #mbpp {'pass@1': 0.41344969199178644, 'pass@5': 0.6461898243212412, 'pass@10': 0.7053388090349076}
    #humaneval {'pass@1': 0.34024390243902436, 'pass@5': 0.5820267131242741, 'pass@10': 0.6585365853658537}