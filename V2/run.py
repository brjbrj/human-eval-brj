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

    # 生成样例的输出jsonl文件路径
    "OUT_GENERATE_FILE" : "/root/brjverl/verl/data/humaneval/human-eval/human_eval/example/mbpp_samples.jsonl",

    # 问题jsonl文件路径
    "INPUT_PROBLEM_FILE" : "/root/brjverl/verl/data/humaneval/human-eval/human_eval/example/mbpp_test.jsonl",

    # 日志输出txt文件路径
    "LOG_FILE" : "/root/brjverl/verl/data/humaneval/human-eval/human_eval/example/log/mbpp.txt",

    # 是否加载代码进日志文件
    "LOG_CODE" : True,

    #####适配部分#####
    # 问题提示词适配函数名--传入参数 data_line为问题集的一条数据
    "PROMPT_FUNCTION_NAME" : "prompt_example",

    # 测试代码拼接函数名--传入参数 data_line为问题集的一条数据，completion为生成内容
    "TEST_CODE_FUNCTION_NAME" : "test_code_example",

    # 评测代码拼接函数名--传入参数 data_line为问题集的一条数据
    "TEST_FUNCTION_NAME" : "test_example",

    # 进入点函数名
    "ENTRY_POINT_FUNCTION_NAME" : "entry_point_example",
}

if __name__ == "__main__":
    generate.generate_sample(config)
    evaluate.evaluate_sample(config)