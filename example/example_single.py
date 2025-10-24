# 样例生成+评估 二合一
# 带进度条显示的评估
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.data import write_jsonl, read_problems
import json
import tempfile
import subprocess
import time
from typing import List, Dict, Tuple

# ---------------------- 可配置参数 ----------------------
MODEL_PATH = "/root/brjverl/models/deepseek-math-7b-instruct"
NUM_SAMPLES_PER_TASK = 10 # 每个任务样本数
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.95
OUT_GENERATE_FILE = "example_samples.jsonl"
LOG_CODE = False
# -------------------------------------------------------

def filter_code(completion: str) -> str:
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

def init_model_and_tokenizer(model_path: str):
    print(f"正在加载模型：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    print(f"模型加载完成，设备：{model.device}")
    return tokenizer, model

def generate_one_completion(prompt: str, tokenizer, model) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_code = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )
    return filter_code(generated_code)

# 生成日志内容
def log(message: str, log_file):
    print(message)
    log_file.write(message + "\n")

# 加载生成的样本
def load_generated_samples(sample_path: str, log_file) -> Dict[str, List[str]]:
    task_samples = {}
    with open(sample_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())
            task_id = sample["task_id"]
            completion = sample["completion"]
            if task_id not in task_samples:
                task_samples[task_id] = []
            task_samples[task_id].append(completion)
    log(f"加载完成：共{len(task_samples)}个任务，样本数={sum(len(v) for v in task_samples.values())}", log_file)
    return task_samples

# 生成拼接测试代码
def generate_test_script(problem: Dict, completion: str, log_file) -> str:
    test_script = f"{problem['prompt']}\n{completion}\n\n"
    entry_point = problem.get("entry_point")
    test_code = problem["test"]
    
    if "def check(candidate)" in test_code and entry_point:
        test_code = test_code.replace("def check(candidate):", f"def test_check():\n    candidate = {entry_point}")
        test_script += test_code
    else:
        if "def test():" in test_code:
            test_code = test_code.replace("def test():", "def test_main():")
            test_script += test_code + "\n\ndef test_check():\n    test_main()"
        else:
            test_script += test_code
    if LOG_CODE:
        log("\n" + "="*30 + " 拼接的测试脚本 " + "="*30, log_file)
        log(test_script, log_file)
    log("="*70 + "\n", log_file)
    return test_script

# 测试单样例代码
def run_single_test(test_script: str, timeout: int = 5) -> Tuple[bool, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(test_script)
        temp_file_path = temp_file.name

    try:
        result = subprocess.run(
            ["pytest", temp_file_path, "-v", "-x"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        passed = "PASSED" in result.stdout
        error_msg = f"stdout: {result.stdout}\nstderr: {result.stderr}" if not passed else "无错误"
        return passed, error_msg
    except subprocess.TimeoutExpired:
        return False, "测试超时"
    except Exception as e:
        return False, f"执行出错：{str(e)}"
    finally:
        os.unlink(temp_file_path)

# 评估任务
def evaluate_task(task_id: str, problem: Dict, completions: List[str], log_file) -> bool:
    log(f"\n===== 开始评估任务 {task_id}（共{len(completions)}个样本） =====", log_file)
    for idx, completion in enumerate(completions, 1):
        log(f"----- 测试第{idx}个样本 -----", log_file)
        test_script = generate_test_script(problem, completion, log_file)
        passed, error_msg = run_single_test(test_script)
        if passed:
            log(f"✅ 第{idx}个样本通过测试！", log_file)
            return True
        else:
            log(f"❌ 第{idx}个样本未通过，错误信息：\n{error_msg}", log_file)
    log(f"===== 任务{task_id}：所有样本均未通过 =====", log_file)
    return False

#计算pass@k值和每个任务的评估结果
def calculate_pass_at_k(
    task_samples: Dict[str, List[str]],
    problems: Dict[str, Dict],
    log_file,
    k: List[int] = [1, 10]
) -> Tuple[Dict[int, float], Dict[str, bool]]:
    total_tasks = len(task_samples)
    if total_tasks == 0:
        return {}, {}
    
    passed_tasks = 0
    task_results = {}  # 存储每个任务的评估结果：{task_id: 是否通过}
    for task_id in task_samples:
        if task_id not in problems:
            log(f"跳过未知任务：{task_id}", log_file)
            task_results[task_id] = False
            continue
        # 仅评估一次，结果存入task_results
        result = evaluate_task(task_id, problems[task_id], task_samples[task_id], log_file)
        task_results[task_id] = result
        if result:
            passed_tasks += 1
    pass_at_k = {}
    for current_k in k:
        pass_at_k[current_k] = round(passed_tasks / total_tasks, 4)
    return pass_at_k, task_results  # 返回pass_at_k和任务结果字典

#生成样例
#output_file_name:XXXX.jsonl
def generate_sample(output_file_name: str = "test_samples.jsonl", humen_eval_name:str = "HumanEval.jsonl.gz"):
    tokenizer, model = init_model_and_tokenizer(MODEL_PATH)
    current_work_dir = os.path.dirname(__file__)
    SAMPLE_PATH = os.path.join(current_work_dir, humen_eval_name)
    problems = read_problems(SAMPLE_PATH)
    all_task_ids = list(problems.keys())
    total_tasks = len(all_task_ids)
    total_samples = total_tasks * NUM_SAMPLES_PER_TASK
    
    print(f"\n===== 开始生成样本 =====")
    print(f"任务总数：{total_tasks}")
    print(f"每个任务样本数：{NUM_SAMPLES_PER_TASK}")
    print(f"总样本数：{total_samples}")
    
    samples = []
    # 进度条中实时显示当前任务ID
    with tqdm(total=total_samples, desc="生成进度") as pbar:
        for task_idx, task_id in enumerate(all_task_ids):
            # 每个任务开始时打印一次（关键修改）
            print(f"\n----- 开始处理第{task_idx+1}/{total_tasks}个任务：{task_id} -----")
            
            prompt = problems[task_id]["prompt"]
            for sample_idx in range(NUM_SAMPLES_PER_TASK):
                # 生成样本时，在进度条描述中显示当前任务和样本序号
                pbar.set_description(f"生成进度（任务{task_id}，样本{sample_idx+1}/{NUM_SAMPLES_PER_TASK}）")
                
                completion = generate_one_completion(prompt, tokenizer, model)
                samples.append({
                    "task_id": task_id,
                    "completion": completion
                })
                pbar.update(1)
    
    current_work_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_work_dir, output_file_name)
    write_jsonl(file_path, samples)
    
    print(f"\n===== 生成完成 =====")
    print(f"实际生成样本数：{len(samples)}")
    print(f"保存路径：{file_path}")

# 评估样本
# sample_file_name:XXXX.jsonl
def evaluate_samples(sample_file_name:str = "test_samples.jsonl", humen_eval_name:str = "HumanEval.jsonl.gz"):
    current_work_dir = os.path.dirname(__file__)
    SAMPLE_PATH = os.path.join(current_work_dir, sample_file_name)
    HUMAN_EVAL_LOCAL_PATH = os.path.join(current_work_dir, humen_eval_name)
    
    log_file_name = f"evaluation_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    log_file_path = os.path.join(current_work_dir, log_file_name)
    
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log(" Step 1：加载生成样本和HumanEval问题...", log_file)
        task_samples = load_generated_samples(SAMPLE_PATH, log_file)
        problems = read_problems(HUMAN_EVAL_LOCAL_PATH)
        
        log("\n Step 2：开始评估...", log_file)
        # 调用时获取pass_at_k和任务结果字典
        pass_at_k, task_results = calculate_pass_at_k(task_samples, problems, log_file=log_file, k=[1, 10])
        
        log("\n" + "="*50, log_file)
        log("评估结果汇总", log_file)
        log("-"*50, log_file)
        log(f"总任务数：{len(task_samples)}", log_file)
        
        # 直接从task_results统计通过任务数，无需再次评估
        passed_count = sum(1 for res in task_results.values() if res)
        log(f"通过任务数：{passed_count}", log_file)
        
        log("\nPass@k指标：", log_file)
        for k_val, score in pass_at_k.items():
            log(f"Pass@{k_val}：{score}（{score*100:.2f}%）", log_file)
        log("="*50, log_file)
    
    print(f"\n所有输出已同步保存到日志文件：{log_file_path}")

if __name__ == "__main__":
    generate_sample(OUT_GENERATE_FILE, "example_problem.jsonl")
    evaluate_samples(OUT_GENERATE_FILE, "example_problem.jsonl")
    