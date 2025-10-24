# 带进度条显示的样例生成
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.data import write_jsonl, read_problems

# ---------------------- 可配置参数 ----------------------
MODEL_PATH = "/root/brjverl/models/deepseek-math-7b-instruct"
NUM_SAMPLES_PER_TASK = 10
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.95
OUTPUT_FILE_NAME = "test_samples_all.jsonl"
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

def main():
    tokenizer, model = init_model_and_tokenizer(MODEL_PATH)
    problems = read_problems()
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
    file_path = os.path.join(current_work_dir, OUTPUT_FILE_NAME)
    write_jsonl(file_path, samples)
    
    print(f"\n===== 生成完成 =====")
    print(f"实际生成样本数：{len(samples)}")
    print(f"保存路径：{file_path}")

if __name__ == "__main__":
    main()