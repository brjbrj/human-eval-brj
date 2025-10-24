# 样例生成文件
# 1. 导入需要的库
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.data import write_jsonl, read_problems

# 2. 关键：在当前文件定义filter_code，解决模块缺失问题
def filter_code(completion: str) -> str:
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

# 3. 初始化模型
MODEL_PATH = "/root/brjverl/models/deepseek-math-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)

# 4. 实现generate_one_completion
def generate_one_completion(prompt: str) -> str:
    # 转换prompt为模型输入
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)
    
    # 模型生成
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 解码并清理代码
    generated_code = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )
    cleaned_code = filter_code(generated_code)  # 这里不再报模块错误
    
    return cleaned_code

# 5. 测试运行
if __name__ == "__main__":
    problems = read_problems()
    num_samples_per_task = 10  # 少量测试，避免耗时
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
        for task_id in list(problems.keys())  # 只测第一个任务,list(problems.keys())[:1]为第一个，list(problems.keys())为所有
        for _ in range(num_samples_per_task)
    ]
    current_work_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_work_dir, "test_samples_all.jsonl")
    write_jsonl(file_path, samples)
    print(f"测试完成！生成了{len(samples)}条样本，保存到{file_path}")