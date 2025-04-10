import openai
from dataset import get_examples, is_correct, extract_answer
import time

# Step 1: 设置 Ollama 的 OpenAI 接口参数
openai.api_key = "ollama"
openai.base_url = "http://localhost:11434/v1"
MODEL_NAME = "deepseek-r1:7b"


# Step 2: 设置自定义 Prompt 模板
def make_prompt(question: str) -> str:
    return f"你是一个聪明的小学数学老师，请逐步解决这个问题：\n{question}\n答案是："


# Step 3: 调用 Ollama 模型生成答案
def call_ollama(prompt: str, max_tokens=512, temperature=0.7) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error during model call: {e}")
        return ""


# Step 4: 主评估逻辑
def evaluate_prompt_on_gsm8k():
    examples = get_examples("test")
    correct = 0
    total = len(examples)

    for idx, ex in enumerate(examples):
        prompt = make_prompt(ex["question"])
        output = call_ollama(prompt)

        # 你也可以 print(prompt, output) 来调试
        if is_correct(output, ex):
            correct += 1
        else:
            print(
                f"[{idx}] ❌ Wrong\nQ: {ex['question']}\nExpected: {extract_answer(ex['answer'])}\nGot: {extract_answer(output)}\n")

        time.sleep(0.5)  # 防止请求过快，按需调整

    acc = correct / total
    print(f"\n🎯 Prompt Accuracy: {acc:.2%} ({correct}/{total})")


if __name__ == "__main__":
    evaluate_prompt_on_gsm8k()
