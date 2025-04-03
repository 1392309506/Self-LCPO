import argparse

import torch
import asyncio
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms import Normalize
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll

from chat.chat_llm_openai import ChatLLM
from config_loader import ConfigLoader
from prompt.evaluate_prompt import EVALUATE_PROMPT

# -------------------
# 贝叶斯优化器类
# -------------------
class TokenLengthOptimizer:
    def __init__(self, token_bounds=(100, 4000), config: ConfigLoader = None, model_name: str = "gpt-4"):
        self.token_bounds = token_bounds
        self.token_history = []         # 记录历史 token 数量
        self.comparisons = []           # 记录 pairwise 偏好对 (i, j)
        self.gpmodel = None

        # 配置与 LLM 初始化
        self.config = config
        self.model = config.models[model_name]
        self.llm = ChatLLM(api_key=self.model.get("api_key"), base_url=self.model.get("base_url"))

    def register_ranked_list(self, token_list: list[int], ranked_indices: list[int]) -> None:
        """
        将 listwise 排序结果转化为 pairwise 比较对，并更新 GP 模型
        """
        start_idx = len(self.token_history)
        self.token_history.extend(token_list)
        for i in range(len(ranked_indices)):
            for j in range(i + 1, len(ranked_indices)):
                winner = start_idx + ranked_indices[i]
                loser = start_idx + ranked_indices[j]
                self.comparisons.append((winner, loser))
        self._fit_model()

    def _fit_model(self):
        """
        拟合偏好 GP 模型
        """
        if not self.token_history or not self.comparisons:
            raise ValueError("无法拟合 GP 模型，缺少 token 历史或比较对")
        X = torch.tensor(self.token_history, dtype=torch.float).unsqueeze(-1)
        comps = torch.tensor(self.comparisons, dtype=torch.long)
        self.gpmodel = PairwiseGP(X, comps, input_transform=Normalize(d=1))
        mll = PairwiseLaplaceMarginalLogLikelihood(self.gpmodel.likelihood, self.gpmodel)
        fit_gpytorch_mll(mll)

    def suggest_next(self, n_suggestions=4) -> list[int]:
        """
        使用当前模型生成下一批建议 token 数量
        """
        if not self.gpmodel:
            raise ValueError("GP 模型尚未拟合，请先执行 register_ranked_list")

        bounds = torch.tensor([[self.token_bounds[0]], [self.token_bounds[1]]], dtype=torch.float)
        acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=self.gpmodel)
        candidates, _ = optimize_acqf(acq_func, bounds=bounds.T, q=n_suggestions, num_restarts=5, raw_samples=64)
        return [int(c.item()) for c in candidates.view(-1)]

    async def generate_output(self, prompt: str, token_count: int) -> str:
        """
        使用 LLM 生成指定 token 长度下的回答
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt} (Think with {token_count} tokens.)"}
        ]
        result = await self.llm.chat(messages)
        return result.content.strip()

    async def listwise(self, qa_dict: dict[int, list[dict]]) -> list[int]:
        """
        输入：qa_dict -> {token_count: [{"question": ..., "answer": ...}, ...]}
        输出：返回 token_count 排名 index 列表
        """

        # 构造 Answer Block
        answer_block = ""
        token_list = list(qa_dict.keys())
        for idx, token in enumerate(token_list):
            answer_block += f"Answer {idx} (token={token}):\n"
            for qa in qa_dict[token]:
                answer_block += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"

        # 替换下面这个为你自定义的更具体的评估 prompt
        full_prompt = EVALUATE_PROMPT.format(answer_block=answer_block.strip())

        messages = [
            {"role": "system", "content": "You are a helpful evaluator."},
            {"role": "user", "content": full_prompt}
        ]
        reply = await self.llm.chat(messages)
        print(f"\n🧠 LLM 排序回复:\n{reply.content}\n")

        # 提取排序
        import re, ast
        match = re.search(r"\[.*?\]", reply.content)
        if match:
            try:
                ranked = ast.literal_eval(match.group(0))
                return ranked
            except Exception:
                raise ValueError("排序解析失败，格式错误")
        else:
            raise ValueError("无法从回复中解析排序结果")


def parse_args():
    parser = argparse.ArgumentParser(description='大模型思考长度实验')
    parser.add_argument('--config', type=str, default='../config/config_llm.yaml',
                        help='配置文件路径（默认：config/config_llm.yaml）')
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="Project name")
    return parser.parse_args()

# -------------------
# 主优化流程（异步）
# -------------------
async def main():
    args = parse_args()
    config = ConfigLoader(args.config)
    opt = TokenLengthOptimizer(token_bounds=(100, 4000), config=config, model_name=args.model_name)

    m = 6  # 每轮候选数
    initial_tokens = [400, 800, 1200, 1600, 2000, 2400]

    # 模拟 warm-up 阶段回答数据（真实项目中应该用 execute_prompt 生成）
    qa_mock = [{"question": "What is the capital of France?", "answer": f"Paris with reasoning for {n} tokens"} for n in initial_tokens]
    qa_dict = {n: qa_mock for n in initial_tokens}

    print("🚀 Warm-up with initial token values...")
    ranked_indices = await opt.listwise(qa_dict)
    opt.register_ranked_list(initial_tokens, ranked_indices)

    print(f"初始 token: {initial_tokens}")
    print(f"↳ 排序结果: {ranked_indices}")

    # 进入优化循环
    for step in range(5):
        print(f"\n[Step {step + 1}] 当前采样候选数量: {m}")
        token_list = opt.suggest_next(n_suggestions=m)

        # 模拟该 token 下的回答（此处简化）
        qa_dict = {
            n: [{"question": "What is the capital of France?", "answer": f"Paris with reasoning for {n} tokens"}]
            for n in token_list
        }

        ranked_indices = await opt.listwise(qa_dict)
        opt.register_ranked_list(token_list, ranked_indices)

        print(f"候选 token: {token_list}")
        print(f"↳ 排序结果: {ranked_indices}")
        print(f"✅ 模型已更新")

        m = max(m // 2, 2)

if __name__ == "__main__":
    asyncio.run(main())
