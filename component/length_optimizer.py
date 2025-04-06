# 相关模块导入
import argparse
import asyncio
import torch
import random

# BoTorch 模型与采集函数相关组件
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms import Normalize
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll

# 本地模块：LLM 接口、配置、Prompt模板
from chat.chat_llm_openai import ChatLLM
from config_loader import ConfigLoader
from prompt.evaluate_prompt import EVALUATE_PROMPT
from utils.load_utils import LoadUtils
from utils.logger_utils import LoggerUtil
logger = LoggerUtil.get_logger("optimizer")


class TokenLengthOptimizer:
    """
    使用 Pairwise Gaussian Process 实现的偏好贝叶斯优化器，用于寻找最佳 LLM token 长度。
    """
    def __init__(self, token_bounds=(100, 4000), config: ConfigLoader = None, model_name: str = "gpt-3.5-turbo"):
        self.token_bounds = token_bounds                  # token 长度的搜索边界
        self.token_history = []                           # 记录历史所有 token 值
        self.comparisons = []                             # 存储 pairwise 偏好对 (winner, loser)
        self.gpmodel = None                               # GP 模型
        self.config = config

        # 初始化 LLM 接口
        self.model = config.models[model_name]
        self.llm = ChatLLM(
            api_key=self.model.get("api_key"),
            base_url=self.model.get("base_url"),
            model=model_name
        )

    def update_listwise(self, token_list: list[int], ranked_indices: list[int]) -> None:
        """
        将 listwise 排名转换为所有 pairwise 偏好对，并更新 GP 模型
        """
        init_idx = len(self.token_history)
        self.token_history.extend(token_list)

        # 将排序转为 pairwise：如排序 [0, 2, 1] → (0 > 2), (0 > 1), (2 > 1)
        for i in range(len(ranked_indices)):
            for j in range(i + 1, len(ranked_indices)):
                winner = init_idx + ranked_indices[i]
                loser = init_idx + ranked_indices[j]
                self.comparisons.append((winner, loser))

        logger.info(f"✅ 添加 pairwise 比较对数量: {len(self.comparisons)}")
        # 拟合 GP 模型
        self._fit_model()


    def _fit_model(self):
        """
        拟合基于 pairwise 偏好的高斯过程模型
        """
        if not self.token_history or not self.comparisons:
            raise ValueError("无法拟合 GP 模型，缺少 token 历史或比较对")

        # 将 token 与 pairwise 比较对转为 tensor
        X = torch.tensor(self.token_history, dtype=torch.float).unsqueeze(-1).to(torch.double)
        comps = torch.tensor(self.comparisons, dtype=torch.long)

        # 构建 GP 模型 + 归一化
        self.gpmodel = PairwiseGP(X, comps, input_transform=Normalize(d=1))
        # 拟合 GP 模型（使用边际似然最大化）
        mll = PairwiseLaplaceMarginalLogLikelihood(self.gpmodel.likelihood, self.gpmodel)
        fit_gpytorch_mll(mll)


    def suggest_next(self, n_suggestions=4, anchor_token=None, exclude: set = None) -> list[int]:
        """
        根据 GP 模型采集函数推荐下一个 token 候选集合（避免重复、可加入锚点）
        """
        if not self.gpmodel:
            raise ValueError("GP 模型尚未拟合")

        bounds = torch.tensor([[self.token_bounds[0]], [self.token_bounds[1]]], dtype=torch.float)
        exclude = exclude or set()
        suggestions = set()

        # 优化采集函数，生成新 token 候选
        try:
            while len(suggestions) < n_suggestions:
                acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=self.gpmodel)
                candidates, _ = optimize_acqf(
                    acq_function=acq_func,
                    bounds=bounds,
                    q=2,  # 只支持 q=1 或 q=2
                    num_restarts=5,
                    raw_samples=64,
                )
                for c in candidates:
                    val = int(c.item())
                    if val not in exclude and self.token_bounds[0] <= val <= self.token_bounds[1]:
                        suggestions.add(val)
        except Exception:
            # fallback 随机生成
            while len(suggestions) < n_suggestions:
                fallback = random.randint(*self.token_bounds)
                if fallback not in exclude:
                    suggestions.add(fallback)

        # 替换最后一个为 anchor token（如当前最优值）
        if anchor_token and anchor_token not in suggestions and anchor_token not in exclude:
            suggestions = list(suggestions)
            suggestions[-1] = anchor_token
            suggestions = set(suggestions)

        return sorted(suggestions)


    async def generate_output(self, prompt: str, token_count: int) -> str:
        """
        调用 LLM 生成指定 token 长度的回答
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt} (Think with {token_count} tokens.)"}
        ]
        result = await self.llm.chat(messages)
        return result.content.strip()


    async def listwise(self, qa_dict: dict[int, list[dict]]) -> list[int]:
        """
        输入多个 token 候选对应的 QA 对，让 LLM 返回 listwise 排名结果（索引顺序）
        如果 LLM 返回格式不合要求，将尝试重试，并打印错误内容以供调试。
        """
        import ast

        token_list = list(qa_dict.keys())

        # 构建 answer block 展示给 LLM
        answer_block = ""
        print("🧩 listwise 传入的 token_list:", token_list)
        for idx, token in enumerate(token_list):
            answer_block += f"token {idx} (token={token}):\n"
            for qa in qa_dict[token]:
                answer_block += f"Question: {qa['question']}\nAnswer: {qa['answer']}\n\n"
        # print(answer_block)

        # 构建完整 prompt（注意你可以将 EVALUATE_PROMPT 调整成更强的版本）
        full_prompt = EVALUATE_PROMPT.format(answer_block=answer_block.strip(),
                                             token_list= str(token_list))
        messages = [
            {"role": "system", "content": "You are a helpful evaluator."},
            {"role": "user", "content": full_prompt}
        ]

        # 尝试调用 LLM 进行排序，最多三次尝试
        for attempt in range(3):
            prompt = EVALUATE_PROMPT.format(answer_block=answer_block.strip(),
                                             token_list= str(token_list))
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.chat(messages)
            try:
                # print(response.content)
                # ✅ 使用 LoadUtils 提取 <ranking> 标签内容
                ranking = LoadUtils.extract_content(response.content, "ranking")
                # ✅ 转为 list 对象
                ranked = ast.literal_eval(ranking)
                if isinstance(ranked, list) and all(isinstance(i, int) for i in ranked):
                    logger.info(f"🧠 warm-up listwise 排序结果: {ranked}")
                    return ranked
                else:
                    raise ValueError("解析出的结构不是 int list")

            except Exception as e:
                print(f"[⚠️ 排序解析失败 - 第 {attempt + 1} 次尝试]")
                print(f"⛔ LLM 回复部分内容:\n<ranking>{ranking}</ranking>")
                print(f"🚨 错误信息: {str(e)}")

        # 所有尝试失败，终止
        raise ValueError("无法从 LLM 回复中提取合法排序结果")

        # （可选 fallback）返回默认排序避免崩溃
        # return list(range(len(token_list)))


    def get_best_token(self) -> int:
        """根据最近一次排序返回当前最优token"""
        if not self.token_history:
            raise ValueError("未找到历史token")
        token_tensor = torch.tensor(self.token_history).unsqueeze(-1).double()
        with torch.no_grad():
            preds = self.gpmodel.posterior(token_tensor).mean.squeeze(-1)
        best_idx = torch.argmax(preds).item()
        return self.token_history[best_idx]


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

    m = 2  # 每轮候选数
    initial_tokens = [400, 800, 1200, 1600, 2000, 2400]

    # 模拟 warm-up 阶段回答数据（真实项目中应该用 execute_prompt 生成）
    qa_dict = {
        n: [{"question": "What is the capital of France?", "answer": f"Paris with reasoning for {n} tokens"}]
        for n in initial_tokens
    }

    print("🚀 Warm-up with initial token values...")
    ranked_indices = await opt.listwise(qa_dict)
    opt.update_listwise(initial_tokens, ranked_indices)

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
        opt.update_listwise(token_list, ranked_indices)

        print(f"候选 token: {token_list}")
        print(f"↳ 排序结果: {ranked_indices}")
        print(f"✅ 模型已更新")

        m = max(m // 2, 2)

if __name__ == "__main__":
    asyncio.run(main())
