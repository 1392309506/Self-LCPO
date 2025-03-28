import requests

class General_LLM:
    def __init__(self, model: dict):
        """
        初始化 GeneralLLM。
        :param model: 基于ConfigLoader得到的字典
        """
        print(model)
        self.api_url = model.get("base_url")
        self.api_key = model.get("api_key")
        params = model.get("params")

        self.model = params.get("model")
        self.temperature = params.get("temperature")
        self.max_tokens = params.get("max_tokens")

    def generate_response(self, prompt: str) -> str:
        """
        发送请求到 LLM 并获取回复。
        :param prompt: 输入的提示文本
        :return: 模型生成的回复
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json().get("text", "")
        except requests.RequestException as e:
            print(f"请求失败: {e}")
            return ""


# 使用示例
if __name__ == "__main__":
    api_url = "https://api.example.com/v1/chat"  # 替换为实际 API URL
    api_key = "your_api_key"  # 替换为实际 API 密钥

    # llm = GeneralLLM({api_url, api_key, model="gpt-4o", temperature=0.8})
    # prompt = "请解释量子物理的基本概念。"
    # response = llm.generate_response(prompt)
    # print("模型回复:", response)
