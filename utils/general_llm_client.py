import requests
from utils.logger_utils import LoggerUtil

logger = LoggerUtil.get_logger("General_LLM_Client")


class General_LLM:
    def __init__(self, model: dict):
        """
        初始化 GeneralLLM。
        :param model: 基于ConfigLoader得到的字典
        """
        print(model)
        # 修改为正确的 API URL
        self.api_url = model.get("base_url")
        self.api_key = model.get("api_key")
        params = model.get("params")

        self.model = params.get("model")
        self.temperature = params.get("temperature")
        self.max_tokens = params.get("max_tokens")

    def generate_response(self, prompt: str) -> str:
        try:
            # 构造请求，保持与文档示例一致
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
            print("请求体:", payload)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # 发送请求
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            raw_response = response.text
            response_data = response.json()

            # 记录原始响应
            print(f"API响应原始内容: {response_data}")

            # 错误处理
            if "error" in response_data:
                error_info = response_data["error"]
                logger.error(f"API错误 [{error_info.get('code')}]: {error_info.get('message')}")
                return ""

            # 校验数据格式
            if not isinstance(response_data.get("choices"), list) or len(response_data["choices"]) == 0:
                logger.error("响应格式异常，缺少有效choices字段")
                return ""

            # 提取内容
            content = response_data["choices"][0].get("message", {}).get("content", "")
            return content.strip()

        except requests.exceptions.JSONDecodeError:
            logger.error(f"响应不是合法JSON: {raw_response}")
            return ""
        except Exception as e:
            logger.error(f"未知错误: {str(e)}")
            return ""


# 使用示例
if __name__ == "__main__":
    model_config = {
        "base_url": "https://api.chatanywhere.org",  # 请使用文档中提供的正确接口地址
        "api_key": "your_api_key",  # 替换为实际 API 密钥
        "params": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 150,
        }
    }
    llm = General_LLM(model_config)
    prompt = "Say this is a test!"
    response = llm.generate_response(prompt)
    print("生成的响应:", response)
