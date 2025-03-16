import asyncio
import re
import json
import requests
from enum import Enum
from typing import Any, Dict, List, Optional

# Online LLM API Configuration
DEFAULT_API_URL = "https://api.example.com/llm/generate"  # 线上大模型的API地址
# OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"  # 默认模型名称
TEMPERATURE = 0.7  # 默认温度参数
API_KEY = "your-api-key"  # 线上API需要的API key

class RequestType(Enum):
    OPTIMIZE = "optimize"
    EVALUATE = "evaluate"
    EXECUTE = "execute"
    ANALYZE = "analyze"

class LLMResponse:
    """简单类来模拟LLM的响应结构"""

    def __init__(self, content: str):
        self.choices = [self.Choice(content)]

    class Choice:
        def __init__(self, content: str):
            self.message = self.Message(content)

        class Message:
            def __init__(self, content: str):
                self.content = content

class SPO_LLM:
    _instance: Optional["SPO_LLM"] = None

    def __init__(
            self,
            optimize_kwargs: Optional[dict] = None,
            evaluate_kwargs: Optional[dict] = None,
            execute_kwargs: Optional[dict] = None,
            analyze_kwargs: Optional[dict] = None,
    ) -> None:
        self.optimize_config = self._prepare_config(optimize_kwargs)
        self.evaluate_config = self._prepare_config(evaluate_kwargs)
        self.execute_config = self._prepare_config(execute_kwargs)
        self.analyze_config = self._prepare_config(analyze_kwargs)

        # 打印初始化信息
        print(f"SPO_LLM initialized with model: {MODEL_NAME}")

    def _prepare_config(self, kwargs: Optional[dict]) -> Dict[str, Any]:
        """为LLM请求准备配置"""
        if not kwargs:
            kwargs = {}

        config = {
            "model": kwargs.get("model", MODEL_NAME),
            "base_url": kwargs.get("base_url", DEFAULT_API_URL),
            "temperature": kwargs.get("temperature", TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "api_key": kwargs.get("api_key", API_KEY)  # 增加API key的支持
        }

        return config

    async def _send_request(self, config: Dict[str, Any], messages: List[Dict[str, str]]) -> LLMResponse:
        """向线上大模型API发送请求"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_key']}"  # 通过Authorization头传递API key
        }

        # 对于LLM，我们需要提取最后一条消息的内容
        prompt = messages[-1]["content"]

        payload = {
            "model": config["model"],
            "prompt": prompt,
            "temperature": config["temperature"],
            "max_tokens": config.get("max_tokens", 1000),
            "stream": False
        }

        try:
            response = requests.post(
                config["base_url"],
                headers=headers,
                data=json.dumps(payload),
                timeout=30  # 设置请求超时时间
            )

            if response.status_code == 200:
                data = response.json()
                return LLMResponse(data.get("response", "").strip())
            else:
                error_msg = f"Error: API request failed with status code {response.status_code}"
                print(error_msg)
                return LLMResponse(error_msg)

        except requests.Timeout:
            error_msg = "Error: The request timed out"
            print(error_msg)
            return LLMResponse(error_msg)
        except Exception as e:
            error_msg = f"Exception during API call: {str(e)}"
            print(error_msg)
            return LLMResponse(error_msg)

    async def acompletion(self, config: Dict[str, Any], messages: List[Dict[str, str]]) -> LLMResponse:
        """异步完成方法"""
        return await self._send_request(config, messages)

    async def responser(self, request_type: RequestType, messages: List[dict]) -> str:
        """根据请求类型从LLM获取响应"""
        config_mapping = {
            RequestType.OPTIMIZE: self.optimize_config,
            RequestType.EVALUATE: self.evaluate_config,
            RequestType.EXECUTE: self.execute_config,
            RequestType.ANALYZE: self.analyze_config,
        }

        config = config_mapping.get(request_type)
        if not config:
            raise ValueError(
                f"Invalid request type. Valid types: {', '.join([t.value for t in RequestType])}")

        response = await self.acompletion(config, messages)
        return response.choices[0].message.content

    @classmethod
    def initialize(cls,
                   optimize_kwargs: dict = None,
                   evaluate_kwargs: dict = None,
                   execute_kwargs: dict = None,
                   analyze_kwargs: dict = None) -> None:
        """初始化全局实例"""
        if optimize_kwargs is None:
            optimize_kwargs = {"model": MODEL_NAME}
        if evaluate_kwargs is None:
            evaluate_kwargs = {"model": MODEL_NAME}
        if execute_kwargs is None:
            execute_kwargs = {"model": MODEL_NAME}
        if analyze_kwargs is None:
            analyze_kwargs = {"model": MODEL_NAME}

        cls._instance = cls(optimize_kwargs, evaluate_kwargs, execute_kwargs, analyze_kwargs)

    @classmethod
    def get_instance(cls) -> "SPO_LLM":
        """获取全局实例"""
        if cls._instance is None:
            # 如果没有初始化，使用默认设置自动初始化
            cls.initialize()
            print("Warning: SPO_LLM auto-initialized with default settings")
        return cls._instance


def extract_content(text: str, tag: str) -> Optional[str]:
    """提取文本中<tag>...</tag>标签内的内容"""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


async def main():
    # 使用线上API模型配置初始化
    SPO_LLM.initialize(
        optimize_kwargs={
            "model": MODEL_NAME,
            "base_url": DEFAULT_API_URL,
            "temperature": TEMPERATURE,
            "max_tokens": 1000,
            "api_key": API_KEY  # 传递API key
        },
        evaluate_kwargs={
            "model": MODEL_NAME,
            "base_url": DEFAULT_API_URL,
            "temperature": TEMPERATURE,
            "max_tokens": 500,
            "api_key": API_KEY  # 传递API key
        },
        execute_kwargs={
            "model": MODEL_NAME,
            "base_url": DEFAULT_API_URL,
            "temperature": TEMPERATURE,
            "max_tokens": 500,
            "api_key": API_KEY  # 传递API key
        },
        analyze_kwargs={
            "model": MODEL_NAME,
            "base_url": DEFAULT_API_URL,
            "temperature": TEMPERATURE,
            "max_tokens": 500,
            "api_key": API_KEY  # 传递API key
        },
    )

    llm = SPO_LLM.get_instance()

    # 测试消息
    hello_msg = [{"role": "user", "content": "hello"}]

    print("\nTesting EXECUTE request type:")
    response = await llm.responser(request_type=RequestType.EXECUTE, messages=hello_msg)
    print(f"AI: {response}")

    print("\nTesting OPTIMIZE request type:")
    response = await llm.responser(request_type=RequestType.OPTIMIZE, messages=hello_msg)
    print(f"AI: {response}")

    print("\nTesting EVALUATE request type:")
    response = await llm.responser(request_type=RequestType.EVALUATE, messages=hello_msg)
    print(f"AI: {response}")

    print("\nTesting ANALYZE request type:")
    response = await llm.responser(request_type=RequestType.ANALYZE, messages=hello_msg)
    print(f"AI: {response}")


if __name__ == "__main__":
    asyncio.run(main())
