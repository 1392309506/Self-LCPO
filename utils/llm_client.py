import asyncio
import re
import json
import torch
import aiohttp
from enum import Enum
from typing import Any, Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model Configuration
QWQ_MODEL_PATH = "/root/qwq32b"
DEVICE = "cuda:0"
TEMPERATURE = 0.7  # Default temperature parameter

# OLLama Local API Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Local OLLama server URL
OLLAMA_MODEL_NAME = "llama3.1"  # Default model name


class RequestType(Enum):
    OPTIMIZE = "optimize"
    EVALUATE = "evaluate"
    EXECUTE = "execute"
    ANALYZE = "analyze"
    GENERATE = "generate"  # Added for prompt variant generation

class LLMResponse:
    """封装LLM的响应结果。"""
    def __init__(self, content: str):
        self.choices = [self.Choice(content)]

    class Choice:
        def __init__(self, content: str):
            self.message = self.Message(content)

        class Message:
            def __init__(self, content: str):
                self.content = content
class QWQ_LLM:
    """LLM客户端封装类，支持QWQ和Ollama两种模型。"""
    _instance: Optional["QWQ_LLM"] = None
    def __init__(
        self,
        optimize_kwargs: Optional[dict] = None,
        evaluate_kwargs: Optional[dict] = None,
        execute_kwargs: Optional[dict] = None,
    ) -> None:
        self.evaluate_llm = LLM(llm_config=self._load_llm_config(evaluate_kwargs))
        self.optimize_llm = LLM(llm_config=self._load_llm_config(optimize_kwargs))
        self.execute_llm = LLM(llm_config=self._load_llm_config(execute_kwargs))
    def __init__(
            self,
            optimize_kwargs: Optional[dict] = None,
            evaluate_kwargs: Optional[dict] = None,
            execute_kwargs: Optional[dict] = None,
            analyze_kwargs: Optional[dict] = None,
            generate_kwargs: Optional[dict] = None,  # Added for variant generation
    ) -> None:
        self.optimize_config = self._prepare_config(optimize_kwargs)
        self.evaluate_config = self._prepare_config(evaluate_kwargs)
        self.execute_config = self._prepare_config(execute_kwargs)
        self.analyze_config = self._prepare_config(analyze_kwargs)
        self.generate_config = self._prepare_config(generate_kwargs)  # New config

        # Initialize QWQ model only if needed
        self.model = None
        self.tokenizer = None
        self.use_ollama = {}

        # Track which configs use Ollama
        for req_type, config in [
            (RequestType.OPTIMIZE, self.optimize_config),
            (RequestType.EVALUATE, self.evaluate_config),
            (RequestType.EXECUTE, self.execute_config),
            (RequestType.ANALYZE, self.analyze_config),
            (RequestType.GENERATE, self.generate_config),  # Added new type
        ]:
            self.use_ollama[req_type] = config.get("use_ollama", False)

        # Only load QWQ model if at least one config uses it
        if not all(self.use_ollama.values()):
            print(f"Loading QWQ model: {QWQ_MODEL_PATH} on {DEVICE}")
            self.model = AutoModelForCausalLM.from_pretrained(
                QWQ_MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map=DEVICE,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                QWQ_MODEL_PATH,
                padding_side="left",
                trust_remote_code=True
            )
            print("QWQ model loaded successfully")

        print(f"QWQ_LLM initialized with configurations:")
        for req_type in RequestType:
            config_type = req_type.value
            if self.use_ollama.get(req_type, False):
                print(
                    f"  - {config_type}: Using Ollama model {self._get_config_for_type(req_type).get('model', OLLAMA_MODEL_NAME)}")
            else:
                print(f"  - {config_type}: Using QWQ model on {DEVICE}")

    def _prepare_config(self, kwargs: Optional[dict]) -> Dict[str, Any]:
        if not kwargs:
            kwargs = {}

        # Detect if using Ollama
        use_ollama = kwargs.get("use_ollama", False)

        # If using Ollama
        if use_ollama:
            config = {
                "use_ollama": True,
                "model": kwargs.get("model", OLLAMA_MODEL_NAME),
                "base_url": kwargs.get("base_url", OLLAMA_API_URL),
                "temperature": kwargs.get("temperature", TEMPERATURE),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }
        # If using QWQ model
        else:
            config = {
                "use_ollama": False,
                "model_path": kwargs.get("model_path", QWQ_MODEL_PATH),
                "device": kwargs.get("device", DEVICE),
                "temperature": kwargs.get("temperature", TEMPERATURE),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }

        return config

    def _get_config_for_type(self, request_type: RequestType) -> Dict[str, Any]:
        config_mapping = {
            RequestType.OPTIMIZE: self.optimize_config,
            RequestType.EVALUATE: self.evaluate_config,
            RequestType.EXECUTE: self.execute_config,
            RequestType.ANALYZE: self.analyze_config,
            RequestType.GENERATE: self.generate_config,  # Added new type
        }

        return config_mapping.get(request_type)

    async def _generate_response_qwq(self, config: Dict[str, Any], prompt: str) -> str:
        """使用本地QWQ模型生成响应。"""
        loop = asyncio.get_event_loop()

        def _generate():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                    do_sample=True if config["temperature"] > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Only return the newly generated tokens
            response_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            return response.strip()

        try:
            response = await loop.run_in_executor(None, _generate)
            return response
        except Exception as e:
            error_msg = f"Exception during QWQ model generation: {str(e)}"
            print(error_msg)
            return error_msg

    async def _generate_response_ollama(self, config: Dict[str, Any], prompt: str) -> str:
        """调用Ollama API生成响应。"""
        base_url = config.get("base_url", OLLAMA_API_URL)
        model_name = config.get("model", OLLAMA_MODEL_NAME)
        temperature = config.get("temperature", TEMPERATURE)
        max_tokens = config.get("max_tokens", 1000)

        payload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False  # ✅ 关闭流式返回，确保返回 JSON
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(base_url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return f"Error from Ollama API: {response.status}, {error_text}"

                    result = await response.json()  # ✅ 现在 Ollama 会返回 JSON
                    return result.get("response", "")  # 提取文本
        except Exception as e:
            error_msg = f"Exception during Ollama API call: {str(e)}"
            print(error_msg)
            return error_msg

    async def acompletion(self, request_type: RequestType, messages: List[Dict[str, str]]) -> LLMResponse:
        config = self._get_config_for_type(request_type)
        prompt = messages[-1]["content"]

        # Choose the appropriate generation method based on the config
        if config.get("use_ollama", False):
            response = await self._generate_response_ollama(config, prompt)
        else:
            response = await self._generate_response_qwq(config, prompt)

        return LLMResponse(response)

    async def responser(self, request_type: RequestType, messages: List[dict]) -> str:
        if request_type not in RequestType:
            raise ValueError(
                f"Invalid request type. Valid types: {', '.join([t.value for t in RequestType])}")

        response = await self.acompletion(request_type, messages)
        return response.choices[0].message.content
    async def responser(self, request_type: RequestType, messages: List[dict]) -> str:
        llm_mapping = {
            RequestType.OPTIMIZE: self.optimize_llm,
            RequestType.EVALUATE: self.evaluate_llm,
            RequestType.EXECUTE: self.execute_llm,
        }

        llm = llm_mapping.get(request_type)
        if not llm:
            raise ValueError(f"Invalid request type. Valid types: {', '.join([t.value for t in RequestType])}")

        response = await llm.acompletion(messages)
        return response.choices[0].message.content

    @classmethod
    def initialize(cls, optimize_kwargs: dict, evaluate_kwargs: dict, execute_kwargs: dict) -> None:
        """Initialize the global instance"""
        cls._instance = cls(optimize_kwargs, evaluate_kwargs, execute_kwargs)

    @classmethod
    def get_instance(cls) -> "QWQ_LLM":
        if cls._instance is None:
            cls.initialize()
            print("Warning: QWQ_LLM auto-initialized with default settings")
        return cls._instance


def extract_content(text: str, tag: str) -> Optional[str]:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


async def main():
    # Example with mixed configuration
    QWQ_LLM.initialize(
        # Use QWQ for optimization tasks
        optimize_kwargs={
            "use_ollama": False,
            "model_path": QWQ_MODEL_PATH,
            "device": DEVICE,
            "temperature": 0.7,
            "max_tokens": 1000
        },
        # Use Ollama for evaluation tasks
        evaluate_kwargs={
            "use_ollama": True,
            "model": "llama3.1",
            "base_url": OLLAMA_API_URL,
            "temperature": 0.7,
            "max_tokens": 500
        },
        # Use QWQ for execution tasks
        execute_kwargs={
            "use_ollama": False,
            "model_path": QWQ_MODEL_PATH,
            "device": DEVICE,
            "temperature": 0.7,
            "max_tokens": 1000
        },
        # Use Ollama for analysis tasks
        analyze_kwargs={
            "use_ollama": True,
            "model": "llama3.1",
            "base_url": OLLAMA_API_URL,
            "temperature": 0.7,
            "max_tokens": 500
        },
        # Use QWQ for variant generation
        generate_kwargs={
            "use_ollama": False,
            "model_path": QWQ_MODEL_PATH,
            "device": DEVICE,
            "temperature": 0.8,  # Slightly higher for more creative variants
            "max_tokens": 1500
        }
    )

    llm = QWQ_LLM.get_instance()

    hello_msg = [{"role": "user", "content": "hello"}]

    print("\nTesting EXECUTE request type (QWQ model):")
    response = await llm.responser(request_type=RequestType.EXECUTE, messages=hello_msg)
    print(f"AI: {response}")

    print("\nTesting OPTIMIZE request type (QWQ model):")
    response = await llm.responser(request_type=RequestType.OPTIMIZE, messages=hello_msg)
    print(f"AI: {response}")

    print("\nTesting EVALUATE request type (Ollama model):")
    response = await llm.responser(request_type=RequestType.EVALUATE, messages=hello_msg)
    print(f"AI: {response}")

    print("\nTesting ANALYZE request type (Ollama model):")
    response = await llm.responser(request_type=RequestType.ANALYZE, messages=hello_msg)
    print(f"AI: {response}")

    print("\nTesting GENERATE request type (QWQ model):")
    generate_msg = [{"role": "user", "content": "Generate a prompt for a customer service chatbot"}]
    response = await llm.responser(request_type=RequestType.GENERATE, messages=generate_msg)
    print(f"AI: {response}")


if __name__ == "__main__":
    asyncio.run(main())