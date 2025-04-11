import asyncio
import logging
from openai import AsyncOpenAI
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.logger_utils import LoggerUtil
logger = LoggerUtil.get_logger("ChatLLM")

logging.getLogger("httpx").setLevel(logging.WARNING)

class ChatLLM:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        params: Dict = {},
        debug_http: bool = False,
        api_type: str = "openai",
        name: str = "ChatLLM",
    ) -> None:
        self.api_type = api_type
        self.model = params.get("model", "gpt-3.5-turbo")
        self.temperature = params.get("temperature", 0.7)
        self.max_tokens = params.get("max_tokens", 1024)
        self.total_token = 0
        logger.info("初始化Chat LLM | Name: "+name)
        logger.info(params)

        if debug_http:
            logging.getLogger("httpx").setLevel(logging.INFO)

        if api_type == "openai":
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=300,
            )

        elif api_type == "qwq":
            model_path = params.get("model_path", "/root/qwq32b")
            device = params.get("device", "cuda:1")
            if device.startswith("cuda") and not torch.cuda.is_available():
                print("⚠️ CUDA 不可用，自动降级到 CPU")
                device = "cpu"

            print(f"Loading QWQ model from {model_path} on {device}")
            self.model_qwq = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            )
            self.model_qwq.to(device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side="left",
                trust_remote_code=True
            )

        elif api_type == "ollama":
            self.ollama_url = base_url or "http://localhost:11434/api/generate"

        else:
            raise ValueError(f"Unsupported api_type: {api_type}")

    async def chat(self, messages: List[Dict], max_retries: int = 3):
        prompt = messages[-1]["content"]

        if self.api_type == "openai":
            return await self._generate_openai(messages, max_retries)

        elif self.api_type == "qwq":
            return await self._generate_qwq(prompt)

        # elif self.api_type == "ollama":
        #     return await self._generate_ollama(prompt)

    async def _generate_qwq(self, prompt: str):
        loop = asyncio.get_event_loop()

        def _run():
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = inputs.to(self.model_qwq.device)

            if "attention_mask" not in inputs:
                inputs["attention_mask"] = (inputs["input_ids"] != self.tokenizer.pad_token_id).long()

            with torch.no_grad():
                outputs = self.model_qwq.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # ✅ Token统计
            input_ids = inputs["input_ids"]
            input_token_count = input_ids.shape[1]
            output_token_count = outputs.shape[1] - input_ids.shape[1]
            self.total_token += input_token_count + output_token_count

            response_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            return response.strip()

        content = await loop.run_in_executor(None, _run)
        return type('Message', (), {'content': content})()

    async def _generate_openai(self, messages: List[Dict], max_retries: int):
        retries = 0
        while retries < max_retries:
            try:
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                self.total_token+=completion.usage.total_tokens
                return completion.choices[0].message
            except Exception as e:
                print(f"Error occurred: {e}, retrying... ({retries + 1}/{max_retries})")
                retries += 1
                await asyncio.sleep(1)
        return None

    async def __call__(self, system_prompt: str = None, content: List[Dict] = None, max_retries: int = 3):
        messages = [{"role": "system", "content": system_prompt}, *content]
        return await self.chat(messages, max_retries)

    def get_total_token(self) -> int:
        """
        返回当前累计的 token 总花销
        """
        return self.total_token

    def token2zero(self):
        """
        token花销归零
        """
        self.total_token = 0

if __name__ == "__main__":
    async def main():
        llm = ChatLLM(
            api_type="openai",
            api_key="sk-iX0M9keAJemCgNFqvQMVLyWkcembRT27ix50aymLnvZ18QuT",
            base_url="https://api.chatanywhere.tech",
            params={
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1024
            }
        )

        messages = [{"role": "user", "content": "写一个Python快排函数"}]
        response = await llm.chat(messages)

        print("\nResponse:")
        print(response.content)

    asyncio.run(main())