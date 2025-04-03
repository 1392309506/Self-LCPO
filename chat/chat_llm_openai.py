import os
import asyncio
import logging
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Dict
logging.getLogger("httpx").setLevel(logging.WARNING)
class ChatLLM:
    def __init__(self, api_key:str="", base_url: str="", debug_http: bool = False) -> None:
        # Load environment variables from .env
        load_dotenv()

        # 可选启用 httpx 日志
        if debug_http:
            httpx_logger = logging.getLogger("httpx")
            httpx_logger.setLevel(logging.INFO)

        # self.client = AsyncOpenAI(
        #     api_key=os.getenv('apikey'),
        #     base_url=os.getenv('base_url')
        # )
        self.client = AsyncOpenAI(
            api_key= api_key,
            base_url= base_url
        )

    async def chat(self, messages: List[Dict], max_retries: int = 3):
        retries = 0
        while retries < max_retries:
            try:
                completion = await self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=1.2,
                    max_tokens=4096
                )
                return completion.choices[0].message
            except Exception as e:
                print(f"Error occurred: {e}, retrying... ({retries + 1}/{max_retries})")
                retries += 1
                await asyncio.sleep(1)
        return None

    async def __call__(self, system_prompt: str = None, content: List[Dict] = None, max_retries: int = 3):
        messages = [
            {"role": "system", "content": system_prompt},
            *content
        ]
        return await self.chat(messages, max_retries)

if __name__ == "__main__":
    async def main():
        llm = ChatLLM(
            "sk-iX0M9keAJemCgNFqvQMVLyWkcembRT27ix50aymLnvZ18QuT",
            "https://api.chatanywhere.tech"
        )
        # 测试 chat 方法
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        res = await llm.chat(messages)
        print(res)

    asyncio.run(main())