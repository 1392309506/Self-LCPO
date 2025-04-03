import anthropic
import os 
from dotenv import load_dotenv 
from typing import List,Dict 
import asyncio
from anthropic import AsyncAnthropic

class ChatClaude:
    def __init__(self) -> None:
        # Load environment variables from a .env file
        load_dotenv()
        self.async_client = AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    
    async def __call__(self, system_prompt: str = None, content: List[Dict] = None, max_retries: int = 3):
        retries = 0
        while retries < max_retries:
            try:
                completion = await self.async_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    system=system_prompt,
                    messages=[*content],
                    temperature=0.5,
                    max_tokens=8000
                )
                return completion.content
            except Exception as e:
                print(f"Error occurred: {e}, retrying... ({retries + 1}/{max_retries})")
                retries += 1
                await asyncio.sleep(1)
        return None


if __name__ == "__main__":
    import asyncio
    
    async def main():
        llm = ChatClaude()
        system_prompt = "You are a helpful assistant."
        content = [{"role": "user", "content": "Hello, how are you?"}]
        res = await llm(system_prompt=system_prompt, content=content)
        print(res[0].text)

    asyncio.run(main())