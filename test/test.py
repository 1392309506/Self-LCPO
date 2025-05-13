from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="111111"
)

response = client.chat.completions.create(
    model="deepseek-r1:7b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "what is quick sort algrithom"}
    ],
    temperature=0.7,
    stream=True, )
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)