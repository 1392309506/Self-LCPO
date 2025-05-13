from openai import OpenAI

client = OpenAI(
    base_url= "https://api.chatanywhere.tech",
    api_key= "sk-2xGzyYL3nQDvw3XS9SjoDKn5aNucUxYil16rci6Gcr358zyJ",
)

response = client.chat.completions.create(
    model="o3-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": """
        Please think step by step
        Ensure the response concludes with the answer in the XML format:
        <answer>Yes or No</answer>
        Think for 2100 tokens.
        Would an Evander Holyfield 2020 boxing return set age record
        """}
    ],
    temperature=0,
    stream=True, )
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)