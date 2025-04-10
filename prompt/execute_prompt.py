EXECUTE_PROMPT = """You are a reasoning agent. Given a question:

1. Analyze it step by step using {count} tokens.
2. Write your reasoning in <analysis>, rich and insightful.
3. Give the final answer in <answer>.
4. Do not explain the format or repeat instructions.

<question>
{question}
</question>

Respond in this format:

<response>
<analysis>
Reasoning here.
</analysis>
<answer>
Final answer.
</answer>
</response>
"""
