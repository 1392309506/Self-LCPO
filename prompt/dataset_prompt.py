MATH_PROMPT= """
Please analyze the question in depth and break down the problem into clear, concise steps. \nUse bullet points or numbered lists to ensure clarity and provide logical reasoning with any required calculations or deductions. \nConclude your response with the final answer formatted in XML: <answer>your_answer_here</answer>.\n
"""

GPQA_PROMPT ="""
Please think step by step.
Ensure the response concludes with the answer in the XML format:
<answer>A, B, C, or D</answer>.
"""

WSC_PROMPT = """
Identify who 'they' refers to in the following sentence: \"The city councilmen refused the demonstrators a permit because they advocated violence. Options: A. The city councilmen B. The demonstrators\">
"""

BBH_PROMPT ="""
First, estimate the minimum number of tokens needed to reason through the following question correctly.
Then, reason step-by-step within this estimated budget.

Ensure the response concludes with the answer in the XML format:
<answer>Yes or No</answer>
"""

STR_PROMPT="""
present your answer in the specified XML format: <answer>[Yes or No]</answer>.\n
"""


BOOLQ_PROMPT="""
First, estimate the minimum number of tokens needed to reason through the following question correctly.
Then, reason step-by-step within this estimated budget.

Ensure the response concludes with the answer in the XML format:
<answer>[Yes or No]</answer>.
"""