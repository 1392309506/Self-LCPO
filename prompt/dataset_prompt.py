MATH_PROMPT= """
Please analyze the question in depth and break down the problem into clear, concise steps. 
Use bullet points or numbered lists to ensure clarity and provide logical reasoning with any required calculations or deductions. 
Conclude your response with the final answer formatted in XML: <answer>your_answer_here</answer>. 
Think for {count} tokens.
"""


GPQA_PROMPT ="""
Please think step by step.\nEnsure the response concludes with the answer in the XML format: \n<answer>[A, B, C or D]</answer>.\nThink for {count} tokens.
"""

BBH_PROMPT ="""
Please think step by step.\nEnsure the response concludes with the answer in the XML format: \n<answer>[Yes or No]</answer>.\n
Think for {count} tokens.
"""


LIAR_PROMPT ="""
Please think step by step.\nEnsure the response concludes with the answer in the XML format: \n<answer>[true, mostly-true, half-true, barely-true, false, pants-fire]</answer>.\n
Think for {count} tokens.
"""

WSC_PROMPT = """
Please think step by step.
Ensure the response concludes with the answer in the XML format: 
<answer>[A or B]</answer>.
Think for {count} tokens.
"""