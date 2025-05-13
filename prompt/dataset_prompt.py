MATH_PROMPT= """
Please think step by step.
Ensure the response concludes with the answer in the XML format:
<answer>your_answer_here</answer>
Think for {count} tokens.
"""

GPQA_PROMPT ="""
Please think step by step.
Ensure the response concludes with the answer in the XML format:
<answer>A, B, C, or D</answer>.
Think for {count} tokens.
"""

WSC_PROMPT = """
Please think step by step.
Ensure the response concludes with the answer in the XML format:
<answer>A or B</answer>
Think for {count} tokens.
"""

BBH_PROMPT ="""
Please think step by step.
Ensure the response concludes with the answer in the XML format:
<answer>Yes or No</answer>
Think for {count} tokens.
"""

STR_PROMPT="""
Please think step by step.
Ensure the response concludes with the answer in the XML format:
<answer>Yes or No</answer>
Think for {count} tokens.
"""


BOOLQ_PROMPT="""
Please think step by step.
Ensure the response concludes with the answer in the XML format:
<answer>Yes or No</answer>.
Think for {count} tokens.
"""