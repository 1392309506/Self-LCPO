MATH_PROMPT= """
Please think step by step.Think for {count} tokens.
Ensure the response concludes with the answer in the XML format:
<answer>your_answer_here</answer>.
"""

GPQA_PROMPT ="""
Please think step by step and try to think for {count} tokens.
Ensure the response concludes with the answer in the **XML format**:
<answer>A, B, C, or D</answer>. (for example <answer>A</answer>)
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
Please think step by step.\nEnsure the response concludes with the answer in the XML format: \n<answer>[Yes or No]</answer>.\nThink for {count} tokens.
"""