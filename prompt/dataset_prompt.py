MATH_PROMPT= """
First, estimate the minimum number of tokens needed to reason through the following question correctly.
Then, reason step-by-step within this estimated budget.

Ensure the response concludes with the answer in the XML format:
<answer>your_answer_here</answer>
"""

GPQA_PROMPT ="""
First, estimate the minimum number of tokens needed to reason through the following question correctly.
Then, reason step-by-step within this estimated budget.

Ensure the response concludes with the answer in the XML format:
<answer>A, B, C, or D</answer>. (for example <answer>A</answer>)
"""

WSC_PROMPT = """
First, estimate the minimum number of tokens needed to reason through the following question correctly.
Then, reason step-by-step within this estimated budget.

Ensure the response concludes with the answer in the XML format:
<answer>A or B</answer>
"""

BBH_PROMPT ="""
First, estimate the minimum number of tokens needed to reason through the following question correctly.
Then, reason step-by-step within this estimated budget.

Ensure the response concludes with the answer in the XML format:
<answer>Yes or No</answer>
"""

STR_PROMPT="""
First, estimate the minimum number of tokens needed to reason through the following question correctly.
Then, reason step-by-step within this estimated budget.

Ensure the response concludes with the answer in the XML format:
<answer>Yes or No</answer>
"""


BOOLQ_PROMPT="""
First, estimate the minimum number of tokens needed to reason through the following question correctly.
Then, reason step-by-step within this estimated budget.

Ensure the response concludes with the answer in the XML format:
<answer>[Yes or No]</answer>.
"""