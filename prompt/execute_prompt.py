BLANK_PROMPT="""
Conclude your response with the final answer formatted in XML: <answer>your_answer_here</answer>.
"""


COT_PROMPT = """
Please think step by step.
Ensure the response concludes with the answer in the XML format:
<answer>A, B, C or D</answer>
"""



SPO_PROMPT = """
You are a highly skilled problem solver prepared to tackle any type of multiple-choice question with options (A, B, C, or D) and various formats. For every question, perform a complete internal analysis using your hidden chain-of-thought process, but do not reveal any internal reasoning or intermediate steps in your response. Your final response should contain only one XML-tagged final answer, formatted exactly as <answer>(X)</answer> where X is the correct option. This rule applies regardless of question type or additional formatting.
"""