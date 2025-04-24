MATH_PROMPT= """
Please think step by step.
Ensure the response concludes with the answer in the XML format: 
<answer>your_answer_here</answer>.
Think for {count} tokens.
"""

GPQA_PROMPT ="""
Please think step by step.
Ensure the response concludes with the answer in the XML format:
<answer>[A, B, C, or D]</answer>.\nThink for {count} tokens.
"""

WSC_PROMPT = """
Please think step by step.
Ensure the response concludes with the answer in the XML format:
<answer>[A or B]</answer>
Think for {count} tokens.
"""

BBH_PROMPT ="""
You are an advanced problem-solving assistant designed to handle a broad spectrum of questions, be they spatial, mathematical, logical, or of any other nature. For every query, internally use robust, multi-step reasoning without revealing any of your internal chain-of-thought to the user. Your response must completely omit all details of your reasoning process and provide only the final answer. The final answer must be a single word—either "Yes" or "No"—and must appear at the end of your response enclosed exactly within XML tags as follows:
<answer>(Yes or No)</answer>
Ensure every response, regardless of the question's complexity or topic, follows this strict formatting and leaves no extraneous information.
Think for {count} tokens.
"""

STR_PROMPT="""
When you receive a question of any type, begin by thoroughly analyzing the issue: break it down into key components, evaluate all pertinent data and factors, and explain your reasoning with clear, step-by-step processing. Ensure your analysis is comprehensive enough for complex questions but remains succinct for simpler ones. After this explanation, include a clear separator on a new line reading “Final Answer:” (with no additional text following). Then, provide your definitive response strictly in the XML format <answer>(Yes or No)</answer> (for example, <answer>Yes</answer> or <answer>No</answer>). Ensure that no additional text appears after the final XML tag, and this structure must adapt to various question types.
Think for {count} tokens.
"""


BOOLQ_PROMPT="""
Please think step by step.\nEnsure the response concludes with the answer in the XML format: \n<answer>[Yes or No]</answer>.\nThink for {count} tokens.
"""