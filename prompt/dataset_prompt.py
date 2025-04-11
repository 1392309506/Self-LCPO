MATH_PROMPT= """
Please analyze the question in depth and break down the problem into clear, concise steps. 
Use bullet points or numbered lists to ensure clarity and provide logical reasoning with any required calculations or deductions. 
Conclude your response with the final answer formatted in XML: <answer>your_answer_here</answer>. Thinf for {count} tokens.
"""


GPQA_PROMPT ="""
Please analyze the question thoroughly and break down your reasoning into clear, logical steps. Begin by
identifying the key components of the question, and then evaluate each option systematically. Apply relevant
principles or concepts that may be pertinent to the question. At the end of your analysis, present your final
answer in the required XML format: <answer>[A or B or C or D]</answer>;, ensuring that the choice is in
uppercase letters to match the specified format. For example, if the correct choice is A, format your answer
as<answer>A</answer>. Think for {count} tokens.
"""