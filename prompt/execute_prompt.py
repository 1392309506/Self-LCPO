BLANK_PROMPT="""
Conclude your response with the final answer formatted in XML: <answer>your_answer_here</answer>.
"""


COT_PROMPT = """
<question>There are 15 trees in the grove. Grove workers will plant trees today. After they are done, there will be 21 trees. How many trees did they plant today?</question>
<analysis>There were originally 15 trees. After planting, there are 21 trees. So they planted 21 - 15 = 6 trees. The answer is 6.</analysis>
<answer>6</answer>

<question>Roger has 5 tennis balls. He buys 2 more cans with 3 balls each. How many balls does he have now?</question>
<analysis>Roger started with 5 balls. 2 cans of 3 balls each is 6 balls. 5 + 6 = 11. The answer is 11.</analysis>
<answer>11<answer>

<question>{question}</question>
<analysis>Think step by step</analysis>
Conclude your response with the final answer formatted in XML: <answer>your_answer_here</answer>.
"""



SPO_PROMPT = """
Please think step by step.\nEnsure the response concludes with the answer in the XML format: \n<answer>[Yes or No]</answer>.\n
"""