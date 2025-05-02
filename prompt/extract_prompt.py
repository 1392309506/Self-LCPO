EXTRACT_ANSWER_PROMPT="""
I will first provide you with the standard answer to a question, and then give you a personal answer.
The answers will be provided in XML format <standard> and <personal>.
You need to judge whether the personal answer is correct or not. 1 means right and 0 means wrong.
Conclude with a definitive answer formatted in XML:<judge>[1 or 0]</judge>.

<standard>{standard}</standard>
<personal>{personal}</personal>
"""


EXTRACT_RANKING_PROMPT = """
Given a response from an LLM, extract the **ranking list of indices** (e.g., [2, 0, 1]) if clearly present.

- Accept formats like [2, 0, 1] or 2 > 0 > 1.
- Only return a list if ranking is explicit or clearly inferable.
- If no valid list is found, return None.

Respond strictly in this XML format:
<ranking>[list] or None</ranking>

<content>
{response}
</content>
"""


# EXTRACT_PROMPT="""
# Extract the answer from the content of answer process. If there is no explicit output, return None.
# Ensure the response concludes with the answer in the XML format: \n<answer>true, mostly-true, half-true, barely-true, false, pants-fire or None</answer>.\n

# {content}
# """


EXTRACT_PROMPT="""
Extract the **final label** from the answer process.The possible labels are: A or B.
If no such label is mentioned or inferable, return `None`.
Ensure the response concludes with the answer in the XML format:
<answer>A or B</answer>

question:
{question}

answer process:
{content}
"""

EXTRACT_SCORE_PROMPT = """
Your task is to extract a difficulty score from this response, based on a scale from 1 to 10, where 1 represents the easiest possible question and 10 represents the hardest possible question.

Unstructured Response:
{response}

Ensure the response concludes with the answer in the XML format:
<score>1~10</score>
"""

# EXTRACT_PROMPT="""
# You are given an answer process. Your task is to extract the **final truthfulness label** from it.

# The possible labels are:
# - A
# - No

# If no such label is mentioned or inferable, return `None`.
# Ensure the response concludes with the answer in the XML format:
# <answer>Yes or No</answer>

# # question:
# # {question}

# # answer process:
# # {content}
# """