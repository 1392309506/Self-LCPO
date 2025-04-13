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
