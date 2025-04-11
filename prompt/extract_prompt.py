EXTRACT_ANSWER_PROMPT="""
I will first provide you with the standard answer to a question, and then give you a personal answer. 
The answers will be provided in XML format <standard> and <personal>.
You need to judge whether the personal answer is correct or not. 1 means right and 0 means wrong.
Conclude with a definitive answer formatted in XML:<judge>[1 or 0]</judge>.

<standard>{standard}</standard>
<personal>{personal}</personal>
"""

EXTRACT_RANKING_PROMPT="""
I will give you a response from a LLM, which may contain a *int list*, please extract it.
Conclude with a definitive answer formatted in XML:<ranking>For example:[1, 0, 3] or None</ranking>.

Response:
{response}
"""