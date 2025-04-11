EVALUATE_PROMPT = """You are comparing the performance of the same LLM with different numbers of tokens.
For each "count" generate a response by answering multiple QA pairs. Each "A" is the answer to its corresponding "Q".
There are {cnt_answers} counts. You must rank **all of them**. Return the index of count from best to worst.
Conclude your response with the final answer formatted in XML:<ranking>For example: [2, 0, 1, 3].</ranking>

### Reference QA pairs:
{reference_block}

### QA responses from each count:
{answer_block}

### Token count list:
{token_list}

"""