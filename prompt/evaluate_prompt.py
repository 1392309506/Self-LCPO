EVALUATE_PROMPT = """You are comparing the performance of the same LLM with different numbers of tokens.
For each "count" generate a response by answering multiple QA pairs. Each "A" is the answer to its corresponding "Q".

Please evaluate all counts based on the following four criteria:
  - Correctness
  - Reasoning Quality
  - Completeness
  - Clarity

There are {cnt_answers} counts. You must rank **all of them** from best to worst.

### QA responses from each count:
```
{answer_block}
```

### Token count list:
```
{token_list}
```

Return your response in the following XML format:

<response>
<analysis>
Discuss the differences in quality across counts. Highlight which performed best and why.
</analysis>
<ranking>
Return the index of count from best to worst. For example: [2, 0, 1, 3]. That means index=2 has the best performance.
You must provide a int list. Don't provide None or other values.
</ranking>
</response>
"""