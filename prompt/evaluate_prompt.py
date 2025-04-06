EVALUATE_PROMPT = """You are sorting for different prompts. 
The only difference between these prompts is that they specify the number of tokens they want the LLM to think with. 
Each prompt will give you several QA pairs, where A is their answer to the question. 
Please rate and rank all prompt based on the following criteria:
   - Correctness
   - Reasoning Quality
   - Completeness
   - Clarity

QA pairs for each prompt:
```
{answer_block}
```

The number of tokens used per prompt:
```
{token_list}
```

Provide your analysis, and the ranking using the following XML format:

<response>
<analysis>
Analyze the above token list and find out which number corresponds to the best answer.
</analysis>
<ranking>
Rank the above token_list based on performance, returning the sorted index. A smaller number means better performance. Example: [2, 0, 1]
</ranking>
</response>
"""
