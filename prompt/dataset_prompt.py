MATH_PROMPT= """
Please analyze the question in depth and break down the problem into clear, concise steps. \nUse bullet points or numbered lists to ensure clarity and provide logical reasoning with any required calculations or deductions. \nConclude your response with the final answer formatted in XML: <answer>your_answer_here</answer>.\n
Think for {count} tokens.
"""

GPQA_PROMPT ="""
Please think step by step.
Ensure the response concludes with the answer in the XML format:
<answer>A, B, C or D</answer>.
Think for {count} tokens.
"""

WSC_PROMPT = """
You are a versatile AI problem solver capable of addressing any question type—multiple-choice, quantitative, conceptual, or hybrid. For each user query, apply the DISTIL framework:

1. Distill  
 • Restate the question in your own words to confirm understanding and clarify objectives.  

2. Identify  
 • List the key elements, variables, or answer choices that are relevant.  

3. Solve  
 • Work through the problem step by step, applying logic, relevant knowledge, or formulas in a clear, structured manner.  

4. Test  
 • Validate your intermediate results or eliminate implausible options to ensure the reasoning is sound.  

5. Inference  
 • Derive the final answer based on your validated reasoning.  

6. Logistics  
 • Summarize your critical reasoning or calculations in concise bullet points.  
 • End by outputting only the final answer wrapped in a single XML tag.

Adapt your Solve and Test phases to suit the question type:  
 • Multiple‐choice: explicitly evaluate each option’s merits.  
 • Quantitative: show all mathematical steps and checks.  
 • Conceptual/open‐ended: present a clear explanatory narrative.

At the very end, output only the final answer in XML format:
<answer>[A, B, or your solution]</answer>
Think for {count} tokens.
"""

BBH_PROMPT ="""
You are an AI assistant that adapts to any question format (yes/no, multiple choice, numeric, free text, code, table, JSON, etc.). For each user query, adhere to the following:

1. Internally parse and reason about the question—do not expose your chain-of-thought.
2. Identify the required response format.
3. Provide only the final result as an XML answer tag on its own line:
   - Yes/No: `<answer>Yes</answer>` or `<answer>No</answer>`
   - Multiple choice: `<answer>ChosenOptionText</answer>`
   - Numeric: `<answer>42</answer>`
   - Free text: `<answer>Your response here</answer>`
   - Code: include only the code block, then on a new line `<answer>See code above</answer>`
   - Tables/JSON/other structured data: wrap the exact output inside `<answer>…</answer>`
4. Do not include any additional text, headings, or commentary outside the final `<answer>…</answer>` tag.
Think for {count} tokens.
"""

STR_PROMPT="""
You are a versatile and precise AI assistant. For every question, follow these steps:

1. Understand  
   • Restate the user’s question in your own words to confirm meaning.

2. Classify  
   • Identify the question type (evidence check, comparison, prediction, definition, decision, open-ended, etc.).  
   • Note any assumptions needed to interpret the query.

3. Select Framework  
   • Evidence-Based Reasoning for factual verification  
   • Comparative Analysis for “which is greater/better”  
   • Scenario Planning for future projections  
   • Definition Framework for terminology  
   • Pros & Cons for decision-related judgments  
   • Exploratory Synthesis for open-ended or novel inquiries  

4. Structured Analysis  
   a. List key facts, data, or definitions.  
   b. Draw logical deductions or weigh evidence.  
   c. Present counterpoints, uncertainties, or limitations.

5. Conclusion  
   • Provide a clear “Yes” or “No” answer based on your analysis.

6. Final Output  
   • On its own line, output exactly one XML tag with your verdict—no additional text or punctuation:  
     <answer>Yes</answer>  
     or  
     <answer>No</answer>  
Ensure this procedure applies uniformly to every question.
"""


BOOLQ_PROMPT="""
Please think step by step.
Ensure the response concludes with the answer in the XML format:
<answer>Yes or No</answer>.
Think for {count} tokens.
"""