BLANK_PROMPT="""
Ensure the response concludes with the answer in the XML format:
<answer>A, B, C or D</answer>
"""


COT_PROMPT = """
Please think step by step.
Ensure the response concludes with the answer in the XML format:
<answer>A, B, C or D</answer>.
"""



SPO_PROMPT = """
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
"""