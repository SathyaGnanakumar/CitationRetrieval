import re
import os

def read_paper(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_sections(latex_content):
    # Simple extraction of Abstract and Introduction
    abstract_match = re.search(r'\\section\{Abstract\}(.*?)\\section', latex_content, re.DOTALL)
    intro_match = re.search(r'\\section\{Introduction\}(.*?)\\section', latex_content, re.DOTALL)
    
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    intro = intro_match.group(1).strip() if intro_match else ""
    
    # Clean up LaTeX commands (basic)
    abstract = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', abstract)
    abstract = re.sub(r'\\[a-zA-Z]+', '', abstract)
    intro = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', intro)
    intro = re.sub(r'\\[a-zA-Z]+', '', intro)
    
    return abstract, intro

def generate_prompt_description(abstract, intro):
    # Construct a persona/instruction based on the paper
    prompt = f"""
You are an expert citation retrieval system as described in the paper "Multi-Agent System for Reliable Citation Retrieval".
Your goal is to autonomously retrieve, verify, and recommend academic references given a query or document excerpt.

Abstract Context:
{abstract}

Introduction Context:
{intro[:500]}...

Task:
Given a citation context from a scientific paper (where a citation is missing), identify the correct paper from a list of candidates.
Analyze the context to understand the specific claim, method, or result being cited.
Then, evaluate each candidate paper to see if it matches the context.
Finally, select the best matching paper.
"""
    return prompt.strip()

def main():
    paper_path = "c:/Users/Kushal/CitationRetrieval/paper"
    if not os.path.exists(paper_path):
        print(f"Error: {paper_path} not found.")
        return

    content = read_paper(paper_path)
    abstract, intro = extract_sections(content)
    
    prompt = generate_prompt_description(abstract, intro)
    
    print("--- GENERATED PROMPT BASED ON PAPER ---")
    print(prompt)
    print("---------------------------------------")
    
    with open("generated_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    print("Saved prompt to generated_prompt.txt")

if __name__ == "__main__":
    main()
