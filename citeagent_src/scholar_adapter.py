"""Adapter to run CiteAgent on ScholarCopilot dataset"""

import json
import pandas as pd
from typing import List, Dict
from pathlib import Path


def load_scholar_copilot_data(json_path: str) -> pd.DataFrame:
    """
    Load ScholarCopilot dataset and convert to CiteAgent format.
    
    Args:
        json_path: Path to scholar_copilot_eval_data_1k.json
        
    Returns:
        DataFrame compatible with CiteAgent
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to CiteAgent format
    citations = []
    
    for paper_id, paper_data in data.items():
        # Extract citation contexts
        contexts = paper_data.get('citation_contexts', [])
        
        for context in contexts:
            # Create CiteAgent-compatible entry
            citation = {
                'id': f"{paper_id}_{context.get('cite_idx', 0)}",
                'excerpt': context.get('context', ''),
                'target_paper_title': context.get('true_title', ''),
                'source_paper_title': paper_data.get('title', ''),
                'year': paper_data.get('year', 2024),
                'split': 'test'  # Default to test
            }
            citations.append(citation)
    
    return pd.DataFrame(citations)


def run_citeagent_on_scholar(
    data_path: str,
    output_path: str,
    model_name: str = "gpt-4o",
    max_examples: int = None
):
    """
    Run CiteAgent on ScholarCopilot dataset.
    
    Args:
        data_path: Path to scholar_copilot_eval_data_1k.json
        output_path: Path to save results
        model_name: LLM model to use
        max_examples: Limit number of examples (for testing)
    """
    # Import CiteAgent components
    from retriever.agent import LLMSelfAskAgentPydantic, Output
    from utils.str_matcher import find_match_psr
    import json
    from time import time
    
    # Load and convert data
    print(f"Loading data from {data_path}...")
    df = load_scholar_copilot_data(data_path)
    
    if max_examples:
        df = df.head(max_examples)
    
    print(f"Processing {len(df)} citations...")
    
    # Initialize agent
    agent = LLMSelfAskAgentPydantic(
        model_name=model_name,
        temperature=0.95,
        search_limit=10,
        only_open_access=False,
        prompt_name="few_shot_search",
        pydantic_object=Output
    )
    
    results = []
    
    for idx, row in df.iterrows():
        print(f"Processing {idx+1}/{len(df)}: {row['id']}")
        
        # Reset agent
        agent.reset([row['source_paper_title']])
        
        start_time = time()
        result_data = {
            'id': row['id'],
            'target_title': row['target_paper_title'],
            'excerpt': row['excerpt']
        }
        
        try:
            # Run agent
            selection = agent(
                row['excerpt'],
                str(row['year']),
                max_actions=15
            )
            
            result_data['selected_title'] = selection.title
            result_data['is_correct'] = find_match_psr(
                [row['target_paper_title']], 
                selection.title, 
                0.8
            ) is not None
            result_data['status'] = 'success'
            
        except Exception as e:
            result_data['selected_title'] = None
            result_data['is_correct'] = False
            result_data['status'] = 'error'
            result_data['error'] = str(e)
        
        result_data['duration'] = time() - start_time
        results.append(result_data)
        
        # Save incrementally
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    # Calculate accuracy
    correct = sum(1 for r in results if r['is_correct'])
    total = len(results)
    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
    
    return results