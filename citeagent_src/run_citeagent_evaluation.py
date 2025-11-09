"""
Run CiteAgent on ScholarCopilot dataset using the evaluation framework.
"""

import argparse
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "citeagent_src"))

from evaluation import CitationEvaluator, CitationDataLoader
from evaluation.models.citeagent_model import CiteAgentModel
from evaluation.models import BM25Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="BM25/scholar_copilot_eval_data_1k.json",
        help="Path to dataset"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM model for CiteAgent"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit examples for testing"
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        help="Compare with BM25 baseline"
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading dataset...")
    loader = CitationDataLoader(args.data_path)
    loader.load_data()
    examples = loader.extract_examples()
    
    if args.max_examples:
        examples = examples[:args.max_examples]
    
    print(f"Evaluating on {len(examples)} examples")
    
    # Initialize CiteAgent
    citeagent = CiteAgentModel(
        model_name=args.model,
        temperature=0.95,
        search_limit=10,
        max_actions=15
    )
    
    # Run evaluation
    evaluator = CitationEvaluator(
        fuzzy_match_threshold=0.85,
        track_latency=True,
        log_failures=True
    )
    
    results = evaluator.evaluate_model(
        model=citeagent,
        examples=examples,
        top_k=10,
        verbose=True
    )
    
    print("\nCiteAgent Results:")
    print(results['metrics'])
    
    # Compare with baseline if requested
    if args.compare_baseline:
        print("\nComparing with BM25 baseline...")
        
        models = {
            'BM25': BM25Model(),
            'CiteAgent': citeagent
        }
        
        comparison = evaluator.compare_models(
            models=models,
            examples=examples,
            top_k=10
        )
        
        print("\nComparison Results:")
        print(comparison)


if __name__ == "__main__":
    main()