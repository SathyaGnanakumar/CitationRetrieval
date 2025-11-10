"""
Test CiteAgent-Local without any API keys
"""

import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent))
repo_root = Path(__file__).resolve().parents[1]  # /.../CitationRetrieval
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from evaluation import CitationEvaluator, CitationDataLoader
from evaluation.models.citeagent_model import CiteAgentLocal
from evaluation.models.bm25_model import BM25Model


def main():
    print("="*70)
    print("TESTING LOCAL CITEAGENT (NO API KEYS NEEDED)")
    print("="*70)

    # Load dataset
    print("\nLoading dataset...")
    loader = CitationDataLoader("datasets/scholar_copilot_eval_data_1k.json")
    loader.load_data()
    examples = loader.extract_examples()[:50]  # Test on 50 examples
    
    print(f"Testing on {len(examples)} examples")
    
    # Initialize models
    print("\nInitializing models...")
    citeagent_local = CiteAgentLocal(use_llm=False)
    bm25 = BM25Model()
    
    # Evaluate
    evaluator = CitationEvaluator(
        fuzzy_match_threshold=0.85,
        track_latency=True,
        log_failures=True
    )
    
    # Compare models
    models = {
        'CiteAgent-Local': citeagent_local,
        'BM25': bm25
    }
    
    print("\nRunning comparison...")
    comparison = evaluator.compare_models(
        models=models,
        examples=examples,
        top_k=10,
        verbose=True
    )
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS (NO API KEYS USED)")
    print("="*70)
    
    for model_name, metrics in comparison['results'].items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric:20s}: {value:.4f}")


if __name__ == "__main__":
    main()