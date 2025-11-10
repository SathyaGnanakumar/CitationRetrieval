"""
Quick example demonstrating the evaluation framework.

This script shows how to:
1. Load data
2. Create model instances
3. Run evaluation
4. Analyze results

Usage:
    uv run main.py [--num-examples N] [--top-k K]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluation import CitationEvaluator, CitationDataLoader
from evaluation.models import BM25Model, CiteAgentLocal


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate citation retrieval models on Scholar Copilot dataset"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=50,
        help="Number of examples to evaluate (default: 50)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top candidates to retrieve (default: 20)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["citeagent", "bm25"],
        default="citeagent",
        help="Model to use for evaluation (default: citeagent)"
    )
    args = parser.parse_args()
    print("=" * 70)
    print("CITATION RETRIEVAL EVALUATION - QUICK EXAMPLE")
    print("=" * 70)

    # 1. Load data
    print("\n📂 Step 1: Loading data...")
    loader = CitationDataLoader("datasets/scholar_copilot_eval_data_1k.json")
    loader.load_data()
    examples = loader.extract_examples()

    # Use subset based on CLI argument
    examples = examples[:args.num_examples]
    print(f"   Using {len(examples)} examples for evaluation")

    # Show dataset statistics
    print("\n📊 Dataset info:")
    stats = loader.get_statistics()
    print(f"   Total examples in full dataset: {stats['total_examples']}")
    print(f"   Avg citations per paper: {stats['avg_citations_per_paper']:.1f}")
    print(f"   Avg corpus size: {stats['avg_corpus_size']:.1f}")

    # 2. Initialize model
    print(f"\n🤖 Step 2: Initializing {args.model.upper()} model...")
    if args.model == "citeagent":
        model = CiteAgentLocal(use_llm=False)
    else:  # bm25
        model = BM25Model(use_stemming=True, use_stopwords=True)
    print(f"   Model config: {model.get_config()}")

    # 3. Run evaluation
    print("\n🔍 Step 3: Running evaluation...")
    evaluator = CitationEvaluator(
        fuzzy_match_threshold=0.85,
        track_latency=True,
        log_failures=True
    )

    results = evaluator.evaluate_model(
        model=model,
        examples=examples,
        top_k=args.top_k,
        verbose=True
    )

    # 4. Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(results['metrics'])

    # Show some example predictions
    print("\n📋 Sample predictions:")
    for i, pred in enumerate(results['predictions'][:3], 1):
        print(f"\nExample {i}:")
        print(f"  True title: {pred['true_titles'][0][:60]}...")
        print(f"  Top-3 retrieved:")
        for rank, title in enumerate(pred['retrieved_titles'][:3], 1):
            marker = "✓" if title.lower() in pred['true_titles'][0].lower() else " "
            print(f"    {rank}. {marker} {title[:60]}...")

    # Failure analysis
    if results['failure_log']:
        print(f"\n⚠️  Failures: {len(results['failure_log'])}/{len(examples)}")
        print("\nExample failure:")
        failure = results['failure_log'][0]
        print(f"  Query ID: {failure['query_id']}")
        print(f"  True title: {failure['true_title'][:60]}...")
        print(f"  Context: {failure['context'][:100]}...")
        print(f"  Top-3 retrieved:")
        for i, title in enumerate(failure['retrieved_titles'][:3], 1):
            print(f"    {i}. {title[:60]}...")

    # 5. Save results
    print("\n💾 Step 4: Saving results...")
    evaluator.save_results(
        results=results,
        output_dir="results/example",
        model_name=f"{args.model}_demo",
        include_predictions=True
    )

    print("\n" + "=" * 70)
    print("✅ EXAMPLE COMPLETE!")
    print("=" * 70)
    print("\nGenerated files in results/example/:")
    print(f"  - {args.model}_demo_*_metrics.json")
    print(f"  - {args.model}_demo_*_failures.json")
    print(f"  - {args.model}_demo_*_predictions.json")
    print("\nNext steps:")
    print("  1. Run with different settings: uv run main.py --num-examples 100 --top-k 10")
    print("  2. Try BM25: uv run main.py --model bm25")
    print("  3. Compare multiple models with evaluator.compare_models()")


if __name__ == "__main__":
    main()
