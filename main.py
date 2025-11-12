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
from evaluation.models import BM25Model, DenseRetrievalModel
try:
    from evaluation.models import CiteAgentAPI
    CITEAGENT_AVAILABLE = True
except ImportError:
    CITEAGENT_AVAILABLE = False
    CiteAgentAPI = None
    print("⚠️  Warning: CiteAgentAPI not available. Install dependencies or check API keys.")


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
        choices=["citeagent", "bm25", "dense", "all"],
        default="all",
        help="Model to use for evaluation (default: all)"
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default="intfloat/e5-base-v2",
        help="Dense retrieval model to use (default: intfloat/e5-base-v2)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/scholar_copilot_eval_data_1k.json",
        help="Path to dataset JSON file (default: datasets/scholar_copilot_eval_data_1k.json)"
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="gpt-4o",
        choices=["gpt-4o", "claude-3-5-sonnet-20241022", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"],
        help="LLM backend for CiteAgent (default: gpt-4o)"
    )
    parser.add_argument(
        "--citeagent-search-limit",
        type=int,
        default=10,
        help="Number of papers to return per CiteAgent search (default: 10)"
    )
    parser.add_argument(
        "--citeagent-max-actions",
        type=int,
        default=15,
        help="Maximum actions for CiteAgent per query (default: 15)"
    )
    parser.add_argument(
        "--citeagent-prompt",
        type=str,
        default="few_shot_search",
        choices=["few_shot_search", "few_shot_search_no_read", "one_shot_search", "zero_shot_search"],
        help="Prompt template for CiteAgent (default: few_shot_search)"
    )
    args = parser.parse_args()
    print("=" * 70)
    print("CITATION RETRIEVAL EVALUATION - QUICK EXAMPLE")
    print("=" * 70)

    # 1. Load data
    print("\n📂 Step 1: Loading data...")
    print(f"   Dataset: {args.dataset}")
    loader = CitationDataLoader(args.dataset)
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

    # 2. Initialize models
    print(f"\n🤖 Step 2: Initializing models...")
    models_to_run = []

    if args.model == "all":
        # Initialize all models
        if not CITEAGENT_AVAILABLE:
            print("⚠️  CiteAgent not available (missing API keys). Skipping CiteAgent.")
            models_to_run = [
                ("bm25", BM25Model(use_stemming=True, use_stopwords=True)),
                ("dense", DenseRetrievalModel(model_name=args.dense_model))
            ]
        else:
            try:
                cite_model = CiteAgentAPI(
                    llm_backend=args.llm_backend,
                    search_limit=args.citeagent_search_limit,
                    max_actions=args.citeagent_max_actions,
                    prompt_name=args.citeagent_prompt
                )
                models_to_run = [
                    ("citeagent", cite_model),
                    ("bm25", BM25Model(use_stemming=True, use_stopwords=True)),
                    ("dense", DenseRetrievalModel(model_name=args.dense_model))
                ]
            except ValueError as e:
                print(f"⚠️  Could not initialize CiteAgent: {e}")
                print("   Running BM25 and Dense only.")
                models_to_run = [
                    ("bm25", BM25Model(use_stemming=True, use_stopwords=True)),
                    ("dense", DenseRetrievalModel(model_name=args.dense_model))
                ]
    elif args.model == "citeagent":
        # Use real CiteAgent only
        if not CITEAGENT_AVAILABLE:
            print("❌ CiteAgent not available. Please check API keys and dependencies.")
            sys.exit(1)
        try:
            models_to_run = [("citeagent", CiteAgentAPI(
                llm_backend=args.llm_backend,
                search_limit=args.citeagent_search_limit,
                max_actions=args.citeagent_max_actions,
                prompt_name=args.citeagent_prompt
            ))]
        except ValueError as e:
            print(f"❌ Could not initialize CiteAgent: {e}")
            sys.exit(1)
    elif args.model == "bm25":
        models_to_run = [("bm25", BM25Model(use_stemming=True, use_stopwords=True))]
    elif args.model == "dense":
        models_to_run = [("dense", DenseRetrievalModel(model_name=args.dense_model))]

    # 3. Initialize evaluator
    evaluator = CitationEvaluator(
        fuzzy_match_threshold=0.85,
        track_latency=True,
        log_failures=True
    )

    # 4. Run evaluation for each model
    all_results = {}
    for model_name, model in models_to_run:
        print(f"\n{'=' * 70}")
        print(f"EVALUATING: {model_name.upper()}")
        print(f"{'=' * 70}")
        print(f"   Model config: {model.get_config()}")

        print(f"\n🔍 Running evaluation for {model_name}...")
        results = evaluator.evaluate_model(
            model=model,
            examples=examples,
            top_k=args.top_k,
            verbose=True
        )

        all_results[model_name] = results

        # Display results
        print("\n" + "=" * 70)
        print(f"RESULTS - {model_name.upper()}")
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

        # Save results
        print(f"\n💾 Saving results for {model_name}...")
        evaluator.save_results(
            results=results,
            output_dir="results/example",
            model_name=f"{model_name}_demo",
            include_predictions=True
        )

    # 5. Summary
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)

    if len(models_to_run) > 1:
        print("\n📊 COMPARISON SUMMARY:")
        print("=" * 70)
        for model_name, _ in models_to_run:
            results_data = all_results[model_name]
            metrics = results_data['metrics']
            metrics_dict = metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics

            print(f"\n{model_name.upper()}:")
            print(f"  Recall@20: {metrics_dict['recall@20']:.3f}")
            print(f"  MRR: {metrics_dict['mrr']:.3f}")

            # Get latency from results metadata if available
            latency = results_data.get('avg_latency_ms', 0)
            if latency > 0:
                print(f"  Avg Latency: {latency:.1f}ms")

    print("\n\nGenerated files in results/example/:")
    for model_name, _ in models_to_run:
        print(f"\n{model_name.upper()}:")
        print(f"  - {model_name}_demo_*_metrics.json")
        print(f"  - {model_name}_demo_*_failures.json")
        print(f"  - {model_name}_demo_*_predictions.json")

    print("\n\nNext steps:")
    print("  1. Run with different settings: uv run main.py --num-examples 100 --top-k 10")
    print("  2. Run specific model: uv run main.py --model bm25")
    print("  3. Use CiteAgent with GPT-4o: uv run main.py --model citeagent --llm-backend gpt-4o --num-examples 5")
    print("  4. Use CiteAgent with Claude: uv run main.py --model citeagent --llm-backend claude-3-5-sonnet-20241022 --num-examples 5")
    print("  5. Visualize results: uv run visualization.py")


if __name__ == "__main__":
    main()
