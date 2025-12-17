import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.formulators.reranker import reranker, processed_list
from tests.preprocessing import (
    clean_text,
    extract_query_from_paper,
    load_dataset,
    run_retriever,
)


def test_reranker_with_dataset(retriever_name, reranker_model, dataset_path, example_idx=0):
    """Test reranker using real dataset and retriever"""

    print("\n" + "=" * 70)
    print(f"Testing Reranker with Real Dataset")
    print("=" * 70)

    # Load dataset
    data = load_dataset(dataset_path)

    print(f"\n📚 Loaded {len(data)} papers from dataset")

    # Get example paper
    if example_idx >= len(data):
        print(f"⚠️  Example index {example_idx} out of range. Using index 0.")
        example_idx = 0

    example = data[example_idx]
    print(f"📄 Using example {example_idx}: {example['title']}")

    # Extract query from paper text
    query = extract_query_from_paper(example["paper"])
    print(f"\n🔍 Query (extracted from paper):\n   {query}")

    # Run retriever
    print(f"\n🔎 Running {retriever_name.upper()} retriever...")
    candidate_papers = run_retriever(retriever_name, query, dataset_path)

    print(f"\n📋 Retrieved {len(candidate_papers)} candidate papers:")
    for i, paper in enumerate(candidate_papers[:5], 1):
        print(f"   {i}. {paper['title'][:80]}... (score: {paper['score']:.4f})")

    # Create state for reranker
    class SimpleState:
        def __init__(self, query, papers):
            self.query = query
            self.candidate_papers = papers

    state = SimpleState(query, candidate_papers)

    # Run reranker
    print(f"\n🎯 Running reranker with model: {reranker_model}")
    result = reranker(state, model_name=reranker_model)

    print(f"\n✨ Top 5 papers after reranking:")
    for i, (paper, score) in enumerate(result["ranked_papers"][:5], 1):
        print(f"   {i}. {paper['title'][:80]}...")
        print(f"      Retriever score: {paper['score']:.4f} | Reranker score: {score:.4f}")

    # Verify results
    assert "ranked_papers" in result
    assert len(result["ranked_papers"]) == len(candidate_papers)

    print("\n✅ Test completed successfully!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test reranker with different retrievers")
    parser.add_argument(
        "--retriever",
        type=str,
        default="bm25",
        choices=["bm25", "e5", "specter"],
        help="Which retriever to use (default: bm25)",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
        help="Reranker model name (default: BAAI/bge-reranker-v2-m3)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="../datasets/scholar_copilot_eval_data_1k.json",
        help="Path to dataset JSON file",
    )
    parser.add_argument(
        "--example-idx", type=int, default=0, help="Which example from dataset to use (default: 0)"
    )

    args = parser.parse_args()

    # Run test
    test_reranker_with_dataset(
        retriever_name=args.retriever,
        reranker_model=args.reranker_model,
        dataset_path=args.dataset,
        example_idx=args.example_idx,
    )


if __name__ == "__main__":
    main()
