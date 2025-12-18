#!/usr/bin/env python3
"""
Test the DSPy reformulator on multiple examples from the dataset.
Shows how queries are reformulated for real citation retrieval tasks.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Setup
load_dotenv()

from corpus_loaders.scholarcopilot import load_dataset
from evaluate import extract_query_from_paper, get_ground_truth_ids


def test_reformulator_on_dataset_examples(num_examples=5, mode="optimized", max_words=50):
    """
    Test reformulator on first N examples from the dataset.

    Args:
        num_examples: Number of examples to test
        mode: 'optimized' or 'standalone'
        max_words: Maximum words to extract from paper for query
    """
    print("=" * 80)
    print(f"DSPy REFORMULATOR TEST - Dataset Examples (mode={mode})")
    print("=" * 80)

    # Load dataset
    dataset_path = os.getenv("DATASET_DIR")
    if not dataset_path:
        print("❌ Error: DATASET_DIR not set in .env")
        return 1

    print(f"\n📚 Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"   ✓ Loaded {len(dataset)} papers")

    # Setup reformulator
    import dspy
    from src.agents.formulators.dspy_reformulator import DSPyQueryReformulator

    model = os.getenv("DSPY_MODEL", "gpt-4o-mini")
    print(f"\n🤖 Configuring DSPy with model: {model}")
    lm = dspy.LM(model=model, temperature=0.0, max_tokens=500)
    dspy.configure(lm=lm)

    # Load reformulator based on mode
    if mode == "optimized":
        from src.agents.self_evolve.version_tracker import VersionTracker

        tracker = VersionTracker("reformulator")
        reformulator = tracker.get_best()

        if reformulator is None:
            print(f"   ⚠️  No optimized version found, using base DSPy reformulator")
            reformulator = DSPyQueryReformulator()
        else:
            version = tracker.get_best_version_number()
            stats = tracker.get_statistics()
            print(
                f"   ✓ Loaded optimized reformulator v{version} (score: {stats.get('best_score', 'N/A')})"
            )
    else:
        reformulator = DSPyQueryReformulator()
        print(f"   ✓ Using base DSPy reformulator")

    # Test on examples
    num_examples = min(num_examples, len(dataset))
    print(f"\n🔄 Testing on first {num_examples} examples...")
    print("=" * 80)

    for i, paper in enumerate(dataset[:num_examples], 1):
        print(f"\n{'─' * 80}")
        print(f"📄 EXAMPLE {i}/{num_examples}")
        print(f"{'─' * 80}")

        # Get paper info
        paper_title = paper.get("title", "No title")[:60]
        paper_id = paper.get("paper_id", "unknown")

        print(f"\n📌 Paper: {paper_title}...")
        print(f"   ID: {paper_id}")

        # Extract query from paper
        paper_text = paper.get("paper", "")
        if not paper_text:
            paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"

        query = extract_query_from_paper(paper_text, max_words=max_words)

        print(f"\n🔍 Original Query ({len(query.split())} words):")
        print(f"   {query[:150]}{'...' if len(query) > 150 else ''}")

        # Get ground truth for context
        relevant_ids = get_ground_truth_ids(paper)
        print(f"\n📚 Ground Truth: {len(relevant_ids)} relevant citations")

        # Run reformulator
        try:
            result = reformulator(query=query)

            # Display reasoning
            if hasattr(result, "reasoning"):
                print(f"\n💭 Reasoning:")
                reasoning_preview = (
                    result.reasoning[:200] + "..."
                    if len(result.reasoning) > 200
                    else result.reasoning
                )
                print(f"   {reasoning_preview}")

            # Display reformulated queries
            query_list = getattr(result, "query_list", None) or [query]
            print(f"\n✨ Reformulated Queries ({len(query_list)}):")
            for j, q in enumerate(query_list, 1):
                # Show preview if long
                q_preview = q[:100] + "..." if len(q) > 100 else q
                print(f"   {j}. {q_preview}")

        except Exception as e:
            print(f"\n❌ Error reformulating query: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print(f"✅ Tested {num_examples} examples")
    print("=" * 80)


def compare_reformulators_on_examples(num_examples=3, max_words=50):
    """
    Compare rule-based vs DSPy reformulator side-by-side.

    Args:
        num_examples: Number of examples to compare
        max_words: Maximum words to extract from paper for query
    """
    print("=" * 80)
    print(f"REFORMULATOR COMPARISON - Rule-Based vs DSPy")
    print("=" * 80)

    # Load dataset
    dataset_path = os.getenv("DATASET_DIR")
    if not dataset_path:
        print("❌ Error: DATASET_DIR not set in .env")
        return 1

    print(f"\n📚 Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"   ✓ Loaded {len(dataset)} papers")

    # Setup DSPy reformulator
    import dspy
    from src.agents.formulators.dspy_reformulator import DSPyQueryReformulator
    from src.agents.formulators.query_reformulator import (
        extract_keywords,
        expand_keywords,
        academic_style_rewrite,
    )

    model = os.getenv("DSPY_MODEL", "gpt-4o-mini")
    print(f"\n🤖 Configuring DSPy with model: {model}")
    lm = dspy.LM(model=model, temperature=0.0, max_tokens=500)
    dspy.configure(lm=lm)

    dspy_reformulator = DSPyQueryReformulator()
    print(f"   ✓ Reformulators ready")

    # Test on examples
    num_examples = min(num_examples, len(dataset))
    print(f"\n🔄 Comparing on first {num_examples} examples...")
    print("=" * 80)

    for i, paper in enumerate(dataset[:num_examples], 1):
        print(f"\n{'━' * 80}")
        print(f"📄 EXAMPLE {i}/{num_examples}")
        print(f"{'━' * 80}")

        # Extract query
        paper_text = paper.get("paper", "")
        if not paper_text:
            paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"

        query = extract_query_from_paper(paper_text, max_words=max_words)

        print(f"\n🔍 Original Query:")
        print(f"   {query[:120]}...")

        # Rule-based reformulation
        print(f"\n🔧 RULE-BASED Reformulation:")
        keywords = extract_keywords(query)
        expansions = expand_keywords(keywords)

        rule_queries = [
            query,
            " ".join(keywords + expansions),
            academic_style_rewrite(query, keywords, expansions),
            f"{query} scientific paper architecture analysis",
        ]

        for j, q in enumerate(rule_queries, 1):
            q_preview = q[:80] + "..." if len(q) > 80 else q
            print(f"   {j}. {q_preview}")

        # DSPy reformulation
        print(f"\n🤖 DSPy Reformulation:")
        try:
            dspy_result = dspy_reformulator(query=query)
            dspy_queries = getattr(dspy_result, "query_list", [query])

            for j, q in enumerate(dspy_queries, 1):
                q_preview = q[:80] + "..." if len(q) > 80 else q
                print(f"   {j}. {q_preview}")

            # Show reasoning
            if hasattr(dspy_result, "reasoning"):
                print(f"\n   💭 DSPy Reasoning:")
                reasoning = (
                    dspy_result.reasoning[:150] + "..."
                    if len(dspy_result.reasoning) > 150
                    else dspy_result.reasoning
                )
                print(f"      {reasoning}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print(f"\n{'=' * 80}")
    print(f"✅ Compared {num_examples} examples")
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test DSPy reformulator on dataset examples")
    parser.add_argument(
        "--num-examples", type=int, default=5, help="Number of examples to test (default: 5)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="optimized",
        choices=["standalone", "optimized"],
        help="Reformulator mode: standalone (base DSPy) or optimized (from VersionTracker)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=50,
        help="Maximum words to extract from paper for query (default: 50)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare rule-based vs DSPy reformulators side-by-side",
    )

    args = parser.parse_args()

    try:
        if args.compare:
            compare_reformulators_on_examples(
                num_examples=args.num_examples, max_words=args.max_words
            )
        else:
            test_reformulator_on_dataset_examples(
                num_examples=args.num_examples, mode=args.mode, max_words=args.max_words
            )

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
