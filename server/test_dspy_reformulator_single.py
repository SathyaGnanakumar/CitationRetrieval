#!/usr/bin/env python3
"""
Test the DSPy optimized reformulator on a single query.
Shows the original query, reformulated queries, and reasoning.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Setup
load_dotenv()


def test_dspy_reformulator_standalone():
    """Test DSPy reformulator directly without the full workflow."""
    print("=" * 80)
    print("DSPy REFORMULATOR TEST (Standalone)")
    print("=" * 80)

    # Your test query
    test_query = "efficient transformer models for citation retrieval"

    print(f"\n📝 Original Query:")
    print(f"   {test_query}")

    # Import DSPy reformulator
    from src.agents.formulators.dspy_reformulator import DSPyQueryReformulator
    import dspy

    # Configure DSPy
    model = os.getenv("DSPY_MODEL", "gpt-4o-mini")
    print(f"\n🤖 Configuring DSPy with model: {model}")

    lm = dspy.LM(model=model, temperature=0.0, max_tokens=500)
    dspy.configure(lm=lm)

    # Create reformulator
    reformulator = DSPyQueryReformulator()
    print(f"   ✓ DSPy reformulator initialized")

    # Run reformulation
    print(f"\n🔄 Generating reformulated queries...")
    result = reformulator(query=test_query)

    # Display results
    print(f"\n📊 RESULTS:")
    print("=" * 80)

    print(f"\n1️⃣ Reasoning:")
    print(f"   {result.reasoning}")

    print(f"\n2️⃣ Reformulated Queries ({len(result.query_list)}):")
    for i, q in enumerate(result.query_list, 1):
        print(f"   {i}. {q}")

    print("\n" + "=" * 80)
    return result


def test_dspy_reformulator_optimized():
    """Test the optimized version from VersionTracker."""
    print("=" * 80)
    print("DSPy REFORMULATOR TEST (Optimized from VersionTracker)")
    print("=" * 80)

    # Your test query
    test_query = "efficient transformer models for citation retrieval"

    print(f"\n📝 Original Query:")
    print(f"   {test_query}")

    # Import version tracker
    from src.agents.self_evolve.version_tracker import VersionTracker
    import dspy

    # Configure DSPy
    model = os.getenv("DSPY_MODEL", "gpt-4o-mini")
    print(f"\n🤖 Configuring DSPy with model: {model}")

    lm = dspy.LM(model=model, temperature=0.0, max_tokens=500)
    dspy.configure(lm=lm)

    # Try to load optimized module
    tracker = VersionTracker("reformulator")
    module = tracker.get_best()

    if module is None:
        print("\n⚠️  No optimized reformulator found in VersionTracker")
        print("   Falling back to default DSPy reformulator")
        from src.agents.formulators.dspy_reformulator import DSPyQueryReformulator

        module = DSPyQueryReformulator()
    else:
        version = tracker.get_best_version_number()
        stats = tracker.get_statistics()
        print(f"\n✓ Loaded optimized reformulator v{version}")
        print(f"   Score: {stats.get('best_score', 'N/A')}")
        print(f"   Total versions: {stats.get('count', 0)}")

    # Run reformulation
    print(f"\n🔄 Generating reformulated queries...")
    result = module(query=test_query)

    # Display results
    print(f"\n📊 RESULTS:")
    print("=" * 80)

    if hasattr(result, "reasoning"):
        print(f"\n1️⃣ Reasoning:")
        print(f"   {result.reasoning}")

    query_list = getattr(result, "query_list", None) or [test_query]
    print(f"\n2️⃣ Reformulated Queries ({len(query_list)}):")
    for i, q in enumerate(query_list, 1):
        print(f"   {i}. {q}")

    print("\n" + "=" * 80)
    return result


def test_reformulator_in_workflow():
    """Test reformulator through the workflow state system."""
    print("=" * 80)
    print("DSPy REFORMULATOR TEST (Through Workflow)")
    print("=" * 80)

    from langchain_core.messages import HumanMessage
    from src.agents.formulators.query_reformulator import query_reformulator_optimized
    import dspy

    # Your test query
    test_query = "efficient transformer models for citation retrieval"

    print(f"\n📝 Original Query:")
    print(f"   {test_query}")

    # Configure DSPy
    model = os.getenv("DSPY_MODEL", "gpt-4o-mini")
    lm = dspy.LM(model=model, temperature=0.0, max_tokens=500)

    # Create state
    state = {
        "messages": [HumanMessage(content=test_query)],
        "resources": {"dspy_lm": lm},
        "config": {"use_optimized_reformulator": True},
    }

    # Run reformulator
    print(f"\n🔄 Running optimized reformulator through workflow...")
    result = query_reformulator_optimized(state)

    # Display results
    print(f"\n📊 RESULTS:")
    print("=" * 80)

    print(f"\n1️⃣ Original Query:")
    print(f"   {result.get('query', 'N/A')}")

    queries = result.get("queries", [])
    print(f"\n2️⃣ Reformulated Queries ({len(queries)}):")
    for i, q in enumerate(queries, 1):
        print(f"   {i}. {q}")

    if "reformulator_version" in result:
        print(f"\n3️⃣ Module Version:")
        print(f"   v{result['reformulator_version']}")

    if "dspy_reasoning" in result:
        print(f"\n4️⃣ Reasoning:")
        print(f"   {result['dspy_reasoning']}")

    print("\n" + "=" * 80)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test DSPy reformulator")
    parser.add_argument(
        "--mode",
        type=str,
        default="optimized",
        choices=["standalone", "optimized", "workflow"],
        help="Test mode: standalone (base DSPy), optimized (from VersionTracker), or workflow (through state)",
    )
    parser.add_argument("--query", type=str, help="Custom query to test (overrides default)")

    args = parser.parse_args()

    # Override test query if provided
    if args.query:
        # This is a bit hacky but works for the demo
        import builtins

        original_query = "efficient transformer models for citation retrieval"

        # Monkey patch the query in the functions
        for func in [
            test_dspy_reformulator_standalone,
            test_dspy_reformulator_optimized,
            test_reformulator_in_workflow,
        ]:
            func.__globals__["test_query"] = args.query

    try:
        if args.mode == "standalone":
            result = test_dspy_reformulator_standalone()
        elif args.mode == "optimized":
            result = test_dspy_reformulator_optimized()
        elif args.mode == "workflow":
            result = test_reformulator_in_workflow()

        print("\n✅ Test complete!")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
