#!/usr/bin/env python3
"""
Test script to run a single example through the pipeline with LLM reranker.
Shows detailed output including prompts, candidates, and selections.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus_loaders.scholarcopilot import load_dataset, build_citation_corpus
from src.workflow import RetrievalWorkflow
from src.resources.builders import build_inmemory_resources
from src.resources.cache import load_resources
from evaluate import get_ground_truth_ids, extract_query_from_paper

# Setup
load_dotenv()


def main():
    print("=" * 80)
    print("SINGLE EXAMPLE TEST WITH LLM RERANKER")
    print("=" * 80)

    # Load dataset
    dataset_path = os.getenv("DATASET_DIR")
    if not dataset_path:
        print("❌ Error: DATASET_DIR not set in .env")
        return 1

    print(f"\n📚 Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"✓ Loaded {len(dataset)} papers")

    # Get first example
    paper = dataset[0]
    print(f"\n📄 Paper: {paper.get('title', 'No title')}")
    print(f"   ID: {paper.get('paper_id', 'unknown')}")

    # Extract query
    paper_text = paper.get("paper", "")
    if not paper_text:
        paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"

    query = extract_query_from_paper(paper_text, max_words=50)
    print(f"\n🔍 Query (first 50 words):")
    print(f"   {query[:200]}...")

    # Get ground truth
    relevant_ids = get_ground_truth_ids(paper)
    print(f"\n✅ Ground truth: {len(relevant_ids)} relevant papers")
    print(f"   IDs: {list(relevant_ids)[:5]}")

    # Build resources
    print(f"\n🔧 Building resources...")
    resources = load_resources(dataset_path, enable_bm25=True, enable_e5=True, enable_specter=True)

    if not resources:
        print("   Cache miss, building from scratch...")
        corpus = build_citation_corpus(dataset)
        print(f"   ✓ Corpus: {len(corpus)} documents")

        resources = build_inmemory_resources(
            corpus, enable_bm25=True, enable_e5=True, enable_specter=True
        )
        print(f"   ✓ Resources built")
    else:
        print(f"   ✓ Loaded from cache")

    # Initialize workflow with LLM reranker (disable evolution to avoid conditional routing)
    print(f"\n🤖 Initializing workflow with LLM reranker...")
    workflow = RetrievalWorkflow(use_llm_reranker=True, enable_evolution=False)
    print(f"   ✓ Workflow ready (evolution disabled for simplicity)")

    # Run workflow
    print(f"\n🚀 Running workflow...")
    print("-" * 80)

    # Enable logging to see what nodes are executing
    import logging

    logging.getLogger("src.agents.formulators.llm_agent").setLevel(logging.INFO)
    logging.getLogger("src.agents.formulators.reranker").setLevel(logging.INFO)
    logging.getLogger("src.agents.formulators.dspy_picker").setLevel(logging.INFO)

    final_state = workflow.run(
        {
            "messages": [HumanMessage(content=query)],
            "resources": resources,
            "config": {
                "k": 20,
                "enable_dspy_picker": False,  # Disable DSPy picker to focus on LLM reranker
            },
        }
    )

    print("-" * 80)

    # Extract results
    print(f"\n📊 RESULTS:")
    print("=" * 80)

    # Show reformulated queries
    queries = final_state.get("queries", [])
    if queries:
        print(f"\n1️⃣ Reformulated Queries ({len(queries)}):")
        for i, q in enumerate(queries[:3], 1):
            print(f"   {i}. {q[:100]}...")

    # Show candidate papers before reranking
    candidate_papers = final_state.get("candidate_papers", [])
    print(f"\n2️⃣ Candidate Papers (before reranking): {len(candidate_papers)}")

    # Show ranked papers
    ranked_papers = final_state.get("ranked_papers", [])
    print(f"\n3️⃣ Final Ranked Papers (after reranking): {len(ranked_papers)}")

    if ranked_papers:
        print("\n   Top 10 Results:")
        print("   " + "-" * 76)

        hits = 0
        for i, paper_item in enumerate(ranked_papers[:10], 1):
            # Handle both tuple and dict formats
            if isinstance(paper_item, tuple):
                paper_data = paper_item[0]
                score = paper_item[1] if len(paper_item) > 1 else 0.0
            else:
                paper_data = paper_item
                score = paper_data.get("score", 0.0)

            paper_id = paper_data.get("id", "")
            title = paper_data.get("title", "No title")[:60]

            # Check if this is a ground truth hit
            is_hit = str(paper_id) in {str(gid) for gid in relevant_ids}
            if is_hit:
                hits += 1

            hit_marker = "✓ HIT" if is_hit else ""
            print(f"   {i:2d}. [{score:.4f}] {title}... {hit_marker}")

        print("   " + "-" * 76)
        print(f"   Hits in top-10: {hits}/{len(relevant_ids)}")

    # Show LLM reranker reasoning if available
    messages = final_state.get("messages", [])
    reranker_found = False
    for msg in messages:
        if hasattr(msg, "name") and msg.name and "rerank" in msg.name.lower():
            print(f"\n4️⃣ LLM Reranker Output:")
            print(f"   Name: {msg.name}")
            print(f"   Content: {msg.content[:500]}...")
            reranker_found = True

    if not reranker_found:
        print(f"\n4️⃣ LLM Reranker Output: Not found in messages")
        print(f"   Available message names: {[getattr(m, 'name', 'no-name') for m in messages]}")

    # Calculate metrics only if we have ranked papers
    if ranked_papers:
        from src.evaluation.metrics import evaluate_retrieval

        # Convert to format expected by evaluate_retrieval
        papers_for_eval = []
        for item in ranked_papers:
            if isinstance(item, tuple):
                papers_for_eval.append(item[0])
            else:
                papers_for_eval.append(item)

        metrics = evaluate_retrieval(papers_for_eval, relevant_ids, k_values=[5, 10, 20])

        print(f"\n5️⃣ Metrics:")
        print(f"   • Recall@5:  {metrics['R@5']:.4f} ({metrics['R@5']*100:.1f}%)")
        print(f"   • Recall@10: {metrics['R@10']:.4f} ({metrics['R@10']*100:.1f}%)")
        print(f"   • Recall@20: {metrics['R@20']:.4f} ({metrics['R@20']*100:.1f}%)")
        print(f"   • MRR:       {metrics['MRR']:.4f}")
        print(f"   • Hits:      {metrics['hits']}/{metrics['total_relevant']}")
    else:
        print(f"\n❌ No ranked papers returned!")
        print(f"   Debug info:")
        print(f"   - Final state keys: {list(final_state.keys())}")
        print(f"   - Candidate papers: {len(candidate_papers)}")
        print(f"   - Messages: {len(messages)}")

        # Show first candidate for debugging
        if candidate_papers:
            print(f"\n   First candidate paper:")
            first = candidate_papers[0]
            print(f"   - Title: {first.get('title', 'N/A')[:60]}...")
            print(f"   - ID: {first.get('id', 'N/A')}")

    print("\n" + "=" * 80)
    print("✅ Test complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
