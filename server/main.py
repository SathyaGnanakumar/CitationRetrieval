from __future__ import annotations

import argparse
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.corpus.scholarcopilot import build_citation_corpus, load_dataset
from src.resources.builders import build_inmemory_resources
from src.workflow import RetrievalWorkflow


def build_dev_resources(dataset_path: str, *, bm25_only: bool) -> Dict[str, Any]:
    dataset = load_dataset(dataset_path)
    docs = build_citation_corpus(dataset)

    # In dev/eval mode you can choose whether to pay the cost of dense embeddings.
    resources = build_inmemory_resources(
        docs,
        enable_bm25=True,
        enable_e5=not bm25_only,
        enable_specter=not bm25_only,
    )
    return resources


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Dev runner for the retrieval workflow")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.getenv("DATASET_DIR", ""),
        help="Path to ScholarCopilot JSON dataset (defaults to DATASET_DIR env var).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="transformer architecture for sequence modeling",
        help="Query to run through the pipeline.",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k results per retriever.")
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Only build BM25 resources (skips dense embedding builds).",
    )

    args = parser.parse_args()

    if not args.dataset:
        raise SystemExit("Missing dataset path. Provide --dataset or set DATASET_DIR.")

    print(f"[dev] Loading dataset: {args.dataset}")
    print("[dev] Building citation corpus from bib_info (not top-level papers)")
    resources = build_dev_resources(args.dataset, bm25_only=args.bm25_only)

    workflow = RetrievalWorkflow()
    final_state = workflow.run(
        {
            "messages": [HumanMessage(content=args.query)],
            "resources": resources,
            "config": {"k": args.k},
        }
    )

    print("\n=== Pipeline messages ===")
    RetrievalWorkflow.pretty_print_messages(final_state)

    ranked = final_state.get("ranked_papers", [])
    if ranked:
        print("\n=== Ranked papers (top 5) ===")
        for i, item in enumerate(ranked[:5], 1):
            title = item.get("title", "")
            score = item.get("rerank_score", item.get("score"))
            print(f"{i}. {title} (score={score})")


if __name__ == "__main__":
    main()
