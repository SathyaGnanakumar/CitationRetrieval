from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.corpus.scholarcopilot import build_citation_corpus, load_dataset
from src.corpus.query_extraction import extract_citation_queries
from src.resources.builders import build_inmemory_resources
from src.resources.cache import load_resources, save_resources, clear_cache
from src.utils.timing import setup_logging, timer
from src.workflow import RetrievalWorkflow

logger = logging.getLogger(__name__)


def build_dev_resources(
    dataset_path: str, *, bm25_only: bool, use_cache: bool = True, force_rebuild: bool = False
) -> Dict[str, Any]:
    """
    Build or load cached retrieval resources.

    Args:
        dataset_path: Path to the dataset
        bm25_only: Only build BM25 resources (skip dense embeddings)
        use_cache: Whether to use caching (default: True)
        force_rebuild: Force rebuilding even if cache exists (default: False)
    """
    enable_e5 = not bm25_only
    enable_specter = not bm25_only

    # Try to load from cache
    if use_cache and not force_rebuild:
        resources = load_resources(
            dataset_path,
            enable_bm25=True,
            enable_e5=enable_e5,
            enable_specter=enable_specter,
        )
        if resources:
            logger.info(f"📚 Corpus size: {len(resources['corpus'])} documents (from cache)")
            return resources

    # Build from scratch
    with timer("Loading dataset"):
        dataset = load_dataset(dataset_path)

    with timer("Building citation corpus"):
        docs = build_citation_corpus(dataset)
        logger.info(f"📚 Corpus size: {len(docs)} documents")

    # In dev/eval mode you can choose whether to pay the cost of dense embeddings.
    with timer("Building all retrieval resources"):
        resources = build_inmemory_resources(
            docs,
            enable_bm25=True,
            enable_e5=enable_e5,
            enable_specter=enable_specter,
        )

    # Save to cache
    if use_cache:
        with timer("Saving resources to cache"):
            save_resources(resources, dataset_path)

    return resources


def main():
    load_dotenv()
    setup_logging(level=logging.INFO)

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
        default=None,
        help="Query to run through the pipeline. If not provided and --from-dataset is used, extracts from dataset.",
    )
    parser.add_argument(
        "--from-dataset",
        action="store_true",
        help="Extract query from dataset (cite context). Requires --paper-idx.",
    )
    parser.add_argument(
        "--paper-idx",
        type=int,
        default=0,
        help="Index of paper in dataset to use for query extraction (default: 0).",
    )
    parser.add_argument(
        "--citation-idx",
        type=int,
        default=0,
        help="Which citation from the paper to use as target (default: 0 = first).",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=400,
        help="Number of characters before/after citation for context (default: 400).",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k results per retriever.")
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Disable dense embeddings (only use BM25). Default: use all retrievers.",
    )
    parser.add_argument(
        "--no-e5",
        action="store_true",
        help="Disable E5 dense embeddings. Default: enabled.",
    )
    parser.add_argument(
        "--no-specter",
        action="store_true",
        help="Disable SPECTER dense embeddings. Default: enabled.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging. Default: INFO level.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (rebuild from scratch). Default: use cache.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild even if cache exists. Default: use cache if available.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the cache and exit.",
    )
    parser.add_argument(
        "--llm-reranker",
        action="store_true",
        help="Use LLM-based reranker instead of cross-encoder (slower, requires Ollama/OpenAI). Default: cross-encoder.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation mode (don't show ground truth comparison). Default: show if available.",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.dataset:
        raise SystemExit("Missing dataset path. Provide --dataset or set DATASET_DIR.")

    # Handle cache clearing
    if args.clear_cache:
        clear_cache(args.dataset)
        return

    # Handle query extraction from dataset
    query = args.query
    ground_truth_id = None
    source_paper_title = None

    if args.from_dataset or not query:
        logger.info("📖 Extracting query from dataset...")
        dataset = load_dataset(args.dataset)

        if args.paper_idx >= len(dataset):
            raise SystemExit(
                f"Paper index {args.paper_idx} out of range (dataset has {len(dataset)} papers)"
            )

        paper = dataset[args.paper_idx]
        source_paper_title = paper.get("title", "")

        logger.info(f"📄 Source paper: {source_paper_title}")

        # Extract citation queries
        citation_queries = extract_citation_queries(
            paper,
            context_chars=args.context_chars,
            max_queries=args.citation_idx + 1,
        )

        if not citation_queries:
            logger.warning(
                "⚠️  No citations found in paper text. Using paper title/abstract as query."
            )
            query = f"{paper.get('title', '')} {paper.get('abstract', '')}"[:200]
        else:
            if args.citation_idx >= len(citation_queries):
                logger.warning(
                    f"⚠️  Citation index {args.citation_idx} out of range. Using last citation."
                )
                args.citation_idx = len(citation_queries) - 1

            query_info = citation_queries[args.citation_idx]
            query = query_info["query"]
            ground_truth_id = query_info["target_citation_id"]

            logger.info(f"🎯 Target citation: {query_info['target_citation_title'][:60]}...")
            logger.info(f"🔑 Citation ID: {ground_truth_id}")
            logger.info(f"📝 Context length: {len(query)} chars")

    if not query:
        raise SystemExit("No query provided. Use --query or --from-dataset.")

    logger.info(f"🚀 Starting retrieval pipeline")
    logger.info(f"📂 Dataset: {args.dataset}")
    logger.info(f"🔍 Query: {query[:100]}{'...' if len(query) > 100 else ''}")
    logger.info(f"🔢 Top-k per retriever: {args.k}")

    # Check which features are enabled
    enable_e5 = not (args.bm25_only or args.no_e5)
    enable_specter = not (args.bm25_only or args.no_specter)
    eval_mode = not args.no_eval

    if args.bm25_only:
        logger.info("⚠️  Dense embeddings disabled (BM25 only)")
    else:
        enabled_retrievers = ["BM25"]
        if enable_e5:
            enabled_retrievers.append("E5")
        if enable_specter:
            enabled_retrievers.append("SPECTER")
        logger.info(f"🔧 Enabled retrievers: {', '.join(enabled_retrievers)}")

    if args.no_cache:
        logger.info("⚠️  Caching disabled")
    elif args.force_rebuild:
        logger.info("⚠️  Forcing rebuild (ignoring cache)")
    else:
        logger.info("💾 Caching enabled")

    if eval_mode:
        logger.info("📊 Evaluation mode enabled (will show ground truth if available)")

    resources = build_dev_resources(
        args.dataset,
        bm25_only=args.bm25_only,
        use_cache=not args.no_cache,
        force_rebuild=args.force_rebuild,
    )

    workflow = RetrievalWorkflow(use_llm_reranker=args.llm_reranker)

    if args.llm_reranker:
        logger.info("🤖 Using LLM-based reranker")

    with timer("Running complete workflow"):
        final_state = workflow.run(
            {
                "messages": [HumanMessage(content=query)],
                "resources": resources,
                "config": {"k": args.k},
            }
        )

    print("\n" + "=" * 70)
    print("PIPELINE RESULTS")
    print("=" * 70)
    RetrievalWorkflow.pretty_print_messages(final_state)

    ranked = final_state.get("ranked_papers", [])
    if ranked:
        print("\n" + "=" * 70)
        print("TOP 5 RANKED PAPERS")
        print("=" * 70)
        for i, item in enumerate(ranked[:5], 1):
            title = item.get("title", "")
            score = item.get("rerank_score", item.get("score"))
            paper_id = item.get("id", "unknown")

            # Highlight if this is the ground truth paper
            if ground_truth_id and str(paper_id) == str(ground_truth_id):
                print(f"{i}. ⭐ {title} [GOLD STANDARD]")
            else:
                print(f"{i}. {title}")
            print(f"   ID: {paper_id} | Score: {score:.4f}")

    # Show gold standard position if we have it
    if ground_truth_id and ranked:
        found_at = None
        for i, paper in enumerate(ranked, 1):
            if str(paper.get("id", "")) == str(ground_truth_id):
                found_at = i
                break

        print("\n" + "=" * 70)
        print("GROUND TRUTH EVALUATION")
        print("=" * 70)
        if source_paper_title:
            print(f"Source paper: {source_paper_title}")
        print(f"Target citation ID: {ground_truth_id}")

        if found_at:
            print(f"\n✅ Found target citation at position #{found_at}")
            if found_at <= 5:
                print(f"   🎯 Excellent! In top-5")
            elif found_at <= 10:
                print(f"   👍 Good! In top-10")
            elif found_at <= 20:
                print(f"   👌 OK! In top-20")
        else:
            print(f"\n❌ Target citation NOT found in top-{len(ranked)} results")

    # If eval mode, try to show ground truth comparison
    if eval_mode and not args.from_dataset:
        try:
            from src.corpus.scholarcopilot import load_dataset

            dataset = load_dataset(args.dataset)

            # Try to find if query matches any paper
            query_text = args.query.lower()
            for paper in dataset:
                paper_text = paper.get("paper", "")[:200].lower()
                if query_text in paper_text or paper.get("title", "").lower() == query_text:
                    # Found the source paper - show ground truth
                    bib_info = paper.get("bib_info", {}) or {}
                    ground_truth_ids = set()
                    for entries in bib_info.values():
                        if isinstance(entries, list):
                            for citation in entries:
                                if isinstance(citation, dict):
                                    cit_id = (
                                        citation.get("citation_key")
                                        or citation.get("paper_id")
                                        or citation.get("id")
                                    )
                                    if cit_id:
                                        ground_truth_ids.add(str(cit_id))

                    if ground_truth_ids:
                        print("\n" + "=" * 70)
                        print("GROUND TRUTH COMPARISON")
                        print("=" * 70)
                        print(f"Source paper: {paper.get('title', '')}")
                        print(f"Citations in paper: {len(ground_truth_ids)}")

                        # Check which retrieved papers are in ground truth
                        retrieved_ids = [str(p.get("id", "")) for p in ranked[:10]]
                        hits = [rid for rid in retrieved_ids if rid in ground_truth_ids]

                        print(f"\n✓ Found {len(hits)} gold citations in top-10:")
                        for hit_id in hits:
                            # Find the paper in ranked list
                            for idx, p in enumerate(ranked[:10], 1):
                                if str(p.get("id", "")) == hit_id:
                                    print(f"  #{idx}: {p.get('title', '')[:70]}")
                                    break

                        if len(hits) < len(ground_truth_ids):
                            print(f"\n✗ Missed {len(ground_truth_ids) - len(hits)} citations")

                    break
        except Exception as e:
            logger.debug(f"Could not load ground truth: {e}")

    logger.info(f"🎉 Pipeline completed successfully! Retrieved {len(ranked)} papers.")


if __name__ == "__main__":
    main()
