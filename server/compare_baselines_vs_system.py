#!/usr/bin/env python3
"""
Comprehensive evaluation comparing baseline retrievers vs full system.

Compares:
- BM25 (lexical baseline)
- E5 (dense baseline)
- SPECTER (domain-specific baseline)
- Full System (aggregation + reranking + reformulation + picking)

Outputs:
- Detailed metrics (Recall@5, Recall@10, Recall@20, MRR)
- Comparison visualizations (bar charts, radar charts, heatmaps)
- Statistical significance tests
- Saved results (JSON, CSV, plots)
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus_loaders.scholarcopilot import build_citation_corpus, load_dataset
from evaluate import extract_query_from_paper, get_ground_truth_ids
from src.evaluation.metrics import evaluate_retrieval
from src.resources.builders import build_inmemory_resources
from src.resources.cache import load_resources, save_resources
from src.workflow import RetrievalWorkflow

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logs from httpx (used by Ollama)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Styling
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class BaselineEvaluator:
    """Evaluates individual baseline retrievers."""

    def __init__(self, resources: Dict[str, Any]):
        self.resources = resources

    def evaluate_bm25(self, query: str, relevant_ids: Set[str], k: int = 20) -> Dict[str, Any]:
        """Evaluate BM25 retriever."""
        import bm25s

        bm25_res = self.resources.get("bm25")
        if not bm25_res:
            return None

        # Tokenize query
        q_tokens = bm25s.tokenize(query, stopwords="en", stemmer=bm25_res["stemmer"])

        # Retrieve
        doc_indices, scores = bm25_res["index"].retrieve(q_tokens, k=k)

        # Format results
        retrieved_papers = []
        for idx, score in zip(doc_indices[0], scores[0]):
            retrieved_papers.append(
                {
                    "id": bm25_res["ids"][idx],
                    "title": bm25_res.get("titles", [""])[idx],
                    "score": float(score),
                }
            )

        # Evaluate
        metrics = evaluate_retrieval(retrieved_papers, relevant_ids, k_values=[5, 10, 20])
        return {"retrieved_papers": retrieved_papers, "metrics": metrics}

    def evaluate_e5(self, query: str, relevant_ids: Set[str], k: int = 20) -> Dict[str, Any]:
        """Evaluate E5 retriever."""
        import torch

        e5_res = self.resources.get("e5")
        if not e5_res:
            return None

        # Encode query
        with torch.no_grad():
            q_emb = e5_res["model"].encode(
                [query], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
            )

        # Retrieve
        scores = (q_emb @ e5_res["corpus_embeddings"].T)[0]
        top_k = torch.topk(scores, k=min(k, len(scores)))

        # Format results
        retrieved_papers = []
        for idx, score in zip(top_k.indices, top_k.values):
            retrieved_papers.append(
                {
                    "id": e5_res["ids"][idx.item()],
                    "title": e5_res.get("titles", [""])[idx.item()],
                    "score": score.item(),
                }
            )

        # Evaluate
        metrics = evaluate_retrieval(retrieved_papers, relevant_ids, k_values=[5, 10, 20])
        return {"retrieved_papers": retrieved_papers, "metrics": metrics}

    def evaluate_specter(self, query: str, relevant_ids: Set[str], k: int = 20) -> Dict[str, Any]:
        """Evaluate SPECTER retriever."""
        import torch

        specter_res = self.resources.get("specter")
        if not specter_res:
            return None

        # Get device from resources or detect from embeddings
        device = specter_res.get("device")
        if device is None:
            device = str(specter_res["corpus_embeddings"].device)

        # Ensure model is on correct device
        model = specter_res["model"]
        if str(next(model.parameters()).device) != device:
            model = model.to(device)

        tokenizer = specter_res["tokenizer"]

        # Encode query
        with torch.no_grad():
            inputs = tokenizer(
                [query], padding=True, truncation=True, max_length=256, return_tensors="pt"
            )
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            q_emb = model(**inputs).last_hidden_state.mean(dim=1)
            q_emb = torch.nn.functional.normalize(q_emb, dim=1)

        # Retrieve - ensure embeddings are on same device
        corpus_embs = specter_res["corpus_embeddings"].to(device)
        corpus_embs = torch.nn.functional.normalize(corpus_embs, dim=1)
        scores = (q_emb @ corpus_embs.T)[0]
        top_k = torch.topk(scores, k=min(k, len(scores)))

        # Format results
        retrieved_papers = []
        for idx, score in zip(top_k.indices, top_k.values):
            retrieved_papers.append(
                {
                    "id": specter_res["ids"][idx.item()],
                    "title": specter_res.get("titles", [""])[idx.item()],
                    "score": score.item(),
                }
            )

        # Evaluate
        metrics = evaluate_retrieval(retrieved_papers, relevant_ids, k_values=[5, 10, 20])
        return {"retrieved_papers": retrieved_papers, "metrics": metrics}


class SystemEvaluator:
    """Evaluates the full system."""

    def __init__(
        self,
        workflow: RetrievalWorkflow,
        resources: Dict[str, Any],
        use_llm_reranker: bool = False,
        use_dspy: bool = False,
    ):
        self.workflow = workflow
        self.resources = resources
        self.use_llm_reranker = use_llm_reranker
        self.use_dspy = use_dspy

    def evaluate(self, query: str, relevant_ids: Set[str], k: int = 20) -> Dict[str, Any]:
        """Evaluate full system."""
        try:
            final_state = self.workflow.run(
                {
                    "messages": [HumanMessage(content=query)],
                    "resources": self.resources,
                    "config": {
                        "k": k,
                        "enable_dspy_picker": self.use_dspy,
                        "use_optimized_reformulator": self.use_dspy,
                        "use_dspy_reformulator": self.use_dspy,
                    },
                }
            )

            ranked_papers = final_state.get("ranked_papers", [])

            if not ranked_papers:
                return None

            # Handle tuple format (paper, score)
            papers_for_eval = []
            for item in ranked_papers:
                if isinstance(item, tuple):
                    papers_for_eval.append(item[0])
                else:
                    papers_for_eval.append(item)

            metrics = evaluate_retrieval(papers_for_eval, relevant_ids, k_values=[5, 10, 20])
            return {"retrieved_papers": papers_for_eval, "metrics": metrics}

        except Exception as e:
            logger.error(f"Error evaluating system: {e}")
            return None


def run_comparison(
    dataset_path: str,
    num_examples: int = 100,
    k: int = 20,
    output_dir: str = "comparison_results",
    use_cache: bool = True,
    use_llm_reranker: bool = False,
    use_dspy: bool = False,
) -> Dict[str, Any]:
    """
    Run comprehensive comparison between baselines and full system.

    Args:
        dataset_path: Path to dataset
        num_examples: Number of examples to evaluate
        k: Top-k results
        output_dir: Output directory for results
        use_cache: Whether to use cached resources
        use_llm_reranker: Whether to use LLM-based reranker (vs cross-encoder)
        use_dspy: Whether to use DSPy modules (reformulator and picker)

    Returns:
        Comparison results
    """
    logger.info("=" * 80)
    logger.info("BASELINE vs SYSTEM COMPARISON")
    logger.info("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"\n📚 Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    logger.info(f"   ✓ Loaded {len(dataset)} papers")

    # Build resources
    if use_cache:
        resources = load_resources(
            dataset_path, enable_bm25=True, enable_e5=True, enable_specter=True
        )

    if not use_cache or not resources:
        logger.info(f"\n🔧 Building resources...")
        corpus = build_citation_corpus(dataset)
        logger.info(f"   ✓ Corpus: {len(corpus)} documents")

        resources = build_inmemory_resources(
            corpus,
            enable_bm25=True,
            enable_e5=True,
            enable_specter=True,
            enable_llm_reranker=use_llm_reranker,  # Load LLM model once if using LLM reranker
        )
        logger.info(f"   ✓ Resources built")

        if use_cache:
            save_resources(resources, dataset_path)
    else:
        logger.info(f"   ✓ Loaded from cache")

        # If using LLM reranker and model not in cache, load it now
        if use_llm_reranker and "llm_reranker" not in resources:
            logger.info(f"\n🔧 LLM reranker enabled but not in cache - loading now...")
            from src.resources.builders import build_llm_reranker_resources
            resources["llm_reranker"] = build_llm_reranker_resources()

    # Initialize evaluators
    baseline_evaluator = BaselineEvaluator(resources)
    workflow = RetrievalWorkflow(use_llm_reranker=use_llm_reranker, enable_evolution=False)
    system_evaluator = SystemEvaluator(
        workflow, resources, use_llm_reranker=use_llm_reranker, use_dspy=use_dspy
    )

    logger.info(f"\n⚙️  Configuration:")
    if use_llm_reranker:
        logger.info(f"   🤖 Using LLM-based reranker")
    else:
        logger.info(f"   🔄 Using cross-encoder reranker")

    if use_dspy:
        logger.info(f"   🧠 Using DSPy modules (reformulator + picker)")
    else:
        logger.info(f"   📋 Using rule-based modules")

    # Evaluate on examples
    num_examples = min(num_examples, len(dataset))
    logger.info(f"\n🔍 Evaluating on {num_examples} examples...")
    logger.info("=" * 80)

    results = {
        "bm25": [],
        "e5": [],
        "specter": [],
        "full_system": [],
        "metadata": {
            "num_examples": num_examples,
            "k": k,
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_path,
            "use_llm_reranker": use_llm_reranker,
            "reranker_type": "LLM-based" if use_llm_reranker else "cross-encoder",
            "use_dspy": use_dspy,
            "modules": "DSPy (reformulator + picker)" if use_dspy else "Rule-based",
        },
    }

    examples_evaluated = 0

    # Extract citation contexts (proper evaluation setup)
    import re

    cite_pattern = re.compile(r"<\|cite_\d+\|>")

    citation_queries = []
    for paper in dataset[:num_examples]:
        paper_text = paper.get("paper", "")
        if not paper_text:
            continue

        bib_info = paper.get("bib_info", {})

        # Find all citation markers in text
        for match in cite_pattern.finditer(paper_text):
            cite_token = match.group(0)

            if cite_token not in bib_info:
                continue

            # Get ground truth for this specific citation
            refs = bib_info[cite_token]
            if not refs:
                continue

            relevant_ids = set()
            for ref in refs:
                # Use citation_key (or paper_id as fallback) - this matches the corpus builder!
                # See: corpus_loaders/scholarcopilot/loader.py line 84-86
                ref_id = ref.get("citation_key") or ref.get("paper_id")
                if ref_id:
                    relevant_ids.add(str(ref_id))

            if not relevant_ids:
                continue

            # Extract local context around citation (±200 words is ~400 chars)
            pos = match.start()
            # Find word boundaries
            words_before = paper_text[:pos].split()[-100:]  # 100 words before
            words_after = paper_text[match.end() :].split()[:100]  # 100 words after

            context = " ".join(words_before + words_after)
            # Remove citation markers from context
            context = re.sub(r"<\|cite_\d+\|>", "", context)
            context = " ".join(context.split())  # Clean whitespace

            if len(context.split()) < 10:  # Skip if context too short
                continue

            citation_queries.append(
                {
                    "query": context,
                    "relevant_ids": relevant_ids,
                    "paper_id": paper.get("paper_id", "unknown"),
                }
            )

    logger.info(f"   Extracted {len(citation_queries)} citation contexts")

    # Limit to num_examples
    citation_queries = citation_queries[:num_examples]

    for i, example in enumerate(tqdm(citation_queries, desc="Evaluating"), 1):
        query = example["query"]
        relevant_ids = example["relevant_ids"]

        examples_evaluated += 1

        # Evaluate each baseline (using the citation context query)
        try:
            bm25_result = baseline_evaluator.evaluate_bm25(query, relevant_ids, k=k)
            if bm25_result:
                results["bm25"].append(bm25_result["metrics"])
        except Exception as e:
            logger.error(f"   ❌ BM25 error: {e}")

        try:
            e5_result = baseline_evaluator.evaluate_e5(query, relevant_ids, k=k)
            if e5_result:
                results["e5"].append(e5_result["metrics"])
        except Exception as e:
            logger.error(f"   ❌ E5 error: {e}")

        try:
            specter_result = baseline_evaluator.evaluate_specter(query, relevant_ids, k=k)
            if specter_result:
                results["specter"].append(specter_result["metrics"])
        except Exception as e:
            logger.error(f"   ❌ SPECTER error: {e}")

        # Evaluate full system
        try:
            system_result = system_evaluator.evaluate(query, relevant_ids, k=k)
            if system_result:
                results["full_system"].append(system_result["metrics"])
            else:
                logger.warning(f"   ⚠️  Full system returned None for query {i}")
        except Exception as e:
            logger.error(f"   ❌ Full system error for query {i}: {e}")
            import traceback

            traceback.print_exc()

    results["metadata"]["examples_evaluated"] = examples_evaluated

    logger.info(f"\n✅ Evaluated {examples_evaluated} examples")
    logger.info("=" * 80)

    return results


def aggregate_results(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across all examples."""
    aggregated = {}

    for method in ["bm25", "e5", "specter", "full_system"]:
        metrics_list = results[method]

        if not metrics_list:
            logger.warning(f"⚠️  No results for {method} - skipping")
            continue

        aggregated[method] = {
            "R@5": np.mean([m["R@5"] for m in metrics_list]),
            "R@10": np.mean([m["R@10"] for m in metrics_list]),
            "R@20": np.mean([m["R@20"] for m in metrics_list]),
            "MRR": np.mean([m["MRR"] for m in metrics_list]),
            "R@5_std": np.std([m["R@5"] for m in metrics_list]),
            "R@10_std": np.std([m["R@10"] for m in metrics_list]),
            "R@20_std": np.std([m["R@20"] for m in metrics_list]),
            "MRR_std": np.std([m["MRR"] for m in metrics_list]),
        }

    return aggregated


def save_results(results: Dict[str, Any], aggregated: Dict[str, Dict[str, float]], output_dir: str):
    """Save results to files."""
    output_path = Path(output_dir)

    # Save full results as JSON
    with open(output_path / "full_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n💾 Saved full results to {output_path / 'full_results.json'}")

    # Save aggregated results as JSON
    with open(output_path / "aggregated_results.json", "w") as f:
        json.dump(aggregated, f, indent=2)
    logger.info(f"💾 Saved aggregated results to {output_path / 'aggregated_results.json'}")

    # Save as CSV
    df = pd.DataFrame(aggregated).T
    df.to_csv(output_path / "comparison_table.csv")
    logger.info(f"💾 Saved comparison table to {output_path / 'comparison_table.csv'}")

    # Print summary table
    logger.info(f"\n📊 RESULTS SUMMARY")
    logger.info("=" * 80)
    print(df[["R@5", "R@10", "R@20", "MRR"]].round(4).to_string())
    logger.info("=" * 80)


def create_visualizations(aggregated: Dict[str, Dict[str, float]], output_dir: str):
    """Create comprehensive visualizations."""
    output_path = Path(output_dir)

    # Method names for display
    method_names = {
        "bm25": "BM25",
        "e5": "E5",
        "specter": "SPECTER",
        "full_system": "Full System",
    }

    methods = list(aggregated.keys())
    display_names = [method_names.get(m, m) for m in methods]

    # 1. Bar Chart - Recall Metrics
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(methods))
    width = 0.2

    r5 = [aggregated[m]["R@5"] for m in methods]
    r10 = [aggregated[m]["R@10"] for m in methods]
    r20 = [aggregated[m]["R@20"] for m in methods]
    mrr = [aggregated[m]["MRR"] for m in methods]

    ax.bar(x - 1.5 * width, r5, width, label="R@5", alpha=0.8)
    ax.bar(x - 0.5 * width, r10, width, label="R@10", alpha=0.8)
    ax.bar(x + 0.5 * width, r20, width, label="R@20", alpha=0.8)
    ax.bar(x + 1.5 * width, mrr, width, label="MRR", alpha=0.8)

    ax.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Retrieval Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "bar_chart_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"📈 Saved bar chart to {output_path / 'bar_chart_comparison.png'}")

    # 2. Radar Chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    categories = ["R@5", "R@10", "R@20", "MRR"]
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for method, display_name in zip(methods, display_names):
        values = [aggregated[method][cat] for cat in categories]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=display_name)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 1)
    ax.set_title("Performance Radar Chart", size=16, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path / "radar_chart_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"📈 Saved radar chart to {output_path / 'radar_chart_comparison.png'}")

    # 3. Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    data = [[aggregated[m][cat] for cat in categories] for m in methods]

    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(display_names)

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(categories)):
            text = ax.text(
                j, i, f"{data[i][j]:.3f}", ha="center", va="center", color="black", fontsize=11
            )

    ax.set_title("Performance Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar(im, ax=ax, label="Score")

    plt.tight_layout()
    plt.savefig(output_path / "heatmap_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"📈 Saved heatmap to {output_path / 'heatmap_comparison.png'}")

    # 4. Improvement over Best Baseline
    # Only use methods that actually have results
    available_baselines = [m for m in ["bm25", "e5", "specter"] if m in aggregated]

    if not available_baselines:
        logger.warning("⚠️  No baseline results available for improvement chart")
        return

    baseline_scores = {
        "R@5": max(aggregated[m]["R@5"] for m in available_baselines),
        "R@10": max(aggregated[m]["R@10"] for m in available_baselines),
        "R@20": max(aggregated[m]["R@20"] for m in available_baselines),
        "MRR": max(aggregated[m]["MRR"] for m in available_baselines),
    }

    system_scores = aggregated.get("full_system", {})

    if system_scores:
        improvements = {
            cat: ((system_scores[cat] - baseline_scores[cat]) / baseline_scores[cat] * 100)
            for cat in categories
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ["green" if v > 0 else "red" for v in improvements.values()]
        bars = ax.bar(categories, improvements.values(), color=colors, alpha=0.7)

        ax.set_ylabel("Improvement (%)", fontsize=12, fontweight="bold")
        ax.set_title("Full System Improvement over Best Baseline", fontsize=14, fontweight="bold")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:+.1f}%",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(output_path / "improvement_chart.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"📈 Saved improvement chart to {output_path / 'improvement_chart.png'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare baseline retrievers vs full system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=os.getenv("DATASET_DIR", ""),
        help="Path to dataset (default: from DATASET_DIR env var)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to evaluate (default: 100)",
    )
    parser.add_argument("--k", type=int, default=20, help="Top-k results to retrieve (default: 20)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Output directory for results (default: comparison_results)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable resource caching")
    parser.add_argument(
        "--llm-reranker",
        action="store_true",
        help="Use LLM-based reranker instead of cross-encoder (slower but may be better)",
    )
    parser.add_argument(
        "--use-dspy",
        action="store_true",
        help="Use DSPy modules for reformulator and picker (requires API calls)",
    )

    args = parser.parse_args()

    if not args.dataset:
        raise SystemExit("Error: Dataset path required. Use --dataset or set DATASET_DIR env var.")

    # Run comparison
    results = run_comparison(
        dataset_path=args.dataset,
        num_examples=args.num_examples,
        k=args.k,
        output_dir=args.output_dir,
        use_cache=not args.no_cache,
        use_llm_reranker=args.llm_reranker,
        use_dspy=args.use_dspy,
    )

    # Aggregate and analyze
    aggregated = aggregate_results(results)

    # Save results
    save_results(results, aggregated, args.output_dir)

    # Create visualizations
    create_visualizations(aggregated, args.output_dir)

    logger.info(f"\n✅ Comparison complete! Results saved to {args.output_dir}/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
