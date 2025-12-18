#!/usr/bin/env python3
"""
Comprehensive evaluation comparing baselines with and without LLM reranking.

Evaluates:
- BM25, E5, SPECTER (baseline)
- BM25 + Reranker, E5 + Reranker, SPECTER + Reranker

Generates comprehensive visualizations similar to the reference images.
"""

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFacePipeline
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus_loaders.scholarcopilot import build_citation_corpus, load_dataset
from src.evaluation.metrics import evaluate_retrieval
from src.prompts.llm_reranker import LLMRerankerPrompt
from src.resources.builders import build_inmemory_resources
from src.resources.cache import load_resources, save_resources

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
    """Evaluates baseline retrievers with optional LLM reranking."""

    def __init__(
        self,
        resources: Dict[str, Any],
        llm_model: str = "gemma3:4b",
        inference_engine: str = "ollama",
    ):
        self.resources = resources
        self.llm_model = llm_model
        self.inference_engine = inference_engine.lower()

        if self.inference_engine == "ollama":
            # Using Ollama for local inference (faster, less memory)
            logger.info(f"🔄 Using Ollama with model: {llm_model}")
            self.llm = ChatOllama(model=llm_model, temperature=0)
            logger.info(f"✅ Ollama ready!")
        else:
            # Using Hugging Face model loaded locally via transformers pipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            logger.info(f"🔄 Loading Hugging Face model: {llm_model}...")
            tok = AutoTokenizer.from_pretrained(llm_model)
            model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

            gen = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                max_new_tokens=1024,  # Increased for reasoning + JSON output
                do_sample=False,
            )

            self.llm = HuggingFacePipeline(pipeline=gen)
            logger.info(f"✅ Hugging Face model loaded!")

    def _llm_rerank(
        self, query: str, papers: List[Dict[str, Any]], top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Apply LLM reranking to papers."""
        if not papers:
            return []

        # Use top 20 for reranking (to save time)
        papers_to_rerank = papers[:top_k]

        # Build prompt
        prompt = LLMRerankerPrompt(query=query, candidate_papers=papers_to_rerank).get_prompt()

        try:
            # Call LLM
            response = self.llm.invoke(prompt)
            response_text = response.content

            logger.info(f"📝 LLM response length: {len(response_text)} chars")
            logger.info(f"📝 First 200 chars: {response_text[:200]}...")
            logger.info(f"📝 Last 200 chars: ...{response_text[-200:]}")

            # Extract JSON from response
            json_match = re.search(r"\[[\s\S]*\]", response_text)
            if json_match:
                json_str = json_match.group()
                logger.info(f"✓ Found JSON array ({len(json_str)} chars)")
                rankings = json.loads(json_str)
            else:
                logger.warning(f"⚠️  No JSON array pattern found, trying to parse entire response")
                rankings = json.loads(response_text)

            logger.info(f"✓ Successfully parsed {len(rankings)} rankings")

            # Build ranked results
            ranked_papers = []
            seen_indices = set()

            # Add papers from LLM ranking
            for item in rankings:
                idx = item["index"] - 1  # Convert to 0-based
                score = item["score"]

                if 0 <= idx < len(papers_to_rerank) and idx not in seen_indices:
                    paper = papers_to_rerank[idx].copy()
                    paper["reranked_score"] = score
                    ranked_papers.append(paper)
                    seen_indices.add(idx)

            # Add any papers that weren't ranked
            unranked_count = 0
            for i, paper in enumerate(papers_to_rerank):
                if i not in seen_indices:
                    paper = paper.copy()
                    paper["reranked_score"] = 0.0
                    ranked_papers.append(paper)
                    unranked_count += 1

            if unranked_count > 0:
                logger.warning(
                    f"⚠️  {unranked_count}/{len(papers_to_rerank)} papers not ranked by LLM (assigned score 0.0)"
                )

            return ranked_papers

        except Exception as e:
            logger.error(f"❌ LLM reranking failed: {e}")
            logger.error(f"❌ Response was: {response_text[:500] if 'response_text' in locals() else 'N/A'}...")
            logger.warning(f"⚠️  Returning original order")
            return papers

    def evaluate_bm25(
        self, query: str, relevant_ids: Set[str], k: int = 20, use_reranker: bool = False
    ) -> Dict[str, Any]:
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
                    "abstract": bm25_res.get("texts", [""])[idx],
                    "score": float(score),
                }
            )

        # Apply LLM reranking if requested
        if use_reranker:
            retrieved_papers = self._llm_rerank(query, retrieved_papers, top_k=k)

        # Evaluate
        metrics = evaluate_retrieval(retrieved_papers, relevant_ids, k_values=[5, 10, 20])
        return {"retrieved_papers": retrieved_papers, "metrics": metrics}

    def evaluate_e5(
        self, query: str, relevant_ids: Set[str], k: int = 20, use_reranker: bool = False
    ) -> Dict[str, Any]:
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

        # Compute similarities
        similarities = torch.matmul(q_emb, e5_res["corpus_embeddings"].T)[0]
        top_indices = torch.argsort(similarities, descending=True)[:k]

        # Format results
        retrieved_papers = []
        for idx in top_indices:
            idx = idx.item()
            retrieved_papers.append(
                {
                    "id": e5_res["ids"][idx],
                    "title": e5_res.get("titles", [""])[idx],
                    "abstract": e5_res.get("texts", [""])[idx],
                    "score": float(similarities[idx]),
                }
            )

        # Apply LLM reranking if requested
        if use_reranker:
            retrieved_papers = self._llm_rerank(query, retrieved_papers, top_k=k)

        # Evaluate
        metrics = evaluate_retrieval(retrieved_papers, relevant_ids, k_values=[5, 10, 20])
        return {"retrieved_papers": retrieved_papers, "metrics": metrics}

    def evaluate_specter(
        self, query: str, relevant_ids: Set[str], k: int = 20, use_reranker: bool = False
    ) -> Dict[str, Any]:
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

        # Encode query
        inputs = specter_res["tokenizer"](
            query, padding=True, truncation=True, return_tensors="pt", max_length=512
        )

        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)
            q_emb = output.last_hidden_state[:, 0, :]  # CLS token

        # Normalize
        q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1)

        # Ensure embeddings are on same device
        corpus_embeddings = specter_res["corpus_embeddings"].to(device)

        # Compute similarities
        similarities = torch.matmul(q_emb, corpus_embeddings.T)[0]
        top_indices = torch.argsort(similarities, descending=True)[:k]

        # Format results
        retrieved_papers = []
        for idx in top_indices:
            idx = idx.item()
            retrieved_papers.append(
                {
                    "id": specter_res["ids"][idx],
                    "title": specter_res.get("titles", [""])[idx],
                    "abstract": specter_res.get("texts", [""])[idx],
                    "score": float(similarities[idx]),
                }
            )

        # Apply LLM reranking if requested
        if use_reranker:
            retrieved_papers = self._llm_rerank(query, retrieved_papers, top_k=k)

        # Evaluate
        metrics = evaluate_retrieval(retrieved_papers, relevant_ids, k_values=[5, 10, 20])
        return {"retrieved_papers": retrieved_papers, "metrics": metrics}


def extract_citation_contexts(
    dataset: List[Dict[str, Any]], num_examples: int = 50
) -> List[Dict[str, Any]]:
    """Extract citation contexts from papers."""
    cite_pattern = re.compile(r"<\|cite_\d+\|>")
    citation_queries = []

    for paper in dataset[:num_examples]:
        paper_text = paper.get("paper", "")
        if not paper_text:
            continue

        bib_info = paper.get("bib_info", {})

        # Find all citation markers
        for match in cite_pattern.finditer(paper_text):
            cite_token = match.group(0)

            if cite_token not in bib_info:
                continue

            refs = bib_info[cite_token]
            if not refs:
                continue

            # Get ground truth IDs
            relevant_ids = set()
            for ref in refs:
                ref_id = ref.get("citation_key") or ref.get("paper_id")
                if ref_id:
                    relevant_ids.add(str(ref_id))

            if not relevant_ids:
                continue

            # Extract local context
            pos = match.start()
            words_before = paper_text[:pos].split()[-100:]
            words_after = paper_text[match.end() :].split()[:100]

            context = " ".join(words_before + words_after)
            context = re.sub(r"<\|cite_\d+\|>", "", context)
            context = " ".join(context.split())

            if len(context.split()) < 10:
                continue

            citation_queries.append({"query": context, "relevant_ids": relevant_ids})

    return citation_queries


def run_evaluation(
    dataset_path: str,
    num_examples: int = 50,
    k: int = 20,
    output_dir: str = "baseline_reranking_results",
    llm_model: str = "gemma3:4b",
    inference_engine: str = "ollama",
) -> Dict[str, Any]:
    """Run comprehensive evaluation."""
    logger.info("=" * 80)
    logger.info("BASELINE + LLM RERANKING EVALUATION")
    logger.info("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"\n📚 Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    logger.info(f"   ✓ Loaded {len(dataset)} papers")

    # Build resources
    resources = load_resources(dataset_path, enable_bm25=True, enable_e5=True, enable_specter=True)

    if not resources:
        logger.info(f"\n🔧 Building resources...")
        corpus = build_citation_corpus(dataset)
        logger.info(f"   ✓ Corpus: {len(corpus)} documents")

        resources = build_inmemory_resources(
            corpus, enable_bm25=True, enable_e5=True, enable_specter=True
        )
        logger.info(f"   ✓ Resources built")
        save_resources(resources, dataset_path)
    else:
        logger.info(f"   ✓ Loaded from cache")

    # Initialize evaluator
    logger.info(f"\n🤖 Using LLM: {llm_model} (engine: {inference_engine})")
    evaluator = BaselineEvaluator(resources, llm_model=llm_model, inference_engine=inference_engine)

    # Extract queries
    logger.info(f"\n🔍 Extracting citation contexts from {num_examples} papers...")
    citation_queries = extract_citation_contexts(dataset, num_examples)
    logger.info(f"   ✓ Extracted {len(citation_queries)} citation contexts")

    # Limit to num_examples citation contexts
    citation_queries = citation_queries[:num_examples]
    logger.info(f"   ✓ Evaluating on {len(citation_queries)} examples")

    # Run evaluation
    results = {
        "bm25": [],
        "bm25_reranked": [],
        "e5": [],
        "e5_reranked": [],
        "specter": [],
        "specter_reranked": [],
        "queries": [],
        "metadata": {
            "num_examples": len(citation_queries),
            "k": k,
            "llm_model": llm_model,
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_path,
        },
    }

    logger.info(f"\n🔄 Running evaluation...")
    for i, example in enumerate(tqdm(citation_queries, desc="Evaluating"), 1):
        query = example["query"]
        relevant_ids = example["relevant_ids"]

        # Store query info
        results["queries"].append(
            {
                "query": query[:200] + "..." if len(query) > 200 else query,
                "num_relevant": len(relevant_ids),
            }
        )

        # BM25
        bm25_result = evaluator.evaluate_bm25(query, relevant_ids, k=k, use_reranker=False)
        if bm25_result:
            results["bm25"].append(bm25_result["metrics"])

        # BM25 + Reranker
        bm25_reranked = evaluator.evaluate_bm25(query, relevant_ids, k=k, use_reranker=True)
        if bm25_reranked:
            results["bm25_reranked"].append(bm25_reranked["metrics"])

        # E5
        e5_result = evaluator.evaluate_e5(query, relevant_ids, k=k, use_reranker=False)
        if e5_result:
            results["e5"].append(e5_result["metrics"])

        # E5 + Reranker
        e5_reranked = evaluator.evaluate_e5(query, relevant_ids, k=k, use_reranker=True)
        if e5_reranked:
            results["e5_reranked"].append(e5_reranked["metrics"])

        # SPECTER
        specter_result = evaluator.evaluate_specter(query, relevant_ids, k=k, use_reranker=False)
        if specter_result:
            results["specter"].append(specter_result["metrics"])

        # SPECTER + Reranker
        specter_reranked = evaluator.evaluate_specter(query, relevant_ids, k=k, use_reranker=True)
        if specter_reranked:
            results["specter_reranked"].append(specter_reranked["metrics"])

    logger.info(f"\n✅ Evaluation complete!")

    # Save raw results
    with open(output_path / "full_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Saved raw results to {output_path / 'full_results.json'}")

    return results


def aggregate_results(results: Dict[str, Any]) -> pd.DataFrame:
    """Aggregate metrics into a summary DataFrame."""
    methods = ["bm25", "bm25_reranked", "e5", "e5_reranked", "specter", "specter_reranked"]

    aggregated = []
    for method in methods:
        metrics_list = results[method]
        if not metrics_list:
            continue

        row = {
            "Method": method.replace("_", " + ").replace("reranked", "LLM Reranker").upper(),
            "R@5": np.mean([m["R@5"] for m in metrics_list]),
            "R@10": np.mean([m["R@10"] for m in metrics_list]),
            "R@20": np.mean([m["R@20"] for m in metrics_list]),
            "MRR": np.mean([m["MRR"] for m in metrics_list]),
        }
        aggregated.append(row)

    df = pd.DataFrame(aggregated)
    return df


def create_visualizations(results: Dict[str, Any], output_dir: str = "baseline_reranking_results"):
    """Create comprehensive visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Aggregate results
    df = aggregate_results(results)

    # Save summary table
    df.to_csv(output_path / "summary_table.csv", index=False)
    logger.info(f"💾 Saved summary table to {output_path / 'summary_table.csv'}")

    # Print summary
    logger.info("\n📊 RESULTS SUMMARY")
    logger.info("=" * 80)
    print(df.to_string(index=False))
    logger.info("=" * 80)

    # Create figure with subplots (3 rows, 2 columns)
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Color scheme
    colors = {
        "BM25": "#FF6B6B",
        "BM25 + LLM RERANKER": "#FF4757",
        "E5": "#4ECDC4",
        "E5 + LLM RERANKER": "#00C9A7",
        "SPECTER": "#A29BFE",
        "SPECTER + LLM RERANKER": "#6C5CE7",
    }

    # 1. Metrics Comparison (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(df))
    width = 0.2

    ax1.bar(x - 1.5 * width, df["R@5"], width, label="Recall@5", alpha=0.8)
    ax1.bar(x - 0.5 * width, df["R@10"], width, label="Recall@10", alpha=0.8)
    ax1.bar(x + 0.5 * width, df["R@20"], width, label="Recall@20", alpha=0.8)
    ax1.bar(x + 1.5 * width, df["MRR"], width, label="MRR", alpha=0.8)

    ax1.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax1.set_title("Metrics Comparison", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Method"], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 2. Recall@K Curve
    ax2 = fig.add_subplot(gs[0, 1])
    methods = df["Method"].tolist()

    for i, method in enumerate(methods):
        k_values = [5, 10, 20]
        recall_values = [df.iloc[i]["R@5"], df.iloc[i]["R@10"], df.iloc[i]["R@20"]]
        ax2.plot(k_values, recall_values, marker="o", label=method, linewidth=2, markersize=8)

    ax2.set_xlabel("K", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Recall@K", fontsize=12, fontweight="bold")
    ax2.set_title("Recall@K Curve", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Performance Metrics (Horizontal Bar)
    ax3 = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(len(df))

    # Plot each metric as a horizontal bar
    for i, (idx, row) in enumerate(df.iterrows()):
        color = colors.get(row["Method"], "#95a5a6")
        ax3.barh(
            i * 3, row["R@5"], height=0.6, label="R@5" if i == 0 else "", color=color, alpha=0.6
        )
        ax3.barh(
            i * 3 + 1,
            row["R@10"],
            height=0.6,
            label="R@10" if i == 0 else "",
            color=color,
            alpha=0.8,
        )
        ax3.barh(
            i * 3 + 2, row["MRR"], height=0.6, label="MRR" if i == 0 else "", color=color, alpha=1.0
        )

        # Add text labels
        ax3.text(row["R@5"] + 0.01, i * 3, f'{row["R@5"]:.3f}', va="center", fontsize=8)
        ax3.text(row["R@10"] + 0.01, i * 3 + 1, f'{row["R@10"]:.3f}', va="center", fontsize=8)
        ax3.text(row["MRR"] + 0.01, i * 3 + 2, f'{row["MRR"]:.3f}', va="center", fontsize=8)

    ax3.set_yticks([i * 3 + 1 for i in range(len(df))])
    ax3.set_yticklabels(df["Method"], fontsize=9)
    ax3.set_xlabel("Score", fontsize=12, fontweight="bold")
    ax3.set_title("Performance Metrics", fontsize=14, fontweight="bold")
    ax3.grid(axis="x", alpha=0.3)

    # 4. Improvement Matrix (Heatmap)
    ax4 = fig.add_subplot(gs[1, 1])

    # Calculate improvements
    improvement_data = []
    baseline_methods = ["BM25", "E5", "SPECTER"]
    for baseline in baseline_methods:
        baseline_row = df[df["Method"] == baseline]
        reranked_row = df[df["Method"] == f"{baseline} + LLM RERANKER"]

        if not baseline_row.empty and not reranked_row.empty:
            improvements = []
            for metric in ["R@5", "R@10", "R@20", "MRR"]:
                base_val = baseline_row[metric].values[0]
                rerank_val = reranked_row[metric].values[0]
                if base_val > 0:
                    improvement = ((rerank_val - base_val) / base_val) * 100
                else:
                    improvement = 0
                improvements.append(improvement)
            improvement_data.append(improvements)

    improvement_df = pd.DataFrame(
        improvement_data, columns=["R@5", "R@10", "R@20", "MRR"], index=baseline_methods
    )

    sns.heatmap(
        improvement_df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        ax=ax4,
        cbar_kws={"label": "% Improvement"},
    )
    ax4.set_title("LLM Reranking Improvement (%)", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Metric", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Baseline", fontsize=12, fontweight="bold")

    # 5. Method Comparison (Grouped Bar)
    ax5 = fig.add_subplot(gs[2, 0])

    # Group by baseline
    baseline_methods = ["BM25", "E5", "SPECTER"]
    x_pos = np.arange(len(baseline_methods))
    width = 0.35

    baseline_mrr = [
        df[df["Method"] == method]["MRR"].values[0] if not df[df["Method"] == method].empty else 0
        for method in baseline_methods
    ]
    reranked_mrr = [
        (
            df[df["Method"] == f"{method} + LLM RERANKER"]["MRR"].values[0]
            if not df[df["Method"] == f"{method} + LLM RERANKER"].empty
            else 0
        )
        for method in baseline_methods
    ]

    bars1 = ax5.bar(x_pos - width / 2, baseline_mrr, width, label="Baseline", alpha=0.8)
    bars2 = ax5.bar(x_pos + width / 2, reranked_mrr, width, label="+ LLM Reranker", alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax5.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax5.set_ylabel("MRR", fontsize=12, fontweight="bold")
    ax5.set_title("MRR: Baseline vs LLM Reranked", fontsize=14, fontweight="bold")
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(baseline_methods)
    ax5.legend()
    ax5.grid(axis="y", alpha=0.3)

    # 6. Summary Statistics Table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    # Create summary text
    summary_text = "Summary Statistics\n\n"

    for method in baseline_methods:
        baseline_row = df[df["Method"] == method]
        reranked_row = df[df["Method"] == f"{method} + LLM RERANKER"]

        if not baseline_row.empty and not reranked_row.empty:
            base_mrr = baseline_row["MRR"].values[0]
            rerank_mrr = reranked_row["MRR"].values[0]
            improvement = ((rerank_mrr - base_mrr) / base_mrr * 100) if base_mrr > 0 else 0

            summary_text += f"{method}:\n"
            summary_text += f"  Baseline MRR: {base_mrr:.4f}\n"
            summary_text += f"  Reranked MRR: {rerank_mrr:.4f}\n"
            summary_text += f"  Improvement: {improvement:+.1f}%\n\n"

    # Add overall statistics
    all_baseline_mrr = [
        df[df["Method"] == method]["MRR"].values[0]
        for method in baseline_methods
        if not df[df["Method"] == method].empty
    ]
    all_reranked_mrr = [
        df[df["Method"] == f"{method} + LLM RERANKER"]["MRR"].values[0]
        for method in baseline_methods
        if not df[df["Method"] == f"{method} + LLM RERANKER"].empty
    ]

    avg_baseline = np.mean(all_baseline_mrr)
    avg_reranked = np.mean(all_reranked_mrr)
    avg_improvement = (
        ((avg_reranked - avg_baseline) / avg_baseline * 100) if avg_baseline > 0 else 0
    )

    summary_text += f"Average Across All Methods:\n"
    summary_text += f"  Baseline: {avg_baseline:.4f}\n"
    summary_text += f"  Reranked: {avg_reranked:.4f}\n"
    summary_text += f"  Improvement: {avg_improvement:+.1f}%"

    ax6.text(
        0.1,
        0.9,
        summary_text,
        transform=ax6.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # Save figure
    plt.suptitle(
        f'Baseline Retrieval with LLM Reranking - {results["metadata"]["num_examples"]} Examples',
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    output_file = output_path / "comprehensive_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"📈 Saved comprehensive visualization to {output_file}")
    plt.close()


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate baselines with LLM reranking")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/Users/ishaankalra/Dev/Retrieval/corpus_loaders/scholarcopilot/scholar_copilot_eval_data_1k.json",
        help="Path to dataset",
    )
    parser.add_argument(
        "--num-examples", type=int, default=50, help="Number of examples to evaluate"
    )
    parser.add_argument("--k", type=int, default=20, help="Number of results to retrieve")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_reranking_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.getenv("LOCAL_LLM", "gemma3:4b"),
        help="Model name for reranking (Ollama: 'gemma3:4b', HF: 'google/gemma-3-4b-it')",
    )
    parser.add_argument(
        "--inference-engine",
        type=str,
        default=os.getenv("INFERENCE_ENGINE", "ollama"),
        choices=["ollama", "huggingface"],
        help="Inference engine to use (default: ollama)",
    )

    args = parser.parse_args()

    # Run evaluation
    results = run_evaluation(
        dataset_path=args.dataset,
        num_examples=args.num_examples,
        k=args.k,
        output_dir=args.output_dir,
        llm_model=args.llm_model,
        inference_engine=args.inference_engine,
    )

    # Create visualizations
    create_visualizations(results, output_dir=args.output_dir)

    logger.info(f"\n✅ All done! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
