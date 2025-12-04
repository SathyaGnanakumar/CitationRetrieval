"""
Citation Retrieval Pipeline - Main Orchestration Script

This script coordinates the entire citation retrieval evaluation pipeline:
  1) Load dataset and build corpus
  2) BM25 sparse retrieval
  3) E5-Large-v2 dense retrieval
  4) Hybrid fusion (BM25 + Dense)
  5) Optional cross-encoder reranking
  6) Evaluation and visualization
"""

import pandas as pd
from tqdm import tqdm
from typing import Dict

# Import all components
from src.config.config import Config
from src.data_loader import CitationDataLoader
from src.retriever.bm25 import BM25Retriever
from src.retriever.e5 import E5Retriever
from src.retriever.hybrid import HybridRetriever
from src.reranker.cross_encoder import CrossEncoderReranker
from src.evaluator import CitationEvaluator
from src.visualization import ResultsVisualizer
from src.utils.text_processing import fuzzy_hit


def main():
    """Run the complete citation retrieval pipeline."""

    # =====================
    # 1) Initialize configuration
    # =====================
    config = Config()
    print(f"🚀 Starting Citation Retrieval Pipeline")
    print(f"📍 Device: {config.device}")

    # =====================
    # 2) Load data and build corpus
    # =====================
    loader = CitationDataLoader(config.data_file)
    loader.load_dataset()
    corpus_ids, corpus_titles, corpus_texts = loader.build_corpus()
    query_contexts, query_true_titles = loader.extract_query_contexts()

    # =====================
    # 3) Initialize retrievers
    # =====================
    # BM25
    bm25_retriever = BM25Retriever(language=config.bm25_language, stopwords=config.bm25_stopwords)
    bm25_retriever.index(corpus_texts)

    # E5 Dense
    e5_retriever = E5Retriever(
        model_name=config.dense_model_name, device=config.device, batch_size=config.batch_size
    )
    e5_retriever.load_model()
    e5_retriever.encode_corpus(corpus_texts)
    e5_retriever.encode_queries(query_contexts)

    # Hybrid
    hybrid_retriever = HybridRetriever(
        bm25_weight=config.fusion_bm25_weight, dense_weight=config.fusion_dense_weight
    )

    # =====================
    # 4) Retrieve and collect results
    # =====================
    print("🔗 Retrieving + fusing...")
    all_rows_bm25 = []
    all_rows_dense = []
    all_rows_hybrid = []

    for i in tqdm(range(len(query_contexts)), desc="Processing queries"):
        qtext = query_contexts[i]
        qemb = e5_retriever.get_query_embedding(i)
        gold = query_true_titles[i]

        # BM25 retrieval
        doc_ids_bm25, scores_bm25 = bm25_retriever.retrieve(qtext, k=config.top_k_bm25)

        # Dense retrieval
        doc_ids_dense, scores_dense = e5_retriever.retrieve(qemb, k=config.top_k_dense)

        # Hybrid fusion
        fused = hybrid_retriever.fuse(
            doc_ids_bm25,
            scores_bm25,
            doc_ids_dense,
            scores_dense,
            top_k=max(config.top_k_rerank, config.top_k_final),
        )

        # Collect BM25 results
        for rank, (doc_id, sc) in enumerate(zip(doc_ids_bm25, scores_bm25), 1):
            all_rows_bm25.append(
                {
                    "query_id": i,
                    "citation_context": qtext,
                    "true_title": gold,
                    "retrieved_title": corpus_titles[doc_id],
                    "retrieved_id": corpus_ids[doc_id],
                    "bm25_score": float(sc),
                    "rank": rank,
                    "hit": int(fuzzy_hit(gold, [corpus_titles[doc_id]])),
                }
            )

        # Collect Dense results
        for rank, (doc_id, sc) in enumerate(zip(doc_ids_dense, scores_dense), 1):
            all_rows_dense.append(
                {
                    "query_id": i,
                    "citation_context": qtext,
                    "true_title": gold,
                    "retrieved_title": corpus_titles[doc_id],
                    "retrieved_id": corpus_ids[doc_id],
                    "dense_score": float(sc),
                    "rank": rank,
                    "hit": int(fuzzy_hit(gold, [corpus_titles[doc_id]])),
                }
            )

        # Collect Hybrid results
        for rank, (doc_id, sc) in enumerate(fused[: config.top_k_final], 1):
            all_rows_hybrid.append(
                {
                    "query_id": i,
                    "citation_context": qtext,
                    "true_title": gold,
                    "retrieved_title": corpus_titles[doc_id],
                    "retrieved_id": corpus_ids[doc_id],
                    "hybrid_score": float(sc),
                    "rank": rank,
                    "hit": int(fuzzy_hit(gold, [corpus_titles[doc_id]])),
                }
            )

    # =====================
    # 5) Optional reranking
    # =====================
    rerank_rows = []
    if config.use_reranker:
        reranker = CrossEncoderReranker(model_name=config.reranker_model, device=config.device)
        reranker.load_model()

        df_h = pd.DataFrame(all_rows_hybrid)
        for qid, group in tqdm(df_h.groupby("query_id", sort=False), desc="Reranking"):
            cand = group.sort_values("rank").head(config.top_k_rerank)

            # Rerank candidates
            reranked = reranker.rerank_with_ids(
                query_contexts[qid],
                cand["retrieved_id"].tolist(),
                cand["retrieved_title"].tolist(),
                top_k=config.top_k_final,
            )

            # Build results
            for rank, (ret_id, score) in enumerate(reranked, 1):
                # Find the corresponding row
                match_row = cand[cand["retrieved_id"] == ret_id].iloc[0]
                rerank_rows.append(
                    {
                        "query_id": qid,
                        "citation_context": match_row["citation_context"],
                        "true_title": match_row["true_title"],
                        "retrieved_title": match_row["retrieved_title"],
                        "retrieved_id": ret_id,
                        "rerank_score": float(score),
                        "rank": rank,
                        "hit": int(
                            fuzzy_hit(match_row["true_title"], [match_row["retrieved_title"]])
                        ),
                    }
                )

    # =====================
    # 6) Evaluation
    # =====================
    evaluator = CitationEvaluator()
    visualizer = ResultsVisualizer(config.results_dir)

    # Convert to DataFrames
    df_bm25 = pd.DataFrame(all_rows_bm25)
    df_dense = pd.DataFrame(all_rows_dense)
    df_hybrid = pd.DataFrame(all_rows_hybrid)

    # Save individual results
    visualizer.save_results(df_bm25, "bm25_results.csv")
    visualizer.save_results(df_dense, "dense_results_e5.csv")
    visualizer.save_results(df_hybrid, "hybrid_results.csv")

    # Compute metrics
    all_metrics: Dict[str, Dict[str, float]] = {}

    metrics_bm25 = evaluator.compute_all_metrics(df_bm25)
    all_metrics["BM25"] = metrics_bm25
    visualizer.print_summary("BM25", metrics_bm25)

    metrics_dense = evaluator.compute_all_metrics(df_dense)
    all_metrics["Dense (E5)"] = metrics_dense
    visualizer.print_summary("Dense (E5)", metrics_dense)

    metrics_hybrid = evaluator.compute_all_metrics(df_hybrid)
    all_metrics["Hybrid"] = metrics_hybrid
    visualizer.print_summary("Hybrid (BM25+E5)", metrics_hybrid)

    if config.use_reranker:
        df_rerank = pd.DataFrame(rerank_rows)
        visualizer.save_results(df_rerank, "rerank_results.csv")
        metrics_rerank = evaluator.compute_all_metrics(df_rerank)
        all_metrics["Reranked"] = metrics_rerank
        visualizer.print_summary("Reranked", metrics_rerank)

    # =====================
    # 7) Generate visualizations
    # =====================
    visualizer.create_all_visualizations(all_metrics)

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
