"""
Main evaluation harness for citation retrieval models.

Supports evaluation of multiple models with comprehensive metrics and error analysis.
"""

import json
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import pandas as pd

from .metrics import MetricsCalculator, RetrievalMetrics
from .data_loader import CitationExample


class CitationEvaluator:
    """
    Unified evaluation harness for citation retrieval models.

    Supports:
    - Multiple model comparison
    - Comprehensive metrics (Recall@k, MRR, Precision@k)
    - Error analysis and failure logging
    - Latency tracking
    - Cost estimation
    """

    def __init__(
        self,
        fuzzy_match_threshold: float = 0.85,
        track_latency: bool = True,
        log_failures: bool = True
    ):
        """
        Args:
            fuzzy_match_threshold: Threshold for fuzzy title matching
            track_latency: Whether to track per-query latency
            log_failures: Whether to log failure cases
        """
        self.metrics_calculator = MetricsCalculator(fuzzy_match_threshold)
        self.track_latency = track_latency
        self.log_failures = log_failures
        self.failure_log = []

    def evaluate_model(
        self,
        model: Any,
        examples: List[CitationExample],
        top_k: int = 20,
        batch_size: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate a model on a set of examples.

        Args:
            model: Model object with `retrieve(query, corpus, k)` method
            examples: List of CitationExample objects
            top_k: Number of candidates to retrieve
            batch_size: Batch size for processing (None = all at once)
            verbose: Print progress

        Returns:
            Dictionary containing:
                - metrics: RetrievalMetrics object
                - predictions: List of prediction dicts
                - latencies: List of per-query latencies
                - failure_log: List of failure cases
        """
        if verbose:
            print(f"\n🔍 Evaluating model on {len(examples)} examples...")
            print(f"   Retrieving top-{top_k} candidates per query")

        predictions = []
        latencies = []
        self.failure_log = []

        # Group examples by paper_id to optimize retrieval
        examples_by_paper = {}
        for ex in examples:
            if ex.paper_id not in examples_by_paper:
                examples_by_paper[ex.paper_id] = []
            examples_by_paper[ex.paper_id].append(ex)

        # Process each paper group
        total_papers = len(examples_by_paper)
        for i, (paper_id, paper_examples) in enumerate(examples_by_paper.items()):
            if verbose and (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{total_papers} papers...")

            # All examples in this group share the same corpus (bibliography)
            corpus = paper_examples[0].corpus_entries
            queries = [ex.citation_context for ex in paper_examples]

            # Batch retrieve
            start_time = time.time()
            try:
                # This will encode corpus ONCE and queries in batch
                if hasattr(model, 'retrieve_batch'):
                    batch_retrieved = model.retrieve_batch(queries, corpus, k=top_k)
                else:
                    # Fallback for models without batch support
                    batch_retrieved = [model.retrieve(q, corpus, k=top_k) for q in queries]
            except Exception as e:
                print(f"⚠️  Error on paper {paper_id}: {e}")
                batch_retrieved = [[] for _ in queries]

            # Calculate average latency per query in this batch
            total_latency_ms = (time.time() - start_time) * 1000
            avg_latency_ms = total_latency_ms / len(queries) if queries else 0

            # Process results
            for j, ex in enumerate(paper_examples):
                retrieved = batch_retrieved[j]
                retrieved_titles = [item['title'] for item in retrieved]

                # Store prediction
                prediction = {
                    'query_id': ex.query_id,
                    'true_titles': [ex.true_title],
                    'retrieved_titles': retrieved_titles,
                    'paper_id': ex.paper_id,
                    'context': ex.citation_context[:200] + "..."
                }
                predictions.append(prediction)

                if self.track_latency:
                    latencies.append(avg_latency_ms)

                # Log failure
                per_query_metrics = self.metrics_calculator.calculate_per_query_metrics(
                    [ex.true_title],
                    retrieved_titles,
                    k=top_k
                )

                if not per_query_metrics['hit'] and self.log_failures:
                    self._log_failure(ex, retrieved_titles, "not_in_top_k")

        # Calculate overall metrics
        metrics = self.metrics_calculator.calculate_metrics(predictions)

        # Prepare results
        results = {
            'metrics': metrics,
            'predictions': predictions,
            'latencies': latencies if self.track_latency else None,
            'failure_log': self.failure_log if self.log_failures else None,
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'total_time_sec': sum(latencies) / 1000 if latencies else 0
        }

        if verbose:
            print(f"\n✅ Evaluation complete!")
            print(f"   Total time: {results['total_time_sec']:.2f}s")
            print(f"   Avg latency: {results['avg_latency_ms']:.2f}ms per query")
            print(metrics)

        return results

    def _log_failure(
        self,
        example: CitationExample,
        retrieved_titles: List[str],
        category: str
    ):
        """Log a failure case for error analysis"""
        self.failure_log.append({
            'query_id': example.query_id,
            'paper_id': example.paper_id,
            'true_title': example.true_title,
            'context': example.citation_context,
            'retrieved_titles': retrieved_titles[:5],  # Top 5
            'category': category,
            'corpus_size': len(example.corpus_entries)
        })

    def compare_models(
        self,
        models: Dict[str, Any],
        examples: List[CitationExample],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same dataset.

        Args:
            models: Dict of {model_name: model_object}
            examples: List of examples to evaluate on
            output_path: Optional path to save comparison table

        Returns:
            DataFrame with comparison results
        """
        print(f"\n📊 Comparing {len(models)} models on {len(examples)} examples")
        print("=" * 70)

        results = []

        for model_name, model in models.items():
            print(f"\n▶️  Evaluating: {model_name}")
            eval_results = self.evaluate_model(model, examples, verbose=True)

            metrics = eval_results['metrics']
            results.append({
                'Model': model_name,
                'R@1': metrics.recall_at_1,
                'R@5': metrics.recall_at_5,
                'R@10': metrics.recall_at_10,
                'R@20': metrics.recall_at_20,
                'P@1': metrics.precision_at_1,
                'P@5': metrics.precision_at_5,
                'P@10': metrics.precision_at_10,
                'MRR': metrics.mrr,
                'Exact Match': metrics.exact_match_rate,
                'Avg Rank': metrics.avg_rank,
                'Avg Latency (ms)': eval_results['avg_latency_ms'],
                'Total Time (s)': eval_results['total_time_sec']
            })

        # Create comparison DataFrame
        df = pd.DataFrame(results)

        # Sort by R@10 (primary metric)
        df = df.sort_values('R@10', ascending=False)

        print("\n" + "=" * 70)
        print("📋 MODEL COMPARISON RESULTS")
        print("=" * 70)
        print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        print("=" * 70)

        # Save if path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\n💾 Results saved to: {output_path}")

        return df

    def save_results(
        self,
        results: Dict,
        output_dir: str,
        model_name: str,
        include_predictions: bool = True
    ):
        """
        Save evaluation results to disk.

        Args:
            results: Results dict from evaluate_model()
            output_dir: Directory to save results
            model_name: Name of model for filename
            include_predictions: Whether to save full predictions
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{model_name}_{timestamp}"

        # Save metrics summary
        metrics_file = output_path / f"{base_name}_metrics.json"
        metrics_data = {
            'model': model_name,
            'timestamp': timestamp,
            'metrics': results['metrics'].to_dict(),
            'avg_latency_ms': results['avg_latency_ms'],
            'total_time_sec': results['total_time_sec'],
            'num_queries': results['metrics'].num_queries
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        print(f"💾 Metrics saved to: {metrics_file}")

        # Save failure log
        if results['failure_log']:
            failures_file = output_path / f"{base_name}_failures.json"
            with open(failures_file, 'w') as f:
                json.dump(results['failure_log'], f, indent=2, ensure_ascii=False)

            print(f"💾 Failure log saved to: {failures_file}")

            # Create failure analysis CSV
            failures_df = pd.DataFrame(results['failure_log'])
            failures_csv = output_path / f"{base_name}_failures.csv"
            failures_df.to_csv(failures_csv, index=False)

        # Save full predictions if requested
        if include_predictions:
            predictions_file = output_path / f"{base_name}_predictions.json"
            with open(predictions_file, 'w') as f:
                json.dump(results['predictions'], f, indent=2, ensure_ascii=False)

            print(f"💾 Predictions saved to: {predictions_file}")

    def analyze_failures(
        self,
        failure_log: List[Dict],
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Analyze failure patterns.

        Args:
            failure_log: List of failure cases
            output_path: Optional path to save analysis

        Returns:
            Dict with failure analysis
        """
        if not failure_log:
            print("No failures to analyze!")
            return {}

        df = pd.DataFrame(failure_log)

        analysis = {
            'total_failures': len(failure_log),
            'by_category': df['category'].value_counts().to_dict(),
            'avg_corpus_size': df['corpus_size'].mean(),
            'corpus_size_quartiles': df['corpus_size'].quantile([0.25, 0.5, 0.75]).to_dict()
        }

        print("\n📊 Failure Analysis:")
        print(f"   Total failures: {analysis['total_failures']}")
        print(f"\n   By category:")
        for cat, count in analysis['by_category'].items():
            print(f"      {cat}: {count}")
        print(f"\n   Avg corpus size: {analysis['avg_corpus_size']:.1f}")

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\n💾 Analysis saved to: {output_path}")

        return analysis
