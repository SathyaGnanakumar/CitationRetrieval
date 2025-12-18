"""
Self-evolving retrieval system using GEPA optimization.

Core engine for continuous learning and automated module optimization.
"""

import logging
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

import dspy

from src.evaluation.eval_store import EvaluationStore, QueryEvaluation, is_evolution_enabled
from src.evaluation.dspy_metrics import citation_retrieval_metric
from src.evaluation.metrics import evaluate_retrieval
from src.agents.self_evolve.version_tracker import VersionTracker
from src.agents.self_evolve.optimizers import GEPAAdapter, get_gepa_optimizer, optimize_with_gepa

logger = logging.getLogger(__name__)


class SelfEvolvingRetrievalSystem:
    """
    Self-evolving retrieval system with GEPA optimization.

    Continuously evaluates queries, collects metrics, and optimizes DSPy modules
    using GPT-5 Mini as reflection LM.
    """

    def __init__(
        self,
        workflow,
        resources: Dict[str, Any],
        dataset_path: str,
        local_model: str = "ollama_chat/gemma:2b",
        teacher_model: str = "openai/gpt-5-mini",
        data_dir: str = "./data",
    ):
        """
        Initialize self-evolving system.

        Args:
            workflow: RetrievalWorkflow instance
            resources: Built resources (indexes, embeddings)
            dataset_path: Path to ScholarCopilot dataset
            local_model: Local model for inference (Gemma via Ollama)
            teacher_model: Teacher model for optimization (GPT-5 Mini)
            data_dir: Data directory for storage
        """
        # Check evolution flag
        self._evolution_enabled = is_evolution_enabled()

        if not self._evolution_enabled:
            logger.info("DSPy Evolution is DISABLED. Set ENABLE_DSPY_EVOLUTION=true to enable.")
            return

        logger.info("🔄 DSPy Evolution ENABLED - Initializing self-evolving system...")

        self.workflow = workflow
        self.resources = resources
        self.dataset_path = dataset_path
        self.data_dir = Path(data_dir)

        # Configure local student model
        self.local_model = local_model
        self.teacher_model = teacher_model

        try:
            self.lm = dspy.LM(local_model, api_base="http://localhost:11434", api_key="")
            dspy.configure(lm=self.lm)
            logger.info(f"✓ Configured local model: {local_model}")
        except Exception as e:
            logger.warning(f"Could not configure local model: {e}")
            self.lm = None

        # Initialize stores
        self.eval_store = EvaluationStore(storage_dir=str(self.data_dir / "evaluations"))
        self.picker_tracker = VersionTracker(
            "picker", storage_dir=str(self.data_dir / "module_versions")
        )
        self.reformulator_tracker = VersionTracker(
            "reformulator", storage_dir=str(self.data_dir / "module_versions")
        )

        logger.info("✓ Initialized evaluation store and version trackers")

        # Load dataset
        try:
            # Import here to avoid module loading issues
            _parent_dir = Path(__file__).resolve().parents[3]
            if str(_parent_dir) not in sys.path:
                sys.path.insert(0, str(_parent_dir))

            from corpus_loaders.scholarcopilot import load_dataset as load_scholarcopilot_dataset

            self.dataset = load_scholarcopilot_dataset(dataset_path)
            logger.info(f"✓ Loaded dataset: {len(self.dataset)} papers")
        except Exception as e:
            logger.error(f"Could not load dataset: {e}")
            self.dataset = []

    def evaluate_batch(
        self,
        papers: List[Dict[str, Any]],
        k: int = 20,
        max_queries: Optional[int] = None,
        checkpoint_every: Optional[int] = None,
        checkpoint_file: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation on a batch of queries and store results.

        Args:
            papers: List of papers to evaluate
            k: Number of results to retrieve per query
            max_queries: Maximum number of queries to evaluate

        Returns:
            Dict with 'score' (weighted), 'R@5', 'R@10', 'MRR' averages
        """
        if not self._evolution_enabled:
            logger.debug("Evolution disabled - skipping evaluation")
            return {"score": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0}

        papers = papers[:max_queries] if max_queries else papers
        total_queries = len(papers)
        logger.info(f"Evaluating {total_queries} queries...")

        scores = []
        all_r5 = []
        all_r10 = []
        all_mrr = []
        skipped_no_gt = 0  # Count papers skipped due to no ground truth

        from langchain_core.messages import HumanMessage

        progress = tqdm(
            enumerate(papers, 1),
            total=total_queries,
            desc="Evaluating",
            unit="query",
            leave=False,
        )

        for i, paper in progress:
            progress.set_postfix({"query": i})
            logger.info(f"Query {i}/{total_queries} starting")
            # Extract query
            paper_text = paper.get("paper", "")
            if not paper_text:
                paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"

            query_words = paper_text.split()[:50]
            query_str = " ".join(query_words)

            # Get ground truth
            from evaluate import get_ground_truth_ids

            relevant_ids = get_ground_truth_ids(paper)

            if not relevant_ids:
                logger.debug(f"Query {i}: No ground truth IDs found, skipping")
                skipped_no_gt += 1
                continue

            if i <= 3:  # Debug first 3 queries
                logger.info(
                    f"Query {i} ground_truth_ids ({len(relevant_ids)}): {list(relevant_ids)[:5]}"
                )

            # Run workflow
            try:
                final_state = self.workflow.run(
                    {
                        "messages": [HumanMessage(content=query_str)],
                        "resources": self.resources,
                        "config": {"k": k, "enable_dspy_picker": True},
                    }
                )
            except Exception as e:
                logger.error(f"Query {i} failed: {e}")
                continue

            # Extract results
            ranked_papers = final_state.get("ranked_papers", [])
            retrieved_ids = [
                p[0].get("id") if isinstance(p, tuple) else p.get("id") for p in ranked_papers
            ]

            if i <= 3:  # Debug first 3 queries
                logger.info(f"Query {i} retrieved_ids ({len(retrieved_ids)}): {retrieved_ids[:5]}")
                overlap = set(str(rid) for rid in retrieved_ids) & set(
                    str(gid) for gid in relevant_ids
                )
                logger.info(f"Query {i} overlap: {len(overlap)} hits")

            # Calculate metrics
            metrics = evaluate_retrieval(
                [p[0] if isinstance(p, tuple) else p for p in ranked_papers],
                relevant_ids,
                k_values=[5, 10, 20],
            )

            if i <= 3:  # Debug first 3 queries
                logger.info(f"Query {i} metrics: {metrics}")

            # Track individual metrics
            r5 = metrics.get("R@5", 0)
            r10 = metrics.get("R@10", 0)
            mrr = metrics.get("MRR", 0)
            all_r5.append(r5)
            all_r10.append(r10)
            all_mrr.append(mrr)

            # Overall score (weighted combination)
            score = 0.4 * r5 + 0.3 * r10 + 0.3 * mrr
            scores.append(score)

            # Store evaluation
            evaluation = QueryEvaluation(
                query=query_str,
                paper_id=paper.get("paper_id", ""),
                retrieved_ids=retrieved_ids,
                relevant_ids=list(relevant_ids),
                metrics=metrics,
                inputs={"query": query_str, "paper": paper},
                outputs={"ranked_papers": ranked_papers, "final_state": final_state},
                score=score,
            )
            self.eval_store.add_evaluation(evaluation)

            if i % 10 == 0:
                avg_so_far = sum(scores) / len(scores)
                logger.info(f"Progress: {i}/{len(papers)}, Avg Score: {avg_so_far:.3f}")

            # Lightweight checkpointing so long runs can be resumed/monitored
            if checkpoint_every and i % checkpoint_every == 0:
                ckpt_path = (
                    Path(checkpoint_file)
                    if checkpoint_file
                    else self.data_dir / "eval_checkpoint.json"
                )
                avg_so_far = sum(scores) / len(scores) if scores else 0.0
                checkpoint_payload = {
                    "processed": i,
                    "total": total_queries,
                    "avg_score": avg_so_far,
                }
                try:
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(ckpt_path, "w") as f:
                        json.dump(checkpoint_payload, f, indent=2)
                    logger.info(f"Checkpoint saved: {checkpoint_payload}")
                except Exception as e:
                    logger.warning(f"Could not save checkpoint to {ckpt_path}: {e}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_r5 = sum(all_r5) / len(all_r5) if all_r5 else 0.0
        avg_r10 = sum(all_r10) / len(all_r10) if all_r10 else 0.0
        avg_mrr = sum(all_mrr) / len(all_mrr) if all_mrr else 0.0

        logger.info(
            f"Batch evaluation complete. Scored: {len(scores)}/{total_queries} "
            f"(skipped {skipped_no_gt} no ground-truth). Average score: {avg_score:.3f}"
        )

        return {
            "score": avg_score,
            "R@5": avg_r5,
            "R@10": avg_r10,
            "MRR": avg_mrr,
            "num_evaluated": len(scores),
            "num_skipped": skipped_no_gt,
        }

    def optimize_module(
        self, module_name: str = "picker", min_score: float = 0.5, auto_budget: str = "medium"
    ) -> Optional[Any]:
        """
        Optimize a DSPy module using GEPA.

        Args:
            module_name: Name of module to optimize ('picker' or 'reformulator')
            min_score: Minimum score threshold for training data
            auto_budget: GEPA budget level ('light', 'medium', 'heavy')

        Returns:
            Optimized module or None if optimization failed
        """
        if not self._evolution_enabled:
            logger.debug("Evolution disabled - skipping optimization")
            return None

        logger.info(f"\n{'='*70}")
        logger.info(f"⚡ OPTIMIZATION TRIGGERED: {module_name}")
        logger.info(f"{'='*70}")

        # Get training data
        all_examples = self.eval_store.to_dspy_trainset(min_score=min_score)

        if len(all_examples) < 10:
            logger.warning(f"Only {len(all_examples)} training examples. Need more data!")
            return None

        # Split into train/val (90/10)
        split_idx = int(0.9 * len(all_examples))
        trainset = [
            {"citation_context": ex.citation_context, "ground_truth_ids": ex.ground_truth_ids}
            for ex in all_examples[:split_idx]
        ]
        valset = [
            {"citation_context": ex.citation_context, "ground_truth_ids": ex.ground_truth_ids}
            for ex in all_examples[split_idx:]
        ]

        logger.info(f"GEPA: Training on {len(all_examples)} examples (score >= {min_score})")
        logger.info(f"GEPA: Split into trainset={len(trainset)}, valset={len(valset)}")

        # Create GEPA adapter
        adapter = GEPAAdapter(
            workflow=self.workflow, resources=self.resources, metric=citation_retrieval_metric
        )

        # Get GEPA optimizer config
        optimizer_config = get_gepa_optimizer(
            metric=citation_retrieval_metric,
            reflection_lm=self.teacher_model,
            max_metric_calls=50,
            auto=auto_budget,
            num_threads=4,
        )

        # Seed candidate (initial prompt)
        seed_candidate = {
            "system_prompt": "Select the most relevant paper from the candidates based on the citation context."
        }

        # Run GEPA optimization
        logger.info(f"GEPA: Starting reflective optimization (auto='{auto_budget}')")

        try:
            result = optimize_with_gepa(
                adapter=adapter,
                seed_candidate=seed_candidate,
                trainset=trainset,
                valset=valset,
                **optimizer_config,
            )

            logger.info(f"GEPA: Optimization complete (best_score={result.best_score:.4f})")

            # Save new version
            tracker = self.picker_tracker if module_name == "picker" else self.reformulator_tracker

            # TODO: Extract optimized module from result
            # For now, we log the success
            logger.info(f"✓ Optimization completed for {module_name}")

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None

    def continuous_evolution_loop(
        self,
        eval_interval: int = 100,
        optimize_interval: int = 1000,
        improvement_threshold: float = 0.05,
    ):
        """
        Continuous learning loop.

        Args:
            eval_interval: Evaluate every N queries
            optimize_interval: Optimize every N queries
            improvement_threshold: Minimum improvement to deploy new version
        """
        if not self._evolution_enabled:
            logger.info("Evolution disabled. Set ENABLE_DSPY_EVOLUTION=true to enable.")
            return

        logger.info("🔄 Starting continuous evolution loop...")
        logger.info(f"  Eval interval: {eval_interval}")
        logger.info(f"  Optimize interval: {optimize_interval}")
        logger.info(f"  Improvement threshold: {improvement_threshold}")

        query_count = 0
        last_optimization_score = 0.0

        # Process dataset in batches
        batch_size = eval_interval
        for i in range(0, len(self.dataset), batch_size):
            batch = self.dataset[i : i + batch_size]

            # Evaluate current version
            current_score = self.evaluate_batch(batch, max_queries=eval_interval)
            query_count += len(batch)

            logger.info(f"Processed {query_count} queries. Current score: {current_score:.3f}")

            # Time to optimize?
            if query_count >= optimize_interval:
                logger.info(f"\n{'='*70}")
                logger.info("⚡ OPTIMIZATION TRIGGERED")
                logger.info(f"{'='*70}")

                # Run optimization
                optimized = self.optimize_module(
                    module_name="picker", min_score=0.5, auto_budget="medium"
                )

                if optimized:
                    # Evaluate improvement
                    improvement = current_score - last_optimization_score

                    if improvement >= improvement_threshold:
                        logger.info(f"✅ Deploying improved module (+{improvement:.3f})")
                        # Deploy by saving to version tracker
                    else:
                        logger.info(
                            f"⚠️  Improvement too small (+{improvement:.3f}), keeping current version"
                        )

                    last_optimization_score = current_score

                query_count = 0  # Reset counter

                logger.info(f"{'='*70}\n")

        logger.info("Continuous evolution loop completed.")
