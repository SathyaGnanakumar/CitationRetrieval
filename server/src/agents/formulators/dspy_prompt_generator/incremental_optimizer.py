"""
Incremental DSPy Optimization Script.

Runs optimization in batches of 100 examples from ScholarCopilot dataset,
with checkpointing and Recall@k evaluation.

Requirements:
- OpenAI API key (DSPy optimization currently limited to OpenAI models)
- ScholarCopilot dataset prepared with data_prep.py

Usage:
    python incremental_optimizer.py --batch-size 100 --max-batches 10
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2

from .modules import get_module, RerankAndSelect
from .trainer import (
    load_examples_from_json,
    exact_match_metric,
    fuzzy_match_metric,
)


def recall_at_k_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    k: int = 5,
    trace=None
) -> float:
    """
    Calculate Recall@k for DSPy optimization.

    Returns 1.0 if correct paper is in top-k, 0.0 otherwise.
    """
    true_title = example.positive_title.lower().strip().strip('{}"\' ')

    # Get ranked titles from prediction
    ranked_titles = []
    if hasattr(prediction, 'ranked_titles') and prediction.ranked_titles:
        ranked_titles = [
            t.lower().strip().strip('{}"\' ')
            for t in prediction.ranked_titles.split(',')
        ]
    elif hasattr(prediction, 'selected_title') and prediction.selected_title:
        ranked_titles = [prediction.selected_title.lower().strip().strip('{}"\' ')]
    else:
        return 0.0

    # Check if true title in top-k
    top_k = ranked_titles[:k]

    # Exact match
    if true_title in top_k:
        return 1.0

    # Fuzzy match (>80% word overlap)
    true_words = set(true_title.split())
    for pred in top_k:
        pred_words = set(pred.split())
        if true_words and pred_words:
            overlap = len(true_words & pred_words)
            union = len(true_words | pred_words)
            if union > 0 and (overlap / union) > 0.8:
                return 1.0

    return 0.0


def recall_at_5(example, pred, trace=None):
    """Recall@5 wrapper."""
    return recall_at_k_metric(example, pred, k=5, trace=trace)


def recall_at_10(example, pred, trace=None):
    """Recall@10 wrapper."""
    return recall_at_k_metric(example, pred, k=10, trace=trace)


def recall_at_20(example, pred, trace=None):
    """Recall@20 wrapper."""
    return recall_at_k_metric(example, pred, k=20, trace=trace)


def combined_metric(example, pred, trace=None):
    """
    Combined metric: weighted Recall@5 (50%), Recall@10 (30%), Recall@20 (20%).

    This is the primary optimization target.
    """
    r5 = recall_at_5(example, pred, trace)
    r10 = recall_at_10(example, pred, trace)
    r20 = recall_at_20(example, pred, trace)
    return 0.5 * r5 + 0.3 * r10 + 0.2 * r20


def evaluate_module(
    module: dspy.Module,
    examples: List[dspy.Example],
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate module on examples with multiple metrics.

    Returns:
        Dictionary with Recall@5, Recall@10, Recall@20, and combined scores
    """
    r5_scores = []
    r10_scores = []
    r20_scores = []
    exact_scores = []
    fuzzy_scores = []

    for i, example in enumerate(examples):
        try:
            prediction = module(
                citation_context=example.citation_context,
                candidates=example.candidates
            )

            r5 = recall_at_5(example, prediction)
            r10 = recall_at_10(example, prediction)
            r20 = recall_at_20(example, prediction)
            exact = float(exact_match_metric(example, prediction))
            fuzzy = fuzzy_match_metric(example, prediction)

            r5_scores.append(r5)
            r10_scores.append(r10)
            r20_scores.append(r20)
            exact_scores.append(exact)
            fuzzy_scores.append(fuzzy)

            if verbose and i < 5:
                print(f"\nExample {i+1}:")
                print(f"  Context: {example.citation_context[:100]}...")
                print(f"  True: {example.positive_title}")
                print(f"  Pred: {prediction.selected_title if hasattr(prediction, 'selected_title') else 'N/A'}")
                print(f"  R@5: {r5}, R@10: {r10}, Exact: {exact}")

        except Exception as e:
            print(f"⚠️  Error on example {i}: {e}")
            continue

    if not r5_scores:
        return {
            "recall@5": 0.0,
            "recall@10": 0.0,
            "recall@20": 0.0,
            "combined": 0.0,
            "exact_match": 0.0,
            "fuzzy_match": 0.0,
            "n_evaluated": 0
        }

    return {
        "recall@5": sum(r5_scores) / len(r5_scores),
        "recall@10": sum(r10_scores) / len(r10_scores),
        "recall@20": sum(r20_scores) / len(r20_scores),
        "combined": sum([combined_metric(examples[i], None, None) for i in range(len(examples))]) / len(examples),
        "exact_match": sum(exact_scores) / len(exact_scores),
        "fuzzy_match": sum(fuzzy_scores) / len(fuzzy_scores),
        "n_evaluated": len(r5_scores)
    }


class IncrementalOptimizer:
    """
    Incremental optimizer that processes ScholarCopilot dataset in batches.

    Features:
    - Batch processing (default: 100 examples per batch)
    - Checkpointing (can resume from previous run)
    - Recall@k optimization target
    - OpenAI teacher LLM (required)
    """

    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        output_dir: str = "dspy_prompt_generator/optimized",
        batch_size: int = 100,
        teacher_model: str = "gpt-5-mini-2025-08-07",
        student_model: str = "gpt-5-mini-2025-08-07",
        module_name: str = "rerank",
        optimizer_type: str = "bootstrap"
    ):
        """
        Initialize incremental optimizer.

        Args:
            train_path: Path to training data JSON
            val_path: Path to validation data JSON
            test_path: Path to test data JSON
            output_dir: Directory for checkpoints and outputs
            batch_size: Number of examples per optimization batch
            teacher_model: OpenAI model for generating demonstrations
            student_model: Model to be optimized
            module_name: DSPy module to optimize ('simple', 'rerank', etc.)
            optimizer_type: 'bootstrap' or 'mipro'
        """
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.module_name = module_name
        self.optimizer_type = optimizer_type

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint file
        self.checkpoint_file = self.output_dir / "checkpoint.json"

        # Load checkpoint if exists
        self.state = self._load_checkpoint()

        # Configure DSPy with OpenAI
        self._configure_dspy()

    def _configure_dspy(self):
        """Configure DSPy with OpenAI models."""
        # Verify OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable required. "
                "DSPy optimization currently limited to OpenAI models."
            )

        # Teacher LM (for generating demonstrations)
        # Note: GPT-5 models don't support custom temperature, using default
        teacher_lm = dspy.OpenAI(
            model=self.teacher_model,
            max_tokens=800
        )

        # Student LM (model being optimized)
        # Note: GPT-5 models don't support custom temperature, using default
        student_lm = dspy.OpenAI(
            model=self.student_model,
            max_tokens=800
        )

        # Set as default (student will be used for inference)
        dspy.settings.configure(lm=student_lm, trace=[])

        self.teacher_lm = teacher_lm
        self.student_lm = student_lm

        print(f"✓ Configured DSPy with OpenAI")
        print(f"  Teacher: {self.teacher_model}")
        print(f"  Student: {self.student_model}")

    def _load_checkpoint(self) -> Dict:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
            print(f"✓ Loaded checkpoint from {self.checkpoint_file}")
            print(f"  Batch: {state.get('current_batch', 0)}/{state.get('total_batches', '?')}")
            print(f"  Best R@5: {state.get('best_metrics', {}).get('recall@5', 0.0):.4f}")
            return state
        return {
            "current_batch": 0,
            "total_batches": 0,
            "history": [],
            "best_metrics": None,
            "best_batch": None
        }

    def _save_checkpoint(self):
        """Save checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def run_batch(
        self,
        batch_idx: int,
        train_examples: List[dspy.Example],
        val_examples: List[dspy.Example]
    ) -> dspy.Module:
        """
        Run optimization on a single batch.

        Args:
            batch_idx: Batch index
            train_examples: Training examples for this batch
            val_examples: Validation examples

        Returns:
            Optimized module
        """
        print(f"\n{'='*60}")
        print(f"BATCH {batch_idx + 1}: Optimizing on {len(train_examples)} examples")
        print(f"{'='*60}")

        # Initialize module
        module = get_module(self.module_name)

        # Load previous best module if exists
        prev_module_path = self.output_dir / f"batch_{batch_idx - 1}_module.pkl"
        if batch_idx > 0 and prev_module_path.exists():
            try:
                module.load(str(prev_module_path))
                print(f"✓ Loaded previous module from batch {batch_idx - 1}")
            except Exception as e:
                print(f"⚠️  Could not load previous module: {e}")

        # Evaluate before optimization
        print("\n📊 Pre-optimization evaluation:")
        pre_metrics = evaluate_module(module, val_examples[:50], verbose=False)
        print(f"  R@5:  {pre_metrics['recall@5']:.4f}")
        print(f"  R@10: {pre_metrics['recall@10']:.4f}")
        print(f"  R@20: {pre_metrics['recall@20']:.4f}")
        print(f"  Combined: {pre_metrics['combined']:.4f}")

        # Run optimizer
        print(f"\n🔧 Running {self.optimizer_type} optimizer...")

        if self.optimizer_type == "bootstrap":
            optimizer = BootstrapFewShot(
                metric=combined_metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=4,
                teacher_settings=dict(lm=self.teacher_lm)
            )
        elif self.optimizer_type == "mipro":
            optimizer = MIPROv2(
                metric=combined_metric,
                num_candidates=10,
                init_temperature=1.0,
                prompt_model=self.teacher_lm,
                task_model=self.student_lm
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        # Optimize
        try:
            optimized_module = optimizer.compile(
                module,
                trainset=train_examples,
                valset=val_examples[:50]  # Limit val for speed
            )
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return module

        # Evaluate after optimization
        print("\n📊 Post-optimization evaluation:")
        post_metrics = evaluate_module(optimized_module, val_examples[:50], verbose=False)
        print(f"  R@5:  {post_metrics['recall@5']:.4f} (Δ {post_metrics['recall@5'] - pre_metrics['recall@5']:+.4f})")
        print(f"  R@10: {post_metrics['recall@10']:.4f} (Δ {post_metrics['recall@10'] - pre_metrics['recall@10']:+.4f})")
        print(f"  R@20: {post_metrics['recall@20']:.4f} (Δ {post_metrics['recall@20'] - pre_metrics['recall@20']:+.4f})")
        print(f"  Combined: {post_metrics['combined']:.4f} (Δ {post_metrics['combined'] - pre_metrics['combined']:+.4f})")

        # Save module
        module_path = self.output_dir / f"batch_{batch_idx}_module.pkl"
        optimized_module.save(str(module_path))
        print(f"\n✓ Saved optimized module to {module_path}")

        # Update state
        batch_result = {
            "batch_idx": batch_idx,
            "n_train": len(train_examples),
            "n_val": len(val_examples),
            "pre_metrics": pre_metrics,
            "post_metrics": post_metrics,
            "improvement": post_metrics['combined'] - pre_metrics['combined'],
            "timestamp": datetime.now().isoformat()
        }
        self.state["history"].append(batch_result)

        # Track best
        if (self.state["best_metrics"] is None or
            post_metrics['combined'] > self.state["best_metrics"]['combined']):
            self.state["best_metrics"] = post_metrics
            self.state["best_batch"] = batch_idx
            print(f"\n🏆 New best model! Combined score: {post_metrics['combined']:.4f}")

        return optimized_module

    def optimize_incremental(
        self,
        max_batches: Optional[int] = None,
        shuffle: bool = True
    ):
        """
        Run incremental optimization over the dataset.

        Args:
            max_batches: Maximum number of batches to process (None = all)
            shuffle: Shuffle training data before batching
        """
        print("\n" + "="*60)
        print("INCREMENTAL DSPy OPTIMIZATION")
        print("="*60)
        print(f"Module: {self.module_name}")
        print(f"Optimizer: {self.optimizer_type}")
        print(f"Batch size: {self.batch_size}")
        print(f"Teacher: {self.teacher_model}")

        # Load data
        print(f"\n📂 Loading data...")
        train_examples = load_examples_from_json(
            self.train_path,
            max_examples=None,
            shuffle=shuffle
        )
        val_examples = load_examples_from_json(
            self.val_path,
            max_examples=200,  # Limit validation set
            shuffle=False
        )
        test_examples = load_examples_from_json(
            self.test_path,
            max_examples=None,
            shuffle=False
        )

        print(f"✓ Loaded {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test")

        # Calculate batches
        n_batches = (len(train_examples) + self.batch_size - 1) // self.batch_size
        if max_batches:
            n_batches = min(n_batches, max_batches)

        self.state["total_batches"] = n_batches

        # Resume from checkpoint
        start_batch = self.state["current_batch"]

        # Run batches
        for batch_idx in range(start_batch, n_batches):
            # Get batch examples
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(train_examples))
            batch_train = train_examples[start_idx:end_idx]

            # Optimize
            optimized_module = self.run_batch(batch_idx, batch_train, val_examples)

            # Update checkpoint
            self.state["current_batch"] = batch_idx + 1
            self._save_checkpoint()

        # Final evaluation on test set
        print(f"\n{'='*60}")
        print("FINAL EVALUATION ON TEST SET")
        print(f"{'='*60}")

        best_module_path = self.output_dir / f"batch_{self.state['best_batch']}_module.pkl"
        if best_module_path.exists():
            final_module = get_module(self.module_name)
            final_module.load(str(best_module_path))

            test_metrics = evaluate_module(final_module, test_examples, verbose=True)

            print(f"\n📊 Test Set Results (Best model from batch {self.state['best_batch']}):")
            print(f"  R@5:  {test_metrics['recall@5']:.4f}")
            print(f"  R@10: {test_metrics['recall@10']:.4f}")
            print(f"  R@20: {test_metrics['recall@20']:.4f}")
            print(f"  Combined: {test_metrics['combined']:.4f}")
            print(f"  Exact Match: {test_metrics['exact_match']:.4f}")
            print(f"  Fuzzy Match: {test_metrics['fuzzy_match']:.4f}")

            self.state["test_metrics"] = test_metrics
            self._save_checkpoint()

        print(f"\n✅ Optimization complete!")
        print(f"📁 Results saved to {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Incremental DSPy Optimization")
    parser.add_argument(
        "--train-path",
        default="dspy_prompt_generator/data/train.json",
        help="Path to training data"
    )
    parser.add_argument(
        "--val-path",
        default="dspy_prompt_generator/data/val.json",
        help="Path to validation data"
    )
    parser.add_argument(
        "--test-path",
        default="dspy_prompt_generator/data/test.json",
        help="Path to test data"
    )
    parser.add_argument(
        "--output-dir",
        default="dspy_prompt_generator/optimized",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of examples per batch"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum batches to process (default: all)"
    )
    parser.add_argument(
        "--teacher-model",
        default="gpt-5-mini-2025-08-07",
        help="OpenAI teacher model"
    )
    parser.add_argument(
        "--student-model",
        default="gpt-5-mini-2025-08-07",
        help="OpenAI student model"
    )
    parser.add_argument(
        "--module",
        default="rerank",
        choices=["simple", "query", "rerank", "verify", "ensemble"],
        help="DSPy module to optimize"
    )
    parser.add_argument(
        "--optimizer",
        default="bootstrap",
        choices=["bootstrap", "mipro"],
        help="Optimization algorithm"
    )

    args = parser.parse_args()

    # Initialize and run
    optimizer = IncrementalOptimizer(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        module_name=args.module,
        optimizer_type=args.optimizer
    )

    optimizer.optimize_incremental(max_batches=args.max_batches)


if __name__ == "__main__":
    main()
