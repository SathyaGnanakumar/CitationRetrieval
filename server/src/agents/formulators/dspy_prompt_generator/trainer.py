"""
DSPy Training and Optimization Pipeline for Citation Retrieval.

Handles:
- Loading prepared data into DSPy format
- Defining evaluation metrics
- Running optimization (BootstrapFewShot, MIPRO)
- Saving and loading optimized modules
"""

import dspy
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

from .modules import (
    SimpleCitationRetriever,
    QueryThenRetrieve,
    RerankAndSelect,
    get_module,
)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Data
    train_path: str = "dspy_prompt_generator/data/train.json"
    val_path: str = "dspy_prompt_generator/data/val.json"
    test_path: str = "dspy_prompt_generator/data/test.json"
    
    # Sampling
    max_train_examples: int = 100  # Limit for faster iteration
    max_val_examples: int = 50
    max_test_examples: int = 100
    
    # Optimization
    optimizer: str = "bootstrap"  # 'bootstrap' or 'mipro'
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 4
    num_threads: int = 4
    
    # LLM
    model: str = "gpt-5-mini-2025-08-07"  # or "gpt-5-2025-08-07", "claude-3-5-sonnet-20241022"
    # Note: GPT-5 models don't support custom temperature, using default

    # Output
    output_dir: str = "dspy_prompt_generator/optimized"


def load_examples_from_json(
    path: str,
    max_examples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42
) -> List[dspy.Example]:
    """
    Load prepared data and convert to DSPy Examples.
    
    Args:
        path: Path to JSON file (train.json, val.json, or test.json)
        max_examples: Maximum examples to load
        shuffle: Whether to shuffle before limiting
        seed: Random seed
        
    Returns:
        List of DSPy Example objects
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
    
    if max_examples:
        data = data[:max_examples]
    
    examples = []
    for item in data:
        # Build candidate pool (positive + negatives)
        candidates = [
            {
                "title": item["positive_title"],
                "abstract": item["positive_abstract"]
            }
        ]
        for neg in item["negatives"]:
            candidates.append({
                "title": neg["title"],
                "abstract": neg["abstract"]
            })
        
        # Shuffle candidates so positive isn't always first
        random.shuffle(candidates)
        
        # Create DSPy Example
        example = dspy.Example(
            citation_context=item["citation_context"],
            candidates=candidates,
            # Ground truth for evaluation
            positive_title=item["positive_title"]
        ).with_inputs("citation_context", "candidates")
        
        examples.append(example)
    
    return examples


def exact_match_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None
) -> bool:
    """
    Check if predicted title exactly matches the ground truth.
    
    Args:
        example: DSPy Example with positive_title
        prediction: Model prediction with selected_title
        trace: Optional trace for debugging
        
    Returns:
        True if exact match, False otherwise
    """
    pred_title = prediction.selected_title.lower().strip()
    true_title = example.positive_title.lower().strip()
    
    # Clean titles (remove leading/trailing punctuation)
    pred_title = pred_title.strip('{}"\' ')
    true_title = true_title.strip('{}"\' ')
    
    return pred_title == true_title


def fuzzy_match_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None
) -> float:
    """
    Fuzzy match score based on title overlap.
    
    Args:
        example: DSPy Example with positive_title
        prediction: Model prediction with selected_title
        trace: Optional trace for debugging
        
    Returns:
        Score between 0 and 1
    """
    pred_title = prediction.selected_title.lower().strip()
    true_title = example.positive_title.lower().strip()
    
    # Clean titles
    pred_title = pred_title.strip('{}"\' ')
    true_title = true_title.strip('{}"\' ')
    
    # Exact match = 1.0
    if pred_title == true_title:
        return 1.0
    
    # Check if one contains the other
    if pred_title in true_title or true_title in pred_title:
        return 0.8
    
    # Word overlap
    pred_words = set(pred_title.split())
    true_words = set(true_title.split())
    
    if not pred_words or not true_words:
        return 0.0
    
    overlap = len(pred_words & true_words)
    union = len(pred_words | true_words)
    
    return overlap / union if union > 0 else 0.0


def contains_match_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None
) -> bool:
    """
    Check if the prediction contains key words from the correct title.
    More lenient than exact match.
    """
    pred_title = prediction.selected_title.lower().strip()
    true_title = example.positive_title.lower().strip()
    
    # Clean
    pred_title = pred_title.strip('{}"\' ')
    true_title = true_title.strip('{}"\' ')
    
    # Get significant words (length > 3, not common words)
    stop_words = {'the', 'for', 'and', 'with', 'from', 'that', 'this', 'are', 'was'}
    true_words = [w for w in true_title.split() if len(w) > 3 and w not in stop_words]
    
    # Check if at least 50% of significant words appear
    if not true_words:
        return pred_title == true_title
    
    matches = sum(1 for w in true_words if w in pred_title)
    return matches >= len(true_words) * 0.5


class CitationTrainer:
    """
    Trainer for DSPy citation retrieval modules.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.train_examples = []
        self.val_examples = []
        self.test_examples = []
        self.module = None
        self.optimized_module = None
    
    def setup_lm(self):
        """Configure the language model."""
        print(f"🔧 Setting up LM: {self.config.model}")

        # Note: GPT-5 models don't support custom temperature, using default
        lm = dspy.LM(
            model=self.config.model,
            max_tokens=1000
        )
        dspy.configure(lm=lm)

        print("✅ LM configured")
    
    def load_data(self):
        """Load train/val/test data."""
        print("📂 Loading data...")
        
        self.train_examples = load_examples_from_json(
            self.config.train_path,
            max_examples=self.config.max_train_examples
        )
        print(f"   Train: {len(self.train_examples)} examples")
        
        self.val_examples = load_examples_from_json(
            self.config.val_path,
            max_examples=self.config.max_val_examples
        )
        print(f"   Val: {len(self.val_examples)} examples")
        
        self.test_examples = load_examples_from_json(
            self.config.test_path,
            max_examples=self.config.max_test_examples
        )
        print(f"   Test: {len(self.test_examples)} examples")
    
    def evaluate(
        self,
        module: dspy.Module,
        examples: List[dspy.Example],
        metric: Callable = exact_match_metric,
        verbose: bool = False
    ) -> Dict:
        """
        Evaluate a module on examples.
        
        Args:
            module: DSPy module to evaluate
            examples: List of examples
            metric: Metric function
            verbose: Print detailed results
            
        Returns:
            Dict with accuracy and details
        """
        correct = 0
        total = len(examples)
        results = []
        
        for i, example in enumerate(examples):
            try:
                prediction = module(
                    citation_context=example.citation_context,
                    candidates=example.candidates
                )
                
                score = metric(example, prediction)
                correct += score
                
                results.append({
                    "example_id": i,
                    "correct": score,
                    "predicted": prediction.selected_title,
                    "actual": example.positive_title
                })
                
                if verbose and not score:
                    print(f"❌ Example {i}:")
                    print(f"   Predicted: {prediction.selected_title[:50]}...")
                    print(f"   Actual: {example.positive_title[:50]}...")
                    
            except Exception as e:
                print(f"⚠️ Error on example {i}: {e}")
                results.append({
                    "example_id": i,
                    "correct": 0,
                    "error": str(e)
                })
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
    
    def optimize(
        self,
        module_name: str = "simple",
        metric: Callable = exact_match_metric
    ) -> dspy.Module:
        """
        Optimize a module using the configured optimizer.
        
        Args:
            module_name: Name of module ('simple', 'query', 'rerank')
            metric: Metric function for optimization
            
        Returns:
            Optimized DSPy module
        """
        print(f"\n Starting optimization with {self.config.optimizer}...")
        print(f"   Module: {module_name}")
        print(f"   Train examples: {len(self.train_examples)}")
        
        # Get base module
        self.module = get_module(module_name)
        
        # Choose optimizer
        if self.config.optimizer == "bootstrap":
            optimizer = dspy.BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
            )
        elif self.config.optimizer == "mipro":
            optimizer = dspy.MIPROv2(
                metric=metric,
                num_threads=self.config.num_threads,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Run optimization
        print("⏳ Running optimization (this may take a while)...")
        
        self.optimized_module = optimizer.compile(
            self.module,
            trainset=self.train_examples,
        )
        
        print(" Optimization complete!")
        
        return self.optimized_module
    
    def compare_before_after(
        self,
        examples: Optional[List[dspy.Example]] = None,
        metric: Callable = exact_match_metric
    ):
        """
        Compare performance before and after optimization.
        """
        if examples is None:
            examples = self.val_examples
        
        print("\n Comparing before/after optimization:")
        print("-" * 50)
        
        if self.module:
            before = self.evaluate(self.module, examples, metric)
            print(f"Before optimization: {before['accuracy']:.2%} ({before['correct']}/{before['total']})")
        
        if self.optimized_module:
            after = self.evaluate(self.optimized_module, examples, metric)
            print(f"After optimization:  {after['accuracy']:.2%} ({after['correct']}/{after['total']})")
            
            if self.module:
                improvement = after['accuracy'] - before['accuracy']
                print(f"Improvement: {improvement:+.2%}")
    
    def save_optimized(self, name: str = "optimized_module"):
        """Save the optimized module."""
        if self.optimized_module is None:
            print(" No optimized module to save")
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{name}.json"
        self.optimized_module.save(str(output_path))
        
        print(f" Saved optimized module to {output_path}")
    
    def load_optimized(self, path: str, module_name: str = "simple"):
        """Load an optimized module."""
        self.module = get_module(module_name)
        self.optimized_module = self.module.load(path)
        print(f" Loaded optimized module from {path}")
        return self.optimized_module


def main():
    """Run training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DSPy citation retrieval")
    parser.add_argument(
        "--module",
        type=str,
        default="simple",
        choices=["simple", "query", "rerank"],
        help="Module type to optimize"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="bootstrap",
        choices=["bootstrap", "mipro"],
        help="Optimization strategy"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="LLM model to use"
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=50,
        help="Max training examples"
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=20,
        help="Max validation examples"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation, no optimization"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model=args.model,
        optimizer=args.optimizer,
        max_train_examples=args.max_train,
        max_val_examples=args.max_val,
    )
    
    # Create trainer
    trainer = CitationTrainer(config)
    
    # Setup
    trainer.setup_lm()
    trainer.load_data()
    
    if args.eval_only:
        # Just evaluate the base module
        module = get_module(args.module)
        print(f"\n📊 Evaluating {args.module} module...")
        results = trainer.evaluate(module, trainer.val_examples, verbose=True)
        print(f"\nAccuracy: {results['accuracy']:.2%}")
    else:
        # Run optimization
        trainer.optimize(module_name=args.module)
        
        # Compare results
        trainer.compare_before_after()
        
        # Save
        trainer.save_optimized(f"{args.module}_{args.optimizer}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()


