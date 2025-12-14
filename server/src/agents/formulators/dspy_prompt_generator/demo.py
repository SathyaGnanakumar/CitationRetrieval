#!/usr/bin/env python3
"""
Demo script for DSPy Citation Retrieval.

This script demonstrates how to use the DSPy prompt generator
for citation retrieval without running full optimization.

Usage:
    # Quick test with a single example
    python dspy_prompt_generator/demo.py --quick
    
    # Evaluate on validation set
    python dspy_prompt_generator/demo.py --evaluate --num-examples 10
    
    # Run optimization (requires API key)
    python dspy_prompt_generator/demo.py --optimize --num-examples 20
"""

import os
import sys
import json
import argparse
import random

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import dspy
        print(f"✅ DSPy version: {dspy.__version__ if hasattr(dspy, '__version__') else 'installed'}")
        return True
    except ImportError:
        print("❌ DSPy not installed. Install with: pip install dspy-ai")
        return False


def quick_demo():
    """Run a quick demo with a single example."""
    print("\n" + "=" * 60)
    print("🚀 QUICK DEMO - Single Example Test")
    print("=" * 60)
    
    # Load one example
    with open("dspy_prompt_generator/data/val.json", 'r') as f:
        data = json.load(f)
    
    example = data[0]
    
    print("\n Citation Context:")
    print("-" * 40)
    print(example["citation_context"][:300] + "...")
    
    print("\n Ground Truth:")
    print("-" * 40)
    print(f"Title: {example['positive_title']}")
    
    print("\n Negatives (candidates to distinguish from):")
    print("-" * 40)
    for i, neg in enumerate(example["negatives"][:3], 1):
        print(f"  {i}. {neg['title'][:60]}...")
    
    print("\n" + "=" * 60)
    print("To run actual inference, set OPENAI_API_KEY and run:")
    print("  python dspy_prompt_generator/demo.py --evaluate --num-examples 5")
    print("=" * 60)


def evaluate_demo(num_examples: int = 5, model: str = "gpt-4o-mini"):
    """Run evaluation on a few examples."""
    import dspy
    from dspy_prompt_generator import (
        SimpleCitationRetriever,
        load_examples_from_json,
        exact_match_metric
    )
    
    print("\n" + "=" * 60)
    print(f"EVALUATION DEMO - {num_examples} examples")
    print("=" * 60)
    
    # Setup LLM
    print(f"\n Setting up LLM: {model}")
    lm = dspy.LM(model=model, temperature=0.0)
    dspy.configure(lm=lm)
    
    # Load examples
    print("📂 Loading validation examples...")
    examples = load_examples_from_json(
        "dspy_prompt_generator/data/val.json",
        max_examples=num_examples
    )
    print(f"   Loaded {len(examples)} examples")
    
    # Create module
    module = SimpleCitationRetriever()
    
    # Evaluate
    print("\n⏳ Running evaluation...")
    correct = 0
    
    for i, ex in enumerate(examples):
        print(f"\n--- Example {i+1}/{len(examples)} ---")
        print(f"Context: {ex.citation_context[:100]}...")
        print(f"Expected: {ex.positive_title[:50]}...")
        
        try:
            pred = module(
                citation_context=ex.citation_context,
                candidates=ex.candidates
            )
            
            is_correct = exact_match_metric(ex, pred)
            correct += is_correct
            
            status = "✅" if is_correct else "❌"
            print(f"Predicted: {pred.selected_title[:50]}...")
            print(f"Result: {status}")
            
            if not is_correct:
                print(f"Reasoning: {pred.reasoning[:100]}...")
                
        except Exception as e:
            print(f"⚠️ Error: {e}")
    
    accuracy = correct / len(examples) if examples else 0
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {correct}/{len(examples)} = {accuracy:.1%} accuracy")
    print("=" * 60)


def optimize_demo(num_examples: int = 20, model: str = "gpt-4o-mini"):
    """Run optimization on training examples."""
    import dspy
    from dspy_prompt_generator import (
        CitationTrainer,
        TrainingConfig,
    )
    
    print("\n" + "=" * 60)
    print(f"🎯 OPTIMIZATION DEMO - {num_examples} training examples")
    print("=" * 60)
    
    # Create config
    config = TrainingConfig(
        model=model,
        max_train_examples=num_examples,
        max_val_examples=min(num_examples // 2, 10),
        optimizer="bootstrap",
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
    )
    
    # Create trainer
    trainer = CitationTrainer(config)
    trainer.setup_lm()
    trainer.load_data()
    
    # Run optimization
    trainer.optimize(module_name="simple")
    
    # Compare results
    trainer.compare_before_after()
    
    # Save
    trainer.save_optimized("demo_optimized")
    
    print("\n" + "=" * 60)
    print("✅ Optimization complete!")
    print("   Optimized module saved to: dspy_prompt_generator/optimized/")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="DSPy Citation Retrieval Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dspy_prompt_generator/demo.py --quick
  python dspy_prompt_generator/demo.py --evaluate --num-examples 5
  python dspy_prompt_generator/demo.py --optimize --num-examples 20
        """
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick demo (no API calls)"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on validation examples"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run optimization on training examples"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of examples to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Default to quick demo if no option specified
    if not (args.quick or args.evaluate or args.optimize):
        args.quick = True
    
    if args.quick:
        quick_demo()
    
    if args.evaluate:
        # Check for API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("⚠️ OPENAI_API_KEY not set. Set it with:")
            print("   export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)
        evaluate_demo(args.num_examples, args.model)
    
    if args.optimize:
        # Check for API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("⚠️ OPENAI_API_KEY not set. Set it with:")
            print("   export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)
        optimize_demo(args.num_examples, args.model)


if __name__ == "__main__":
    main()

