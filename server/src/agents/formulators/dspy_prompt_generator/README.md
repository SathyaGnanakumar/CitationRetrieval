# DSPy Prompt Generator & Optimizer

This directory contains DSPy modules, signatures, and optimization tools for learning optimal prompts for citation retrieval.

## Overview

DSPy (Declarative Self-improving Language Programs) is a framework for programming LLMs with automatic prompt optimization. This implementation optimizes citation retrieval by learning from the ScholarCopilot dataset with hard negatives.

**⚠️ IMPORTANT: Optimization requires OpenAI API access.**
DSPy optimization currently only supports OpenAI models (GPT-4o, GPT-4o-mini, etc.) as the teacher LLM.

**📋 Optimization Scope Disclaimer:**
DSPy optimization is designed to work at **two levels** in the retrieval pipeline:
1. **Query Formulation Stage** - Optimizing query reformulation from citation contexts
2. **Reranking Stage** - Optimizing candidate reranking and selection

Due to time constraints, the current implementation focuses on **reranking stage optimization** (`RerankAndSelect` module). Future iterations will extend optimization to the query formulation stage.

## Directory Structure

```
dspy_prompt_generator/
├── README.md                    # This file
├── __init__.py
├── signatures.py                # DSPy signatures (input/output specs)
├── modules.py                   # DSPy modules (retrieval pipelines)
├── trainer.py                   # Training infrastructure
├── data_prep.py                 # Data preparation from ScholarCopilot
├── incremental_optimizer.py     # Incremental batch optimization
├── run_optimization.sh          # Quick-start script
├── data/                        # Training data (created by data_prep.py)
│   ├── train.json
│   ├── val.json
│   └── test.json
└── optimized/                   # Optimization outputs (created automatically)
    ├── checkpoint.json
    ├── batch_0_module.pkl
    ├── batch_1_module.pkl
    └── ...
```

## Quick Start

### 1. Prepare Training Data

Extract training examples from ScholarCopilot dataset with 1:9 positive:negative split:

```bash
python data_prep.py \
  --dataset ../../../corpus/scholarcopilot/scholar_copilot_eval_data_1k.json \
  --output-dir data/ \
  --train-ratio 0.7 \
  --val-ratio 0.15
```

This creates:
- `data/train.json`: Training examples (70%)
- `data/val.json`: Validation examples (15%)
- `data/test.json`: Test examples (15%)

Each example contains:
- Citation context with [CITATION] marker
- 10 candidate papers (1 correct + 9 hard negatives)
- Ground truth positive title

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run Optimization

**Option A: Using the quick-start script:**

```bash
./run_optimization.sh [batch_size] [max_batches] [teacher_model] [student_model] [module] [optimizer]
```

Examples:
```bash
# Default: 100 examples/batch, 10 batches, gpt-5-mini-2025-08-07, rerank module, bootstrap
./run_optimization.sh

# Custom: 200 examples/batch, 5 batches
./run_optimization.sh 200 5

# Use same model for teacher and student
./run_optimization.sh 100 10 gpt-5-mini-2025-08-07 gpt-5-mini-2025-08-07
```

**Option B: Direct Python invocation:**

```bash
python incremental_optimizer.py \
  --batch-size 100 \
  --max-batches 10 \
  --teacher-model gpt-5-mini-2025-08-07 \
  --student-model gpt-5-mini-2025-08-07 \
  --module rerank \
  --optimizer bootstrap
```

## Optimization Details

### Target Metric: Recall@k

The optimizer maximizes **combined Recall@k**:
- **Recall@5**: 50% weight (most important)
- **Recall@10**: 30% weight
- **Recall@20**: 20% weight

Formula: `Combined = 0.5 * R@5 + 0.3 * R@10 + 0.2 * R@20`

This aligns with the evaluation metrics used in the main retrieval system.

### Batch Processing

Optimization runs in **incremental batches**:

1. Load batch of N examples (default: 100)
2. Run optimizer (BootstrapFewShot or MIPROv2)
3. Evaluate on validation set
4. Save optimized module checkpoint
5. Repeat for next batch

**Benefits:**
- Fits limited compute/time budgets
- Checkpointing allows resuming if interrupted
- Tracks progress and best models
- Can stop early if performance plateaus

### Optimizers

**BootstrapFewShot (default)**:
- Fast and simple
- Generates few-shot demonstrations from training data
- Works well with small batches

**MIPROv2 (advanced)**:
- Slower but can find better prompts
- Explores multiple instruction variations
- Better for larger budgets

### Resuming from Checkpoint

If optimization is interrupted, simply re-run the same command. The optimizer automatically:
- Loads `optimized/checkpoint.json`
- Resumes from the last completed batch
- Continues optimization

## Using Optimized Models

After optimization, load and use the best model:

```python
from src.agents.formulators.dspy_prompt_generator.modules import RerankAndSelect
import dspy

# Configure DSPy
dspy.settings.configure(
    lm=dspy.OpenAI(model="gpt-5-mini-2025-08-07", temperature=0.0)
)

# Load optimized module
module = RerankAndSelect()
module.load("optimized/batch_5_module.pkl")  # Use best batch number

# Make predictions
prediction = module(
    citation_context="Recent advances in [CITATION] have shown...",
    candidates=[
        {"title": "Paper A", "abstract": "..."},
        {"title": "Paper B", "abstract": "..."},
        # ...
    ]
)

print(f"Selected: {prediction.selected_title}")
print(f"Ranked: {prediction.ranked_titles}")
```

## Available Modules

| Module                | Description                                          | Use Case                     |
|-----------------------|------------------------------------------------------|------------------------------|
| `SimpleCitationRetriever` | Direct selection with CoT                         | Fast inference               |
| `QueryThenRetrieve`   | Generate query first, then select                    | Better understanding         |
| `RerankAndSelect`     | Rerank all candidates, return top-k                  | **Recommended for training** |
| `VerifyAndSelect`     | Verify each candidate individually                   | High accuracy (slow)         |
| `EnsembleRetriever`   | Combine multiple strategies                          | Best performance             |

**Recommendation:** Use `RerankAndSelect` for optimization as it naturally produces top-k rankings for Recall@k evaluation.

## Command-Line Options

### data_prep.py

```bash
python data_prep.py \
  --dataset PATH                 # ScholarCopilot dataset path
  --output-dir DIR               # Output directory for splits
  --train-ratio FLOAT            # Training split ratio (default: 0.7)
  --val-ratio FLOAT              # Validation split ratio (default: 0.15)
  --max-negatives INT            # Max hard negatives per example (default: 9)
  --context-window INT           # Context chars around citation (default: 300)
  --seed INT                     # Random seed (default: 42)
```

### incremental_optimizer.py

```bash
python incremental_optimizer.py \
  --train-path PATH              # Training data JSON
  --val-path PATH                # Validation data JSON
  --test-path PATH               # Test data JSON
  --output-dir DIR               # Output directory (default: optimized/)
  --batch-size INT               # Examples per batch (default: 100)
  --max-batches INT              # Max batches (default: all)
  --teacher-model MODEL          # OpenAI teacher model (default: gpt-5-mini-2025-08-07)
  --student-model MODEL          # OpenAI student model (default: gpt-5-mini-2025-08-07)
  --module NAME                  # Module to optimize (default: rerank)
  --optimizer TYPE               # bootstrap or mipro (default: bootstrap)
```

## Performance Tips

### Cost Management

1. **Start small**: Use `--max-batches 3` for initial testing
2. **Use gpt-5-mini-2025-08-07**: The default model, cost-effective and performant
3. **Monitor usage**: Check OpenAI dashboard during optimization
4. **Batch sizing**: Larger batches (200-300) are more cost-efficient per example

### Optimization Quality

1. **More batches**: Generally improves performance (diminishing returns after ~10)
2. **Hard negatives**: Ensure negatives are challenging (from same bibliography)
3. **Data quality**: Clean citation contexts and accurate metadata
4. **Optimizer choice**: MIPROv2 for best quality (but slower)

### Compute Constraints

If compute/time is limited:
- Use smaller `--batch-size` (50-100)
- Limit `--max-batches` (5-10)
- Use `bootstrap` optimizer (faster than MIPROv2)
- Stop early if validation metrics plateau

## Monitoring Progress

### During Optimization

Watch for:
- **Pre/post metrics** for each batch
- **Improvement deltas** (Δ) - should be positive
- **Best model tracking** - which batch performed best
- **Validation R@5, R@10, R@20** - primary metrics

### After Optimization

Check `optimized/checkpoint.json`:

```json
{
  "current_batch": 10,
  "total_batches": 10,
  "best_batch": 7,
  "best_metrics": {
    "recall@5": 0.4823,
    "recall@10": 0.6412,
    "recall@20": 0.7654,
    "combined": 0.5598
  },
  "test_metrics": {
    "recall@5": 0.4701,
    "recall@10": 0.6298,
    "recall@20": 0.7532,
    "combined": 0.5486
  }
}
```

### Troubleshooting

**"OPENAI_API_KEY not set"**
- Export your OpenAI API key: `export OPENAI_API_KEY=...`

**"Training data not found"**
- Run `data_prep.py` first to create train/val/test splits

**"Optimization failed"**
- Check OpenAI API limits and billing
- Verify training data format (use `--verbose` for debugging)
- Try smaller batch size or different optimizer

**Low improvement (Δ ≈ 0)**
- Data may be too easy/hard - adjust negative sampling
- Try different module or optimizer
- Increase batch size for more training signal

## Example Workflow

Complete workflow from scratch:

```bash
# 1. Navigate to directory
cd server/src/agents/formulators/dspy_prompt_generator

# 2. Prepare data
python data_prep.py \
  --dataset ../../../corpus/scholarcopilot/scholar_copilot_eval_data_1k.json \
  --output-dir data/

# 3. Set API key
export OPENAI_API_KEY=sk-...

# 4. Run optimization (3 batches for testing)
./run_optimization.sh 100 3

# 5. Check results
cat optimized/checkpoint.json

# 6. Test optimized model
python -c "
from modules import RerankAndSelect
import dspy

dspy.settings.configure(lm=dspy.OpenAI(model='gpt-5-mini-2025-08-07'))
module = RerankAndSelect()
module.load('optimized/batch_2_module.pkl')
print('✓ Model loaded successfully')
"
```

## References

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [ScholarCopilot Dataset](https://github.com/atutej/ScholarCopilot)

## Support

For issues or questions:
1. Check the main project README
2. Review DSPy documentation
3. Open an issue on the project repository
