# Evaluation Scripts Documentation

This directory contains comprehensive evaluation scripts for the retrieval system.

## Quick Start

```bash
# Quick test (10 examples)
./run_evaluation.sh --quick

# Standard evaluation (50 examples)
./run_evaluation.sh

# Full evaluation with LLM reranker (100 examples)
./run_evaluation.sh --full --llm

# Full evaluation with DSPy modules
./run_evaluation.sh --full --dspy
```

## Main Evaluation Runner

**`run_evaluation.sh`** - Master script that runs all evaluations

```bash
./run_evaluation.sh [options]

Options:
  -n NUM      Number of examples (default: 50)
  -k K        Top-k results (default: 20)
  --llm       Use LLM-based reranker (Gemma 3 4B)
  --dspy      Use DSPy modules
  --no-cache  Disable caching
  --quick     Quick mode (10 examples)
  --full      Full mode (100 examples)
  --baseline  Only baseline reranking
  --system    Only system comparison
  --graphs    Only generate graphs
```

## Individual Scripts

### 1. `evaluate_baselines_with_reranking.py`

Evaluates individual retrievers (BM25, E5, SPECTER) with and without LLM reranking.

```bash
uv run python evaluate_baselines_with_reranking.py --num-examples 50 --llm-reranker
```

**Output:**
- `baseline_reranking_results/full_results.json` - Detailed per-query results
- `baseline_reranking_results/summary_table.csv` - Aggregated metrics
- `baseline_reranking_results/visualization.png` - Combined dashboard

**Metrics:**
- Recall@5, Recall@10, Recall@20
- MRR (Mean Reciprocal Rank)
- Comparison of baseline vs reranked performance

### 2. `compare_baselines_vs_system.py`

Compares baseline retrievers against the full system (aggregation + reranking + reformulation + picking).

```bash
uv run python compare_baselines_vs_system.py --num-examples 50 --llm-reranker --use-dspy
```

**Output:**
- `comparison_results/full_results.json` - Raw results
- `comparison_results/aggregated_results.json` - Summary statistics
- `comparison_results/comparison_table.csv` - Metrics table
- `comparison_results/bar_chart_comparison.png` - Bar chart
- `comparison_results/radar_chart_comparison.png` - Radar chart
- `comparison_results/heatmap_comparison.png` - Heatmap
- `comparison_results/improvement_chart.png` - Improvement percentages

### 3. `generate_individual_graphs.py`

Generates focused, individual graphs from baseline reranking results.

```bash
uv run python generate_individual_graphs.py
```

**Output:** `baseline_reranking_results/individual_graphs/`

**Graphs generated:**
- **Method Comparisons** (3 graphs): `bm25_comparison.png`, `e5_comparison.png`, `specter_comparison.png`
- **Recall Curves** (3 graphs): `bm25_recall_curve.png`, `e5_recall_curve.png`, `specter_recall_curve.png`
- **Improvement Charts** (3 graphs): `bm25_improvement.png`, `e5_improvement.png`, `specter_improvement.png`
- **Metric Views** (4 graphs): `metric_R_at_5.png`, `metric_R_at_10.png`, `metric_R_at_20.png`, `metric_MRR.png`
- **Overall MRR** (1 graph): `mrr_comparison_all.png`

### 4. `evaluate.py`

Unified evaluation CLI with multiple modes (used by other scripts).

```bash
# Pipeline mode - Full workflow
uv run python evaluate.py pipeline --num-examples 50

# Baselines mode - Individual retrievers
uv run python evaluate.py baselines --num-examples 50

# Retrievers mode - Benchmarking
uv run python evaluate.py retrievers
```

## Configuration

### Environment Variables (.env)

```bash
# Dataset
DATASET_DIR="/path/to/dataset.json"

# Models
LOCAL_LLM="google/gemma-3-4b-it"
OPENAI_API_KEY="sk-..."

# Hugging Face
hf_key="hf_..."

# Semantic Scholar (optional)
S2_API_KEY="..."
```

### LLM Reranker Configuration

The LLM reranker uses the model specified in `LOCAL_LLM` environment variable.

**Current default:** `google/gemma-3-4b-it`

**Configuration in** `src/agents/formulators/llm_agent.py`:
- Model: Gemma 3 4B Instruct
- Precision: bfloat16 (if CUDA available, else float32)
- Max tokens: 256
- Sampling: Disabled (greedy decoding)

## Workflow

The evaluation runner executes in this order:

1. **Baseline Reranking** (`evaluate_baselines_with_reranking.py`)
   - Tests each retriever individually
   - Applies LLM reranking to each
   - Generates comparison metrics

2. **System Comparison** (`compare_baselines_vs_system.py`)
   - Runs full system pipeline
   - Compares against baseline retrievers
   - Measures improvement from aggregation, reformulation, and picking

3. **Graph Generation** (`generate_individual_graphs.py`)
   - Creates individual, focused visualizations
   - Generates method-specific comparisons
   - Produces presentation-ready charts

## Output Structure

```
server/
├── baseline_reranking_results/
│   ├── full_results.json
│   ├── summary_table.csv
│   ├── visualization.png
│   └── individual_graphs/
│       ├── bm25_comparison.png
│       ├── e5_comparison.png
│       ├── specter_comparison.png
│       ├── bm25_recall_curve.png
│       ├── e5_recall_curve.png
│       ├── specter_recall_curve.png
│       ├── bm25_improvement.png
│       ├── e5_improvement.png
│       ├── specter_improvement.png
│       ├── metric_R_at_5.png
│       ├── metric_R_at_10.png
│       ├── metric_R_at_20.png
│       ├── metric_MRR.png
│       └── mrr_comparison_all.png
│
└── comparison_results/
    ├── full_results.json
    ├── aggregated_results.json
    ├── comparison_table.csv
    ├── bar_chart_comparison.png
    ├── radar_chart_comparison.png
    ├── heatmap_comparison.png
    └── improvement_chart.png
```

## Common Use Cases

### Quick Testing (Development)

```bash
./run_evaluation.sh --quick --llm
# 10 examples, LLM reranker, ~2-5 minutes
```

### Standard Evaluation

```bash
./run_evaluation.sh
# 50 examples, cross-encoder reranker, ~10-15 minutes
```

### Full Evaluation (Production)

```bash
./run_evaluation.sh --full --llm --dspy
# 100 examples, LLM reranker + DSPy, ~30-60 minutes
```

### Regenerate Graphs Only

```bash
./run_evaluation.sh --graphs
# Uses existing results, generates new visualizations
```

### Baseline Testing Only

```bash
./run_evaluation.sh --baseline -n 30
# Only baseline reranking, 30 examples
```

## Troubleshooting

### Issue: "Model not found"

**Solution:** Check `LOCAL_LLM` in `.env` is set to a valid Hugging Face model ID:
```bash
LOCAL_LLM="google/gemma-3-4b-it"
```

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size or use CPU:
- The model will automatically fall back to CPU if CUDA unavailable
- Use float32 instead of bfloat16 on CPU

### Issue: "No dataset found"

**Solution:** Set `DATASET_DIR` in `.env`:
```bash
DATASET_DIR="/path/to/scholar_copilot_eval_data_1k.json"
```

### Issue: Slow evaluation

**Solutions:**
- Use `--quick` for testing
- Enable caching (default)
- Use cross-encoder instead of LLM reranker (remove `--llm`)
- Reduce number of examples with `-n`

## Performance Expectations

| Configuration | Examples | Time | Notes |
|--------------|----------|------|-------|
| Quick test | 10 | 2-5 min | Good for testing |
| Standard | 50 | 10-15 min | Balanced evaluation |
| Full | 100 | 30-60 min | Comprehensive results |
| Full + LLM + DSPy | 100 | 60-90 min | Complete system test |

*Times are approximate and depend on hardware (GPU vs CPU)*

## Metrics Explained

- **Recall@K**: Proportion of relevant documents found in top-K results
- **MRR**: Mean Reciprocal Rank - measures how high the first relevant result appears
- **Improvement**: Percentage change from baseline to reranked performance

## Contributing

When modifying evaluation scripts:

1. Update this README
2. Test with `--quick` first
3. Ensure backward compatibility
4. Document new metrics or visualizations
5. Update `run_evaluation.sh` if adding new scripts
