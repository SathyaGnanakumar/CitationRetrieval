# Self-Evolving DSPy Agent System

A complete self-evolving agent system that continuously learns from ScholarCopilot dataset evaluations, optimizes DSPy modules using GPT-5 Mini as teacher, and deploys improved versions with monitoring and rollback capabilities.

## Overview

The self-evolution system enables your retrieval pipeline to automatically improve over time by:
- **Collecting evaluation data** from every query execution
- **Optimizing DSPy modules** using GEPA (Genetic-Pareto) optimizer with GPT-5 Mini
- **Deploying improved versions** when performance increases
- **Rolling back** when performance degrades

## Key Features

- ✅ **Feature Flag Control**: Disabled by default (`ENABLE_DSPY_EVOLUTION=false`)
- ✅ **GEPA Optimizer**: Uses reflective prompt evolution for better generalization
- ✅ **Local Inference**: Uses Gemma 2B via Ollama (free, fast)
- ✅ **Smart Teacher**: Uses GPT-5 Mini only during optimization (not inference)
- ✅ **Version Tracking**: Complete module version history with rollback
- ✅ **Monitoring**: Structured logging, CSV export, visualizations
- ✅ **Zero Overhead**: No performance impact when disabled

## Quick Start

### 1. Installation

```bash
# Install GEPA dependency
uv sync
# or
pip install gepa
```

### 2. Enable Evolution

Add to `.env`:

```bash
# Enable self-evolution (default: false)
ENABLE_DSPY_EVOLUTION=true

# Evolution configuration
EVOLUTION_EVAL_INTERVAL=100       # Evaluate every 100 queries
EVOLUTION_OPTIMIZE_INTERVAL=1000  # Optimize every 1000 queries
EVOLUTION_MIN_IMPROVEMENT=0.05    # Deploy if improvement >= 5%

# Models
TEACHER_MODEL=openai/gpt-5-mini   # For optimization only
STUDENT_MODEL=ollama_chat/gemma:2b # For inference

# Required: OpenAI API key for teacher model
OPENAI_API_KEY=your_key_here
```

### 3. Run One-Time Optimization (Testing)

```bash
# Test the system with one optimization cycle
python scripts/optimize_once.py
```

This will:
1. Run baseline evaluation (100 queries)
2. Optimize with GEPA using GPT-5 Mini
3. Run post-optimization evaluation
4. Compare and report results

### 4. Run Continuous Evolution (Production)

```bash
# Start continuous learning loop
python scripts/run_evolution.py
```

This will:
1. Run initial baseline evaluation
2. Process queries in batches (eval_interval)
3. Trigger optimization every N queries (optimize_interval)
4. Auto-deploy improvements
5. Continue until stopped (Ctrl+C)

## Architecture

```
Query Input
    ↓
[ENABLE_DSPY_EVOLUTION check]
    ↓
Evaluation Store (collect metrics)
    ↓
[Every 1000 queries]
    ↓
GEPA Optimizer
    ↓
Reflective Analysis (GPT-5 Mini)
    ↓
Propose New Prompts
    ↓
Optimized Module
    ↓
Compare Performance
    ↓
Deploy if Better / Rollback if Worse
    ↓
Version Tracker
```

## Components

### Core Infrastructure

**Evaluation Store** (`src/evaluation/eval_store.py`):
- Stores query evaluations with metrics and scores
- Provides filtering (successes, failures)
- Converts to DSPy training format
- File-based persistence

**DSPy Metrics** (`src/evaluation/dspy_metrics.py`):
- `citation_retrieval_metric()`: Weighted combination of Recall@5, Recall@10, MRR
- `query_reformulation_metric()`: Measures query diversity
- Used by GEPA optimizer to evaluate candidates

**Version Tracker** (`src/agents/self_evolve/version_tracker.py`):
- Tracks module versions with scores and timestamps
- Supports rollback to any previous version
- Pickles modules for persistence
- Identifies best-performing version

### DSPy Modules

**DSPy Reformulator** (`src/agents/formulators/dspy_reformulator.py`):
- Optimizable query reformulation module
- Uses Chain-of-Thought reasoning
- Compatible with existing workflow

**Enhanced DSPy Picker** (`src/agents/formulators/dspy_picker.py`):
- Loads optimized modules from version tracker
- Falls back to default if no optimized version
- Stores predictions for evaluation

### GEPA Optimizer

**Optimizer Factory** (`src/agents/self_evolve/optimizers.py`):
- `GEPAAdapter`: Interfaces with your retrieval workflow
- `get_gepa_optimizer()`: Configures GEPA with GPT-5 Mini
- `optimize_with_gepa()`: Runs reflective optimization

GEPA provides better generalization than MIPROv2 by:
- Using explicit train/validation split (90/10)
- Reflecting on failures with natural language feedback
- Iteratively proposing improved prompts
- Selecting based on validation performance

### Evolution Engine

**SelfEvolvingRetrievalSystem** (`src/agents/self_evolve/evolution_engine.py`):
- `evaluate_batch()`: Runs queries and stores metrics
- `optimize_module()`: Runs GEPA optimization
- `continuous_evolution_loop()`: Main loop with auto-deployment

### Workflow Integration

**RetrievalWorkflow** (enhanced `src/workflow.py`):
- `enable_evolution` parameter (default: False)
- `update_dspy_module()`: Hot-swap modules
- `get_module_versions()`: Check current versions
- `is_evolution_enabled()`: Check flag status

**State Model** (enhanced `src/models/state.py`):
- Added evolution-specific fields
- Tracks module versions used
- Stores evaluation metadata

### Execution Scripts

**optimize_once.py** (`scripts/optimize_once.py`):
- One-time optimization for testing
- Compares baseline vs optimized
- Temporarily enables evolution flag

**run_evolution.py** (`scripts/run_evolution.py`):
- Continuous evolution loop
- Requires evolution flag enabled
- Production-ready monitoring

### Monitoring & Analysis

**EvolutionLogger** (`src/agents/self_evolve/monitoring.py`):
- Structured logging to file and console
- CSV export for analysis
- Tracks evaluations, optimizations, deployments, rollbacks

**analyze_evolution.py** (`scripts/analyze_evolution.py`):
- Generates score progression plots
- Creates score distribution histograms
- Produces summary reports

## Usage Examples

### Basic Usage (Evolution Disabled)

```python
from src.workflow import RetrievalWorkflow

# Evolution disabled by default
workflow = RetrievalWorkflow()
result = workflow.run(initial_state)
```

### With Evolution Enabled

```python
import os
os.environ["ENABLE_DSPY_EVOLUTION"] = "true"

from src.workflow import RetrievalWorkflow
from src.agents.self_evolve.evolution_engine import SelfEvolvingRetrievalSystem

# Create workflow with evolution
workflow = RetrievalWorkflow(enable_evolution=True)

# Create self-evolving system
evolving_system = SelfEvolvingRetrievalSystem(
    workflow=workflow,
    resources=resources,
    dataset_path="path/to/dataset.json"
)

# Run evaluation
score = evolving_system.evaluate_batch(papers[:100])

# Run optimization
optimized = evolving_system.optimize_module(
    module_name="picker",
    min_score=0.5,
    auto_budget="medium"
)
```

### Check Module Versions

```python
versions = workflow.get_module_versions()
# {
#   "picker": {
#     "current_version": 2,
#     "best_version": 2,
#     "best_score": 0.785
#   }
# }
```

### Rollback a Module

```python
from src.agents.self_evolve.version_tracker import VersionTracker

tracker = VersionTracker("picker")
tracker.rollback_to(version=1)  # Roll back to version 1
```

## Storage Structure

```
data/
├── evaluations/
│   ├── batch_20250115_100000.json
│   ├── batch_20250115_100001.json
│   └── summary.json
├── module_versions/
│   ├── picker/
│   │   ├── v0.pkl
│   │   ├── v1.pkl
│   │   ├── v2.pkl
│   │   └── metadata.json
│   └── reformulator/
│       ├── v0.pkl
│       └── metadata.json
└── logs/
    └── evolution_20250115_100000.log
```

## Monitoring Output

### When Disabled (Default)
```
[2025-01-15 10:00:00] Retrieval workflow initialized (evolution disabled)
[2025-01-15 10:00:01] Processing query 1/100...
[2025-01-15 10:00:02] Query complete: 10 papers retrieved
```

### When Enabled
```
[2025-01-15 10:00:00] 🔄 DSPy Evolution ENABLED - Starting evolution loop...
[2025-01-15 10:00:05] Baseline evaluation: 100 queries, avg_score=0.623
[2025-01-15 10:15:30] Processed 1000 queries, triggering optimization...
[2025-01-15 10:15:35] GEPA: Training on 847 examples (score >= 0.5)
[2025-01-15 10:15:36] GEPA: Split into trainset=762, valset=85
[2025-01-15 10:15:37] GEPA: Starting reflective optimization (auto='medium')
[2025-01-15 10:17:15] GEPA: Reflection iteration 1/5 - score=0.641
[2025-01-15 10:18:22] GEPA: Reflection iteration 2/5 - score=0.668
[2025-01-15 10:19:08] GEPA: Reflection iteration 3/5 - score=0.681
[2025-01-15 10:19:42] GEPA: Optimization complete (4min 7s, best_score=0.681)
[2025-01-15 10:20:15] Validation: new_score=0.681, current_score=0.623
[2025-01-15 10:20:15] Improvement: +0.058 (9.3%), DEPLOYING v1
[2025-01-15 10:20:16] ✓ Module picker updated to v1
[2025-01-15 10:20:16] Continuing evolution loop...
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_DSPY_EVOLUTION` | `false` | Enable/disable self-evolution |
| `EVOLUTION_EVAL_INTERVAL` | `100` | Evaluate every N queries |
| `EVOLUTION_OPTIMIZE_INTERVAL` | `1000` | Optimize every N queries |
| `EVOLUTION_MIN_IMPROVEMENT` | `0.05` | Min improvement to deploy (5%) |
| `TEACHER_MODEL` | `openai/gpt-5-mini` | Teacher model for optimization |
| `STUDENT_MODEL` | `ollama_chat/gemma:2b` | Student model for inference |
| `EVOLUTION_DATA_DIR` | `./data` | Data directory |

### GEPA Optimizer Settings

- **auto_budget**: `'light'`, `'medium'`, `'heavy'`
  - Light: ~20 metric calls, fast, basic optimization
  - Medium: ~50 metric calls, balanced (default)
  - Heavy: ~100+ metric calls, thorough but slow

- **max_metric_calls**: Maximum number of evaluations (default: 50)
- **reflection_lm**: Model for reflection (GPT-5 Mini)
- **num_threads**: Parallel threads (default: 4)

## Testing

Run the test suite:

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/test_evolution_engine.py

# Integration tests only
pytest tests/test_self_evolution_integration.py -m integration
```

## Analysis & Reporting

Generate analysis reports:

```bash
python scripts/analyze_evolution.py
```

This creates:
- `reports/score_progression.png`: Score over time
- `reports/score_distribution.png`: Score histogram
- `reports/evolution_report.txt`: Summary statistics

## Cost Optimization

The system is designed to minimize costs:

1. **Local Inference** (Free): All production queries use Gemma via Ollama
2. **Teacher Only for Training**: GPT-5 Mini is only called during optimization
3. **Infrequent Optimization**: Only every 1000 queries by default
4. **Efficient GEPA**: Uses validation set to prevent overfitting

**Example Cost**:
- Process 10,000 queries: $0 (all local)
- Optimize 10 times: ~$5-10 (GPT-5 Mini for training only)
- Total: ~$0.0005-0.001 per query

## Troubleshooting

### Evolution Not Starting

Check:
1. `ENABLE_DSPY_EVOLUTION=true` in `.env`
2. `OPENAI_API_KEY` is set
3. Ollama is running: `ollama serve`
4. Gemma model is available: `ollama pull gemma:2b`

### Low Optimization Scores

Try:
1. Increase `min_score` threshold (default: 0.5)
2. Collect more training data (>500 examples)
3. Use `auto='heavy'` for more thorough optimization
4. Check that ground truth labels are correct

### Module Not Loading

Check:
1. Version tracker storage directory exists
2. Pickled modules are compatible with current DSPy version
3. File permissions are correct

## References

- [GEPA Paper](https://arxiv.org/abs/2507.19457): Reflective Prompt Evolution
- [DSPy Documentation](https://dspy.ai/): DSPy framework
- [OpenAI Cookbook](https://cookbook.openai.com/examples/partners/self_evolving_agents/autonomous_agent_retraining): Self-evolving agents

## License

Same as main project.
