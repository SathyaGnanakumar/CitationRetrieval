# Self-Evolution System - Quick Start Guide

## Prerequisites

1. **Ollama installed** with Gemma model:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull Gemma 2B model
   ollama pull gemma:2b
   
   # Start Ollama server
   ollama serve
   ```

2. **OpenAI API Key** for GPT-5 Mini teacher model:
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

3. **Dependencies installed**:
   ```bash
   uv sync
   # or
   pip install -r requirements.txt
   ```

## Step-by-Step Setup

### Step 1: Configure Environment

Add to your `.env` file:

```bash
# Enable self-evolution
ENABLE_DSPY_EVOLUTION=true

# Dataset path
DATASET_DIR=path/to/scholar_copilot_eval_data_1k.json

# OpenAI API key (for GPT-5 Mini teacher)
OPENAI_API_KEY=your_openai_key_here

# Evolution settings (optional - these are defaults)
EVOLUTION_EVAL_INTERVAL=100
EVOLUTION_OPTIMIZE_INTERVAL=1000
TEACHER_MODEL=openai/gpt-5-mini
STUDENT_MODEL=ollama_chat/gemma:2b
```

### Step 2: Test with One-Time Optimization

```bash
# Run a single optimization cycle
python scripts/optimize_once.py
```

Expected output:
```
======================================================================
DSPy One-Time Optimization
======================================================================
Dataset: path/to/dataset.json

📚 Loading dataset and building resources...
✓ Loaded 1000 papers
✓ Built corpus with 15234 documents
✓ Built retrieval resources

🔧 Initializing workflow...
✓ Workflow initialized

⚡ Creating self-evolving system...
✓ Self-evolving system created

======================================================================
📊 BASELINE EVALUATION
======================================================================
Progress: 10/100, Avg Score: 0.612
Progress: 20/100, Avg Score: 0.635
...
✓ Baseline score: 0.623

======================================================================
⚡ OPTIMIZATION WITH GEPA
======================================================================
GEPA: Training on 847 examples (score >= 0.5)
GEPA: Split into trainset=762, valset=85
GEPA: Starting reflective optimization (auto='medium')
GEPA: Reflection iteration 1/5 - score=0.641
GEPA: Reflection iteration 2/5 - score=0.668
GEPA: Reflection iteration 3/5 - score=0.681
GEPA: Optimization complete (4min 7s, best_score=0.681)
✓ Optimization completed successfully

======================================================================
📊 POST-OPTIMIZATION EVALUATION
======================================================================
✓ Optimized score: 0.681
✓ Improvement: +0.058 (+9.3%)

======================================================================
SUMMARY
======================================================================
Baseline score:   0.6230
Optimized score:  0.6810
Improvement:      +0.0580
======================================================================

✅ One-time optimization complete!
```

### Step 3: Run Continuous Evolution (Production)

```bash
# Start continuous learning loop
python scripts/run_evolution.py
```

This will:
1. Evaluate every 100 queries
2. Optimize every 1000 queries
3. Auto-deploy improvements
4. Run continuously until stopped (Ctrl+C)

### Step 4: Monitor Progress

While evolution is running, you can:

**Check current status**:
```bash
# View logs
tail -f logs/evolution_*.log

# Check evaluation statistics
cat data/evaluations/summary.json | jq '.[-10:]'  # Last 10 evaluations

# Check module versions
cat data/module_versions/picker/metadata.json
```

**Generate analysis reports**:
```bash
python scripts/analyze_evolution.py
```

This creates:
- `reports/score_progression.png`
- `reports/score_distribution.png`
- `reports/evolution_report.txt`

### Step 5: Use Optimized Modules in Production

Once optimization is complete, the optimized modules are automatically used:

```python
from src.workflow import RetrievalWorkflow

# Evolution enabled - will use best optimized modules
workflow = RetrievalWorkflow(enable_evolution=True)

# Run queries - uses optimized Gemma model (no API costs!)
result = workflow.run(initial_state)
```

## Disable Evolution

To disable evolution and return to standard mode:

```bash
# In .env file
ENABLE_DSPY_EVOLUTION=false

# Or remove the variable entirely
```

The system will:
- Run in standard mode
- Use default modules
- Not collect evaluation data
- Not trigger optimizations
- Have zero overhead

## Troubleshooting

### "Ollama connection refused"
```bash
# Start Ollama server
ollama serve

# In another terminal, verify it's running
curl http://localhost:11434/api/tags
```

### "OPENAI_API_KEY not set"
```bash
# Add to .env file
echo "OPENAI_API_KEY=your_key" >> .env

# Or export temporarily
export OPENAI_API_KEY=your_key
```

### "Not enough training data"
The system needs at least 10 successful evaluations (score >= 0.5) to optimize.

Run more queries to collect data:
```bash
# The first 100-200 queries build training data
# Optimization will trigger after 1000 queries
```

### Check if evolution is working
```bash
# Check if evaluation data is being collected
ls -la data/evaluations/

# Check module versions
ls -la data/module_versions/picker/

# View logs
tail -100 logs/evolution_*.log
```

## Next Steps

1. **Monitor first optimization cycle** (~1000 queries)
2. **Analyze results** with `analyze_evolution.py`
3. **Adjust settings** if needed (eval_interval, min_improvement)
4. **Scale up** to continuous production use

## Questions?

See full documentation in `SELF_EVOLUTION_README.md`
