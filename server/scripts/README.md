# Evolution Scripts

Self-evolution system execution scripts.

## Scripts Overview

### 🧪 optimize_once.py
**Purpose**: One-time optimization for testing and development

**Usage**:
```bash
uv run python scripts/optimize_once.py
```

**What it does**:
1. Temporarily enables evolution (overrides flag)
2. Runs baseline evaluation (100 queries)
3. Optimizes with GEPA + GPT-5 Mini
4. Runs post-optimization evaluation (100 queries)
5. Compares and reports improvement

**When to use**:
- First time testing the evolution system
- Validating optimization works
- Comparing baseline vs optimized performance
- Development and debugging

**Output**:
```
Baseline score:   0.6230
Optimized score:  0.6810
Improvement:      +0.0580 (+9.3%)
```

---

### 🔄 run_evolution.py
**Purpose**: Continuous evolution loop for production

**Usage**:
```bash
# Requires ENABLE_DSPY_EVOLUTION=true in .env
uv run python scripts/run_evolution.py
```

**What it does**:
1. Checks evolution flag (exits if disabled)
2. Loads dataset and builds resources
3. Runs initial baseline evaluation
4. Starts continuous loop:
   - Evaluates every 100 queries
   - Optimizes every 1000 queries
   - Auto-deploys improvements
5. Runs until stopped (Ctrl+C)

**When to use**:
- Production continuous learning
- Long-running optimization
- Automated system improvement

**Output**:
```
[10:00:00] DSPy Evolution ENABLED - Starting evolution loop...
[10:00:05] Baseline evaluation: 100 queries, avg_score=0.623
[10:15:30] Processed 1000 queries, triggering optimization...
[10:19:42] GEPA: Optimization complete (4min 7s, best_score=0.681)
[10:20:15] Improvement: +0.058 (9.3%), DEPLOYING v1
[10:20:16] ✓ Module picker updated to v1
[10:20:16] Continuing evolution loop...
```

---

### 📊 analyze_evolution.py
**Purpose**: Generate analysis reports and visualizations

**Usage**:
```bash
uv run python scripts/analyze_evolution.py
```

**What it does**:
1. Loads evaluation data from `data/evaluations/`
2. Loads module versions from `data/module_versions/`
3. Generates plots:
   - Score progression over time
   - Score distribution histogram
4. Creates text summary report

**When to use**:
- After optimization cycles complete
- Periodic performance review
- Understanding system behavior
- Preparing reports

**Output files**:
- `reports/score_progression.png`
- `reports/score_distribution.png`
- `reports/evolution_report.txt`

**Example report**:
```
======================================================================
EVOLUTION SYSTEM REPORT
======================================================================

EVALUATION STATISTICS
----------------------------------------------------------------------
Total evaluations: 1000
Average score: 0.6542
Min score: 0.1234
Max score: 0.9876
Std deviation: 0.1456

Score Distribution:
  Failures (<0.3):   120 (12.0%)
  Moderate (0.3-0.7): 580 (58.0%)
  Successes (≥0.7):  300 (30.0%)

MODULE VERSIONS
----------------------------------------------------------------------

picker:
  Total versions: 3
  Latest: v2 (score: 0.6810)
  Best: v2 (score: 0.6810)
  Version history:
    v0: 0.6230 (2025-01-15T10:00:00)
    v1: 0.6580 (2025-01-15T11:30:00)
    v2: 0.6810 (2025-01-15T13:00:00)
```

---

## Configuration

All scripts read from `.env`:

```bash
# Feature flag
ENABLE_DSPY_EVOLUTION=true

# Intervals
EVOLUTION_EVAL_INTERVAL=100        # Evaluate every N queries
EVOLUTION_OPTIMIZE_INTERVAL=1000   # Optimize every N queries

# Models
TEACHER_MODEL=openai/gpt-5-mini    # For optimization
STUDENT_MODEL=ollama_chat/gemma:2b # For inference

# API Key (required)
OPENAI_API_KEY=your_key_here
```

## Workflow

```
1. Test First
   └─> optimize_once.py (validates system works)

2. Enable in Production
   └─> run_evolution.py (continuous learning)

3. Monitor Performance
   └─> analyze_evolution.py (reports & plots)

4. Iterate
   └─> Adjust settings based on results
```

## Common Use Cases

### Scenario 1: Initial Testing
```bash
# First time - test the system
uv run python scripts/optimize_once.py

# Review results
cat data/evaluations/summary.json | jq '.[-10:]'
```

### Scenario 2: Production Deployment
```bash
# Enable in .env
echo "ENABLE_DSPY_EVOLUTION=true" >> .env

# Start continuous evolution
nohup uv run python scripts/run_evolution.py > evolution.out 2>&1 &

# Monitor progress
tail -f logs/evolution_*.log
```

### Scenario 3: Performance Analysis
```bash
# After 10,000 queries processed
uv run python scripts/analyze_evolution.py

# Review improvements
cat reports/evolution_report.txt
```

### Scenario 4: Rollback
```python
# If a version performs worse
from src.agents.self_evolve.version_tracker import VersionTracker

tracker = VersionTracker("picker")
tracker.rollback_to(version=1)  # Rollback to v1
```

## Troubleshooting

### Script fails with "Evolution not enabled"
```bash
# Check .env file
grep ENABLE_DSPY_EVOLUTION .env

# Should show:
# ENABLE_DSPY_EVOLUTION=true
```

### Script fails with "OPENAI_API_KEY not set"
```bash
# Add to .env
echo "OPENAI_API_KEY=sk-your-key" >> .env
```

### Ollama connection error
```bash
# Start Ollama server
ollama serve

# Pull Gemma model
ollama pull gemma:2b

# Verify running
curl http://localhost:11434/api/tags
```

### Not enough training data
```bash
# Need at least 10 examples with score >= 0.5
# Run more queries to collect data

# Check evaluation count
jq 'length' data/evaluations/summary.json
```

## Performance Tips

### Faster Optimization
```bash
GEPA_AUTO_BUDGET=light           # ~2 min instead of 4 min
EVOLUTION_OPTIMIZE_INTERVAL=500  # Optimize more frequently
```

### Better Quality
```bash
GEPA_AUTO_BUDGET=heavy                # ~10 min but thorough
EVOLUTION_MIN_TRAINING_SCORE=0.7      # Only use high-quality examples
EVOLUTION_OPTIMIZE_INTERVAL=2000      # More data before optimization
```

### Cost Optimization
```bash
EVOLUTION_OPTIMIZE_INTERVAL=2000   # Optimize less frequently
GEPA_MAX_METRIC_CALLS=25          # Reduce GPT-5 Mini calls
```

## Next Steps

1. ✅ Implementation complete
2. ⏭️ Run `optimize_once.py` to test
3. ⏭️ Enable `ENABLE_DSPY_EVOLUTION=true`
4. ⏭️ Start `run_evolution.py` for production
5. ⏭️ Monitor with `analyze_evolution.py`

## Documentation

- **Complete Guide**: [SELF_EVOLUTION_README.md](../SELF_EVOLUTION_README.md)
- **Quick Start**: [EVOLUTION_QUICKSTART.md](../EVOLUTION_QUICKSTART.md)
- **Config Example**: [EVOLUTION_CONFIG.example](../EVOLUTION_CONFIG.example)
- **Summary**: [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md)

---

**Ready to evolve!** 🚀
