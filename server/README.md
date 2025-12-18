# Citation Retrieval System

A self-evolving citation retrieval system that finds relevant academic papers for citation contexts. It combines multiple retrieval methods, learns from its mistakes, and automatically improves over time.

---

## What This System Does

Given a piece of academic text (like a paper introduction), the system finds the most relevant papers to cite. It uses three different search strategies in parallel, merges their results, reranks them, and picks the best match.

**Example input:**

> "Transformer is widely used in natural language processing due to its high training efficiency..."

**Output:** Ranked list of relevant papers (with IDs, titles, and confidence scores)

---

## How It Works

### The Pipeline (5 steps)

```
Query → Reformulator → [BM25 + E5 + SPECTER] → Aggregator → Reranker → Results
```

| Step                       | What it does                                                          | File                                           |
| -------------------------- | --------------------------------------------------------------------- | ---------------------------------------------- |
| **1. Query Reformulator**  | Takes the input text, extracts keywords, generates 4 query variations | `src/agents/formulators/query_reformulator.py` |
| **2. Parallel Retrievers** | Three retrievers search simultaneously:                               |                                                |
| ↳ BM25                     | Keyword matching (fast, no ML)                                        | `src/agents/retrievers/bm25_agent.py`          |
| ↳ E5                       | Dense embeddings (semantic similarity)                                | `src/agents/retrievers/e5_agent.py`            |
| ↳ SPECTER                  | Academic paper embeddings                                             | `src/agents/retrievers/specter_agent.py`       |
| **3. Aggregator**          | Deduplicates results, fuses scores using Reciprocal Rank Fusion (RRF) | `src/agents/formulators/aggregator.py`         |
| **4. Reranker**            | Cross-encoder or LLM scores each candidate against the query          | `src/agents/formulators/reranker.py`           |
| **5. DSPy Picker**         | Final selection of best paper                                         | `src/agents/formulators/dspy_picker.py`        |

### The Self-Evolution System (optional)

When enabled (`ENABLE_DSPY_EVOLUTION=true`), the system learns from its performance using **conditional routing**:

```
                    ┌─ reformulator_default ─┐
START → route() ───┤                         ├──→ [retrievers] → aggregator → reranker
                    └─ reformulator_optimized ┘
                                                                        │
                                                                        ▼
                                                              ┌─ picker_default ─┐
                                                    route() ─┤                   ├──→ END
                                                              └─ picker_optimized ┘
```

**How it works:**

1. **Evaluate** – Runs queries and measures recall, MRR, and hit rate
2. **Store** – Saves evaluations with ground truth to `EvaluationStore`
3. **Optimize** – GEPA optimizer with GPT-5 Mini generates improved prompts
4. **Deploy** – Saves optimized module to `VersionTracker`
5. **Route** – Next query: `route_reformulator()` and `route_picker()` check VersionTracker and route to optimized nodes if available

The routing functions (`_route_reformulator`, `_route_picker` in `workflow.py`) use LangGraph's `add_conditional_edges` to dynamically select between default and optimized versions.

---

## Quick Start

### 1. Setup

```bash
cd server
cp .env.example .env
# Edit .env with your DATASET_DIR and OPENAI_API_KEY
uv sync
```

### 2. Run evaluation (no evolution)

```bash
uv run python evaluate.py
```

### 3. Run with self-evolution

```bash
# One-time optimization test
uv run python scripts/optimize_once.py --max-queries 100

# With LLM reranker and fewer queries
uv run python scripts/optimize_once.py --use-llm-reranker --max-queries 10 -q

# Continuous evolution loop
uv run python scripts/run_evolution.py
```

---

## Project Structure

```
server/
├── src/
│   ├── agents/
│   │   ├── retrievers/           # BM25, E5, SPECTER agents
│   │   ├── formulators/          # Reformulator, Aggregator, Reranker, Picker
│   │   └── self_evolve/          # Evolution engine, optimizers, version tracking
│   ├── evaluation/               # Metrics, evaluation store
│   ├── resources/                # Index builders, caching
│   └── workflow.py               # LangGraph pipeline orchestration
├── scripts/
│   ├── optimize_once.py          # Single optimization cycle
│   ├── run_evolution.py          # Continuous learning loop
│   └── analyze_evolution.py      # Generate reports/plots
├── data/
│   └── evaluations/              # Stored evaluations and checkpoints
├── tests/                        # Test suite
└── evaluate.py                   # Main evaluation script
```

---

## Key Configuration

| Variable                      | Description                   | Default                |
| ----------------------------- | ----------------------------- | ---------------------- |
| `DATASET_DIR`                 | Path to ScholarCopilot JSON   | Required               |
| `ENABLE_DSPY_EVOLUTION`       | Enable self-evolution         | `false`                |
| `OPENAI_API_KEY`              | For GPT-5 Mini teacher        | Required for evolution |
| `EVOLUTION_OPTIMIZE_INTERVAL` | Queries between optimizations | `1000`                 |
| `EVOLUTION_MIN_IMPROVEMENT`   | Min improvement to deploy     | `0.05`                 |

---

## Metrics

The system tracks:

| Metric    | What it measures                                   |
| --------- | -------------------------------------------------- |
| **R@K**   | Recall at K (fraction of relevant papers in top K) |
| **MRR**   | Mean Reciprocal Rank (1/position of first hit)     |
| **Score** | Weighted: `0.4*R@5 + 0.3*R@10 + 0.3*MRR`           |

Example results from `optimize_once.py`:

```
Metric       Baseline   Optimized       Change
----------------------------------------------
R@5            0.1200     0.1400   +0.0200 (+16.7%)
R@10           0.1350     0.1550   +0.0200 (+14.8%)
MRR            0.1300     0.1500   +0.0200 (+15.4%)
----------------------------------------------
Weighted       0.1298     0.1490   +0.0192 (+14.8%)

(Weighted = 0.4×R@5 + 0.3×R@10 + 0.3×MRR)
```

---

## CLI Flags for optimize_once.py

```bash
uv run python scripts/optimize_once.py [OPTIONS]

--max-queries N       # Number of queries per phase (default: 100)
--use-llm-reranker    # Use LLM instead of cross-encoder
--checkpoint-every N  # Save progress every N queries
--no-cache            # Force rebuild resources
-q, --quiet           # Suppress verbose logging
```

---

## How the Retrievers Work

### BM25 (Sparse)

- Classic keyword matching with TF-IDF-like scoring
- Fast, no GPU needed
- Good for exact term matches

### E5 (Dense)

- Encodes query and documents into embedding space
- Finds semantically similar content even with different words
- Uses `intfloat/e5-large-v2` model

### SPECTER (Academic)

- Trained specifically on scientific papers
- Understands academic writing style and citation patterns
- Uses `allenai/specter2_base` model

### Aggregator (Fusion)

- Reciprocal Rank Fusion: `score = Σ(1 / (k + rank_i))`
- Papers found by multiple retrievers get higher scores
- Deduplicates by paper ID

### Reranker

- Cross-encoder (`BAAI/bge-reranker-v2-m3`) or LLM
- Scores each (query, paper) pair directly
- More accurate but slower than embedding similarity

---

## How Self-Evolution Works

The system uses **LangGraph conditional edges** to dynamically route between default and optimized modules:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  WORKFLOW (with conditional routing)                                            │
│                                                                                 │
│  START ──→ _route_reformulator() ──┬──→ reformulator_default ──┐               │
│                                    └──→ reformulator_optimized ─┤               │
│                                                                 │               │
│           ┌─────────────────────────────────────────────────────┘               │
│           ▼                                                                     │
│  [BM25 + E5 + SPECTER] ──→ aggregator ──→ reranker                             │
│                                              │                                  │
│                                              ▼                                  │
│                            _route_picker() ──┬──→ picker_default ──→ END       │
│                                              └──→ picker_optimized ─→ END       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  EVOLUTION LOOP                                                                 │
│                                                                                 │
│  Results ──→ EvaluationStore ──→ (every N queries) ──→ GEPA Optimizer          │
│                                                              │                  │
│                                                              ▼                  │
│                                                   VersionTracker.add_version() │
│                                                              │                  │
│                                                              ▼                  │
│                                         [saved to data/module_versions/]        │
│                                                              │                  │
│                                              ┌───────────────┴───────────────┐  │
│                                              ▼                               ▼  │
│                               reformulator_optimized            picker_optimized│
│                               (loads from VersionTracker)   (loads from VT)     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key files:**

- `src/workflow.py` – `_route_reformulator()` and `_route_picker()` check VersionTracker
- `src/agents/self_evolve/version_tracker.py` – Stores optimized modules with scores
- `src/agents/formulators/query_reformulator.py` – `query_reformulator_optimized()` loads from VT
- `src/agents/formulators/dspy_picker.py` – `dspy_picker_optimized()` loads from VT

---

## Troubleshooting

| Issue                   | Solution                                                       |
| ----------------------- | -------------------------------------------------------------- |
| Score is 0.0            | Ground truth IDs may not match retrieved IDs (format mismatch) |
| Warnings about pickling | Non-serializable objects in state (safe to ignore)             |
| Tokenizer warning       | Normal warning from transformers, use `-q` to suppress         |
| CUDA out of memory      | Use smaller models or reduce batch size                        |
| Optimization didn't run | Need more training examples (lower `min_score` threshold)      |

---

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Format code
uv run black src/ tests/
uv run ruff check src/ tests/

# Visualize workflow
uv run python -c "from src.workflow import RetrievalWorkflow; RetrievalWorkflow().visualize_graph(save_file=True)"
```

---

## Architecture Diagram

See `src/additional/flow.mmd` for the full Mermaid diagram showing:

- Data preparation (dataset → corpus → indexes)
- Retrieval pipeline (reformulator → retrievers → aggregator → reranker → picker)
- Evaluation loop (results → metrics → store)
- Self-evolution (store → optimizer → deploy)

---

## License

[Add your license here]
