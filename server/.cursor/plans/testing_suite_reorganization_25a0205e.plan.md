---
name: Testing Suite Reorganization
overview: "Reorganize the testing suite by: (1) Moving dataset preprocessing code to the datasets/ folder, (2) Creating a unified CLI with --mode flag that consolidates evaluate.py, eval_retrieval_agents.py, retrieval_baselines.py, and test_retrievers_batch.py."
todos:
  - id: datasets-package
    content: Create datasets/ package with scholarcopilot/ subpackage, move loader.py and query_extraction.py
    status: completed
  - id: backwards-compat
    content: Update src/corpus/__init__.py to re-export from datasets/ for backwards compatibility
    status: completed
  - id: unified-cli
    content: Consolidate evaluate.py with --mode flag (pipeline, baselines, retrievers)
    status: completed
  - id: update-imports
    content: Update imports in all files that reference src.corpus
    status: completed
  - id: cleanup
    content: "Delete obsolete files: eval_retrieval_agents.py, retrieval_baselines.py, test_retrievers_batch.py, unit_test1.py"
    status: completed
---

# Testing Suite Reorganization

## Current State

The codebase has evaluation/testing scattered across multiple locations:

- `evaluate.py` (root) - Full pipeline evaluation
- `tests/eval_retrieval_agents.py` - Individual retriever evaluation with batching
- `tests/retrieval_baselines.py` - Similar baseline evaluation
- `tests/test_retrievers_batch.py` - Retriever testing with CLI

Dataset preprocessing is in `src/corpus/` but data lives in `datasets/`.

## Target Structure

```
Retrieval/
├── datasets/
│   ├── __init__.py
│   ├── scholarcopilot/
│   │   ├── __init__.py
│   │   ├── loader.py           # load_dataset, build_citation_corpus
│   │   └── query_extraction.py # extract_citation_queries, find_citation_contexts
│   ├── scholar_copilot_eval_data_1k.json
│   └── .cache/
└── server/
    ├── evaluate.py             # Unified CLI (replaces 4 files)
    ├── src/
    │   ├── corpus/             # DEPRECATED - imports redirect to datasets/
    │   │   └── __init__.py     # Re-exports from datasets for backwards compat
    │   └── ...                 # Rest unchanged
    └── tests/
        ├── conftest.py
        ├── test_workflow_inmemory.py
        ├── test_corpus_scholarcopilot.py
        ├── test_aggregator*.py
        └── ... (other pytest tests)
```

## Changes

### 1. Create datasets/ package structure

Move `src/corpus/scholarcopilot.py` and `src/corpus/query_extraction.py` to new location:

**New files:**

- [datasets/\_\_init\_\_.py](../datasets/__init__.py) - Package init
- [datasets/scholarcopilot/\_\_init\_\_.py](../datasets/scholarcopilot/__init__.py) - Re-exports
- [datasets/scholarcopilot/loader.py](../datasets/scholarcopilot/loader.py) - From `scholarcopilot.py`
- [datasets/scholarcopilot/query_extraction.py](../datasets/scholarcopilot/query_extraction.py) - From `query_extraction.py`

### 2. Update src/corpus/ for backwards compatibility

Modify [src/corpus/\_\_init\_\_.py](src/corpus/__init__.py) and keep thin wrappers that import from datasets/:

```python
# src/corpus/__init__.py
from datasets.scholarcopilot import (
    load_dataset,
    build_citation_corpus,
    iter_citations,
    clean_text,
)
from datasets.scholarcopilot.query_extraction import (
    extract_citation_queries,
    find_citation_contexts,
)
```

### 3. Create unified CLI in evaluate.py

Consolidate into single CLI with `--mode` flag:

```bash
# Full pipeline evaluation (current evaluate.py)
python evaluate.py --mode pipeline --dataset PATH --num-queries 100

# Individual retriever baselines (current eval_retrieval_agents.py + retrieval_baselines.py)
python evaluate.py --mode baselines --dataset PATH --retrievers bm25,e5,specter

# Retriever batch testing (current test_retrievers_batch.py)
python evaluate.py --mode retrievers --test-type batch --num-queries 10
```

**CLI modes:**

- `pipeline` - Full workflow evaluation with reranking (from `evaluate.py`)
- `baselines` - Individual retriever recall metrics (from `eval_retrieval_agents.py`, `retrieval_baselines.py`)
- `retrievers` - Retriever testing/benchmarking (from `test_retrievers_batch.py`)

**Common flags:**

- `--dataset` - Path to dataset
- `--num-queries` - Number of queries
- `--k` - Top-k results
- `--no-cache` - Disable caching

**Mode-specific flags:**

- `--mode pipeline`: `--bm25-only`, `--no-e5`, `--no-specter`, `--llm-reranker`
- `--mode baselines`: `--retrievers` (comma-separated: bm25,e5,specter), `--output-dir`
- `--mode retrievers`: `--test-type` (single/batch/cli/consistency), `--e5-model`, `--specter-model`

### 4. Files to delete after consolidation

- `tests/eval_retrieval_agents.py`
- `tests/retrieval_baselines.py`
- `tests/test_retrievers_batch.py`
- `tests/unit_test1.py` (broken import)
- `src/corpus/scholarcopilot.py` (moved to datasets/)
- `src/corpus/query_extraction.py` (moved to datasets/)

### 5. Update imports across codebase

Files that import from `src.corpus`:

- `evaluate.py` - Already root level, update import
- `tests/test_corpus_scholarcopilot.py` - Update import

## Files untouched (as requested)

- `src/unit_tests/` - All unit tests remain unchanged
- `tests/test_workflow_inmemory.py` - Pytest integration test
- `tests/test_aggregator*.py` - Aggregator tests
- `tests/conftest.py` - Pytest config