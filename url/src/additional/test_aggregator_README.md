# Aggregator Tests

This directory contains comprehensive tests for the aggregator component that combines results from multiple retrievers.

## Test Files

### 1. `test_aggregator.py` - Unit Tests

Fast unit tests using mock data. No external dependencies required.

**Tests:**

- Basic aggregation with multiple retrievers
- RRF (Reciprocal Rank Fusion) score calculation
- Simple max-score aggregation method
- Edge cases (no results, missing fields, single retriever)
- Deduplication logic

**Run:**

```bash
# Run only unit tests
pytest tests/test_aggregator.py -v

# Run with coverage
pytest tests/test_aggregator.py -v --cov=src.agents.formulators.aggregator
```

**Time:** ~0.3 seconds

### 2. `test_aggregator_integration.py` - Integration Tests

Integration tests using the actual ScholarCopilot dataset and real models (BM25, E5, SPECTER).

**Tests:**

- Real retrievers with actual dataset
- Aggregator with real retrieval results
- RRF vs simple method comparison
- Full workflow integration (retrievers → aggregator → reranker)
- Overlapping results analysis

**Requirements:**

- Dataset at `../datasets/scholarcopilot/scholar_copilot_eval_data_1k.json` (relative to server directory)
- Or set `DATASET_DIR` environment variable to custom location
- First run will download E5 and SPECTER models (~2GB)
- Sufficient memory/GPU for embeddings

**Run:**

```bash
# Run all integration tests
pytest tests/test_aggregator_integration.py -v -s

# Run specific test
pytest tests/test_aggregator_integration.py::test_aggregator_with_real_retrievers -v -s

# Run with timing info
pytest tests/test_aggregator_integration.py -v -s --durations=10
```

**Time:** ~25 seconds (after models are cached)

## Test Markers

Tests are marked for easy filtering:

```bash
# Run only fast unit tests
pytest tests/ -v -m unit

# Run only integration tests
pytest tests/ -v -m integration

# Run all aggregator tests
pytest tests/test_aggregator*.py -v
```

## Sample Output

### Unit Test Output

```
tests/test_aggregator.py::test_aggregator_basic PASSED
tests/test_aggregator.py::test_rrf_score_calculation PASSED
...
6 passed in 0.26s
```

### Integration Test Output

```
📂 Loading dataset from ../datasets/scholarcopilot/scholar_copilot_eval_data_1k.json...
✓ Loaded 1000 papers from dataset

🔨 Building citation corpus...
✓ Built corpus with 9740 unique citations

⚙️  Building BM25 resources...
✓ BM25 resources ready

🔍 Running retrievers for query: 'transformer architecture...'
  → Running BM25... ✓ 10 results
  → Running E5... ✓ 10 results
  → Running SPECTER... ✓ 10 results

🔗 Running aggregator...
✓ Aggregator produced 19 unique papers

📊 Top 5 aggregated results:
1. Augmenting self-attention with persistent memory
   RRF: 0.0469, Sources: ['bm25', 'e5', 'specter'], Count: 3
2. BERT: Pre-training of Deep Bidirectional Transformers
   RRF: 0.0465, Sources: ['bm25', 'e5', 'specter'], Count: 3
...

📈 Avg RRF score - Multi-source: 0.0369, Single-source: 0.0149
```

## Understanding the Results

### RRF Scores

- Papers appearing in multiple retrievers get higher RRF scores
- Formula: `RRF_score = sum(1 / (k + rank_i))` across all retrievers
- Default k=60 (as per Cormack et al. 2009)

### Example:

```python
# Paper appears in 3 retrievers at ranks 1, 2, 1
# RRF = 1/(60+1) + 1/(60+2) + 1/(60+1) = 0.0328

# Paper appears in 1 retriever at rank 1
# RRF = 1/(60+1) = 0.0164
```

The multi-source paper has ~2x higher score!

### Key Metrics

- **Retriever Count**: How many retrievers returned this paper
- **Sources**: Which retrievers returned it (bm25, e5, specter)
- **RRF Score**: Combined ranking score
- **Overlap**: Papers common to multiple retrievers

## Debugging Tests

### If Integration Tests Fail:

1. **Dataset not found:**

   ```bash
   # Check dataset exists (relative to server directory)
   ls ../datasets/

   # Or set custom path
   export DATASET_DIR="/path/to/your/dataset.json"
   pytest tests/test_aggregator_integration.py -v
   ```

2. **Out of memory:**

   ```python
   # Edit test to use even smaller corpus
   small = corpus[:50]  # Instead of 100
   ```

3. **Model download fails:**

   ```bash
   # Manually download models first
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-base-v2')"
   ```

4. **GPU issues:**
   ```bash
   # Force CPU mode
   export CUDA_VISIBLE_DEVICES=""
   pytest tests/test_aggregator_integration.py -v
   ```

### Verbose Output:

```bash
# Show all print statements and progress bars
pytest tests/test_aggregator_integration.py -v -s

# Show test times
pytest tests/test_aggregator_integration.py -v --durations=0
```

## Test Configuration

Tests use pytest fixtures for resource management:

- **Fixtures are scoped to "module"** - resources built once per file
- **Small corpus (100 papers)** - fast but realistic
- **Fake reranker** - avoids API calls in workflow tests
- **Deterministic** - same results on every run

## Adding New Tests

To add a new test:

```python
def test_new_aggregator_feature(bm25_resources, e5_resources, specter_resources):
    """Test description."""

    state = {
        "bm25_results": [...],
        "e5_results": [...],
        "specter_results": [...],
        "config": {"aggregation_method": "rrf"},
    }

    result = aggregator(state)

    # Your assertions
    assert result["candidate_papers"]
```

## References

- **Reciprocal Rank Fusion**: Cormack et al. (2009) "Reciprocal rank fusion outperforms condorcet and individual rank learning methods"
- **Aggregator Implementation**: `src/agents/formulators/aggregator.py`
- **Workflow Integration**: `src/workflow.py`

## Quick Commands

```bash
# Fast: Run only unit tests
pytest tests/test_aggregator.py -v

# Full: Run all aggregator tests
pytest tests/test_aggregator*.py -v

# Debug: Run one test with full output
pytest tests/test_aggregator_integration.py::test_aggregator_with_real_retrievers -v -s

# CI: Run all tests with markers
pytest tests/ -v -m "unit or integration"
```
