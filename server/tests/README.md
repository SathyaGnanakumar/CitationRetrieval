# Retrieval System Testing Suite

Comprehensive test suite for E5 and SPECTER retrievers with single and batch query processing capabilities.

## Overview

This test suite validates the functionality and performance of the E5 and SPECTER dense retrieval systems. It tests both single query processing and batch query processing (for GPU efficiency) using the actual ScholarCopilot dataset.

### Key Features

- **Single Query Testing**: Validates individual query processing for both E5 and SPECTER
- **Batch Query Testing**: Tests batch processing of multiple queries simultaneously (10+ queries)
- **Consistency Verification**: Ensures single and batch queries produce identical results
- **CLI Performance Testing**: Configurable models and parameters for performance benchmarking
- **Real Dataset**: Uses actual ScholarCopilot dataset (not synthetic data)

## Prerequisites

### Required Dependencies

- Python 3.10+
- PyTorch (with CUDA support recommended for GPU testing)
- sentence-transformers
- transformers
- All dependencies from `pyproject.toml`

### Dataset Requirements

You need the ScholarCopilot evaluation dataset in JSON format. The dataset should be located at:

- Default location: `../datasets/scholar_copilot_eval_data_1k.json` (relative to server root)
- Or set via environment variable: `DATASET_DIR`
- Or specify via CLI: `--dataset /path/to/dataset.json`

The dataset format should match the ScholarCopilot structure with `bib_info` containing citation entries.

## Quick Start

### 1. Set Up Dataset Path

```bash
# Option 1: Set environment variable
export DATASET_DIR="/path/to/scholar_copilot_eval_data_1k.json"

# Option 2: Use default location (../datasets/scholar_copilot_eval_data_1k.json)
# Option 3: Specify via CLI (see below)
```

### 2. Run All Tests

```bash
# From the server directory
python tests/test_retrievers_batch.py
```

This will:

- Load the dataset and build the citation corpus
- Build E5 and SPECTER resources (embeddings)
- Run all test types: single, batch, consistency, and CLI performance tests

### 3. Run Specific Test Types

```bash
# Only single query tests
python tests/test_retrievers_batch.py --test-type single

# Only batch query tests (10 queries)
python tests/test_retrievers_batch.py --test-type batch

# Only CLI performance test
python tests/test_retrievers_batch.py --test-type cli

# Only consistency verification
python tests/test_retrievers_batch.py --test-type consistency
```

## Test Types Explained

### 1. Single Query Tests

Tests individual query processing for both E5 and SPECTER retrievers.

**What it does:**

- Processes one query at a time
- Retrieves top-k results
- Validates result structure, scores, and sorting

**Example:**

```bash
python tests/test_retrievers_batch.py --test-type single
```

**Output:**

```
Testing E5 Single Query
======================================================================
✓ Query: transformer architecture for natural language processing
✓ Retrieved 3 results
  1. Attention Is All You Need... (score: 0.8234)
  2. BERT: Pre-training of Deep Bidirectional Transformers... (score: 0.7891)
  3. GPT-3: Language Models are Few-Shot Learners... (score: 0.7654)
```

### 2. Batch Query Tests

Tests batch processing of 10 queries simultaneously for GPU efficiency.

**What it does:**

- Processes 10 queries in a single batch
- Retrieves top-k results for each query
- Validates per-query results structure

**Example:**

```bash
python tests/test_retrievers_batch.py --test-type batch
```

**Output:**

```
Testing E5 Batch Query (10 queries)
======================================================================
✓ Processed 10 queries in batch

  Query 1: transformer architecture
    Top result: Attention Is All You Need... (score: 0.8234)

  Query 2: neural network models
    Top result: BERT: Pre-training of Deep Bidirectional... (score: 0.8123)
  ...
```

### 3. Consistency Tests

Verifies that single query and batch query (with 1 query) produce identical results.

**What it does:**

- Runs the same query through both single and batch methods
- Compares results to ensure they match exactly
- Validates correctness of batch implementation

**Example:**

```bash
python tests/test_retrievers_batch.py --test-type consistency
```

**Output:**

```
Testing Consistency: Single Query vs Batch Query (1 query)
======================================================================
✓ E5: Single query and batch query (1 query) produce identical results
✓ SPECTER: Single query and batch query (1 query) produce identical results
```

### 4. CLI Performance Tests

Configurable performance testing with timing measurements.

**What it does:**

- Tests both single and batch processing
- Measures execution time
- Calculates average time per query
- Allows custom model selection

**Example:**

```bash
python tests/test_retrievers_batch.py --test-type cli --num-queries 20
```

**Output:**

```
CLI Test: Configurable Models
======================================================================
E5 Model: intfloat/e5-base-v2
SPECTER Model: allenai/specter2_base
Number of queries: 20
Top-k: 3
======================================================================

[E5 Retriever]
----------------------------------------------------------------------
Testing single query...
  ✓ Single query: 3 results
Testing batch query (20 queries)...
  ✓ Batch query: 20 query results in 4.23s
  ✓ Average time per query: 0.212s

[SPECTER Retriever]
----------------------------------------------------------------------
Testing single query...
  ✓ Single query: 3 results
Testing batch query (20 queries)...
  ✓ Batch query: 20 query results in 5.67s
  ✓ Average time per query: 0.284s
```

## CLI Options Reference

### Model Selection

```bash
# Use different E5 models
python tests/test_retrievers_batch.py \
  --e5-model intfloat/e5-large-v2 \
  --test-type cli

# Use different SPECTER models
python tests/test_retrievers_batch.py \
  --specter-model allenai/specter2 \
  --test-type cli

# Both custom models
python tests/test_retrievers_batch.py \
  --e5-model intfloat/e5-large-v2 \
  --specter-model allenai/specter2 \
  --test-type cli
```

### Query Configuration

```bash
# Test with more queries
python tests/test_retrievers_batch.py \
  --num-queries 50 \
  --test-type cli

# Retrieve more results per query
python tests/test_retrievers_batch.py \
  --k 10 \
  --test-type cli

# Both
python tests/test_retrievers_batch.py \
  --num-queries 50 \
  --k 10 \
  --test-type cli
```

### Dataset Path

```bash
# Specify custom dataset path
python tests/test_retrievers_batch.py \
  --dataset /custom/path/to/dataset.json \
  --test-type all
```

## Complete Examples

### Example 1: Full Test Suite with Custom Models

```bash
python tests/test_retrievers_batch.py \
  --e5-model intfloat/e5-large-v2 \
  --specter-model allenai/specter2 \
  --dataset /path/to/dataset.json \
  --test-type all
```

### Example 2: Performance Benchmarking

```bash
# Test with 100 queries to measure batch efficiency
python tests/test_retrievers_batch.py \
  --test-type cli \
  --num-queries 100 \
  --k 5
```

### Example 3: Quick Validation

```bash
# Fast test with smaller models
python tests/test_retrievers_batch.py \
  --e5-model intfloat/e5-base-v2 \
  --specter-model allenai/specter2_base \
  --test-type consistency
```

### Example 4: Development Testing

```bash
# Test only batch functionality during development
python tests/test_retrievers_batch.py \
  --test-type batch \
  --k 3
```

## Understanding Test Output

### Test Structure

Each test follows this pattern:

1. **Setup Phase**: Loads dataset, builds corpus, initializes resources
2. **Execution Phase**: Runs queries through retrievers
3. **Validation Phase**: Checks results structure and correctness
4. **Reporting Phase**: Prints results and metrics

### Result Validation

All tests validate:

- ✅ Result count matches expected `k`
- ✅ All results have required fields: `id`, `title`, `score`, `source`
- ✅ Scores are floats
- ✅ Results are sorted by score (descending)
- ✅ Source field matches retriever type (`e5` or `specter`)

### Performance Metrics

CLI tests report:

- **Total batch time**: Time to process all queries in batch
- **Average time per query**: `total_time / num_queries`
- **Single query time**: Time for one query (for comparison)

## Performance Considerations

### GPU vs CPU

- **GPU (CUDA)**: Significantly faster for batch processing
  - Batch processing shows 5-10x speedup on GPU
  - Single queries also benefit from GPU acceleration
- **CPU**: Slower but works without GPU
  - Batch processing still faster than sequential single queries
  - Good for development/testing without GPU

### Batch Size Impact

- **Small batches (1-10 queries)**: Minimal overhead, good for testing
- **Medium batches (10-50 queries)**: Optimal GPU utilization
- **Large batches (50+ queries)**: Maximum efficiency, may hit memory limits

### Memory Usage

- **E5**: Lower memory footprint, faster encoding
- **SPECTER**: Higher memory usage, slower but more accurate for academic papers
- **Corpus size**: Larger corpus = more memory for embeddings

### Caching

The test suite uses module-level caching:

- **Corpus**: Built once, reused across all tests
- **Resources**: Built once per model, reused across test functions
- **First run**: Slower (builds everything)
- **Subsequent runs**: Faster (uses cached resources)

## Troubleshooting

### Dataset Not Found

**Error:**

```
FileNotFoundError: Dataset not found at ...
```

**Solutions:**

```bash
# Option 1: Set DATASET_DIR environment variable
export DATASET_DIR="/path/to/dataset.json"

# Option 2: Use --dataset flag
python tests/test_retrievers_batch.py --dataset /path/to/dataset.json

# Option 3: Place dataset in default location
# ../datasets/scholar_copilot_eval_data_1k.json (relative to server/)
```

### Out of Memory (OOM)

**Symptoms:**

- CUDA out of memory errors
- Process killed during resource building

**Solutions:**

```bash
# Use smaller models
python tests/test_retrievers_batch.py \
  --e5-model intfloat/e5-base-v2 \
  --specter-model allenai/specter2_base

# Reduce batch size in resource building (edit test file)
# Or use CPU instead of GPU
```

### Model Download Issues

**Error:**

```
OSError: Can't load tokenizer/model for 'model-name'
```

**Solutions:**

- Check internet connection (models download from HuggingFace)
- Verify model name is correct
- Check HuggingFace Hub access

### Import Errors

**Error:**

```
ModuleNotFoundError: No module named 'src'
```

**Solutions:**

```bash
# Run from server directory
cd /path/to/server
python tests/test_retrievers_batch.py

# Or ensure PYTHONPATH includes server directory
export PYTHONPATH="${PYTHONPATH}:/path/to/server"
```

## Running with pytest

The tests can also be run with pytest:

```bash
# Run all tests
pytest tests/test_retrievers_batch.py -v

# Run specific test
pytest tests/test_retrievers_batch.py::test_e5_single_query -v

# Run with output
pytest tests/test_retrievers_batch.py -v -s
```

**Note:** When using pytest, CLI arguments won't work. Use environment variables or modify the test file directly for configuration.

## Test Architecture

### Module-Level Caching

```python
# Corpus built once
_corpus = build_citation_corpus(load_dataset(path))

# Resources built once per model
_e5_resources = build_e5_resources(_corpus, model_name)
_specter_resources = build_specter_resources(_corpus, model_name)
```

### Test Functions

- `test_e5_single_query()`: E5 single query test
- `test_e5_batch_query()`: E5 batch query test (10 queries)
- `test_specter_single_query()`: SPECTER single query test
- `test_specter_batch_query()`: SPECTER batch query test (10 queries)
- `test_consistency_single_vs_batch()`: Consistency verification
- `run_cli_test()`: Configurable CLI performance test

### Helper Functions

- `get_dataset_path()`: Resolves dataset path from env/default
- `build_corpus_once()`: Builds and caches corpus
- `get_e5_resources()`: Builds and caches E5 resources
- `get_specter_resources()`: Builds and caches SPECTER resources

## Best Practices

1. **First Run**: Let it complete fully to build all resources
2. **Development**: Use `--test-type single` for faster iteration
3. **Performance Testing**: Use `--test-type cli` with varying `--num-queries`
4. **Model Comparison**: Run CLI tests with different models to compare
5. **GPU Testing**: Ensure CUDA is available for meaningful performance metrics

## Expected Test Duration

Approximate times (on GPU):

- **First run** (building resources): 5-15 minutes

  - Corpus building: 1-2 minutes
  - E5 embeddings: 2-5 minutes
  - SPECTER embeddings: 2-8 minutes

- **Subsequent runs** (using cache): 30 seconds - 2 minutes
  - Single query tests: 5-10 seconds
  - Batch query tests: 10-20 seconds
  - CLI tests: 10-30 seconds (depends on num-queries)

## Integration with CI/CD

For continuous integration, you might want to:

```bash
# Quick validation in CI
python tests/test_retrievers_batch.py \
  --test-type consistency \
  --e5-model intfloat/e5-base-v2 \
  --specter-model allenai/specter2_base
```

This runs fast consistency checks without full performance testing.

## Additional Resources

- **Dataset Format**: See `src/corpus/scholarcopilot.py` for dataset structure
- **Retriever Implementation**: See `src/agents/retrievers/e5_agent.py` and `specter_agent.py`
- **Resource Building**: See `src/resources/builders.py`

## Support

For issues or questions:

1. Check troubleshooting section above
2. Verify dataset format matches expected structure
3. Ensure all dependencies are installed
4. Check GPU availability if testing performance

---

**Last Updated**: 2025
**Test Suite Version**: 1.0
