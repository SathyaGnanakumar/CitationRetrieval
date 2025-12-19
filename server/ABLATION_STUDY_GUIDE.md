# Ablation Study Guide - Single Retriever Testing

## What This Does

The new **Step 4a** in the notebook allows you to test individual retrievers (BM25, E5, or SPECTER) with LLM reranking, **skipping the fusion step**.

This is perfect for:
- 📊 **Ablation studies** for your paper
- 🔬 **Understanding** which retriever works best for specific queries
- 📈 **Comparing** performance before/after LLM reranking
- 🎯 **Debugging** retriever behavior

---

## How to Use

### Configuration

At the top of Step 4a:

```python
# ============================================================================
# CONFIGURATION: Choose Retriever
# ============================================================================
TEST_SINGLE_RETRIEVER = False  # Set to True to enable
RETRIEVER_TYPE = "bm25"  # Options: "bm25", "e5", "specter"
```

### Option 1: Skip Ablation Study (Default)

```python
TEST_SINGLE_RETRIEVER = False
```

**Output:**
```
⏭️  Skipping single retriever test (TEST_SINGLE_RETRIEVER = False)
   Set TEST_SINGLE_RETRIEVER = True to test individual retrievers
```

The notebook will proceed directly to Step 4 (RRF fusion).

---

### Option 2: Test BM25 Only

```python
TEST_SINGLE_RETRIEVER = True
RETRIEVER_TYPE = "bm25"
```

**Output:**
```
🔬 Testing single retriever: BM25
================================================================================

📊 Using BM25 (Sparse)
   Retrieved: 20 papers

📋 Top 5 from BM25 (before LLM reranking):
1. Lite Transformer with Long-Short Range Attention... (score: 26.0770)
2. BERT: Pre-training of Deep Bidirectional Transformers... (score: 24.5123)
3. TinyBERT: Distilling BERT for Natural Language Understanding... (score: 23.8456)
...

🤖 Applying LLM reranking to top 20 results from BM25...
🚀 Using cached LLM model: google/gemma-2-9b-it
📝 Invoking LLM...
✅ LLM response received (645 chars)
✅ Successfully parsed 20 rankings

================================================================================
🏆 FINAL RESULTS: BM25 + LLM Reranking (Top 10)
================================================================================

1. BERT: Pre-training of Deep Bidirectional Transformers...
   LLM Score: 0.980
   Original BM25 Score: 24.5123

2. Lite Transformer with Long-Short Range Attention...
   LLM Score: 0.950
   Original BM25 Score: 26.0770
...

================================================================================
📊 EVALUATION: BM25 + LLM Reranking
================================================================================
Ground truth found in top 10: 1/1
Recall@10: 100.00%

Comparison:
  BM25 alone (top 10): 0/1 = 0.00%
  BM25 + LLM (top 10): 1/1 = 100.00%
  ✅ Improvement: +1 citations found

================================================================================
💡 To test a different retriever, change RETRIEVER_TYPE above
   Options: 'bm25', 'e5', 'specter'
================================================================================
```

---

### Option 3: Test E5 Only

```python
TEST_SINGLE_RETRIEVER = True
RETRIEVER_TYPE = "e5"
```

Tests the **E5-Large** dense retriever with LLM reranking.

---

### Option 4: Test SPECTER Only

```python
TEST_SINGLE_RETRIEVER = True
RETRIEVER_TYPE = "specter"
```

Tests the **SPECTER-2** scientific document embeddings with LLM reranking.

---

## Workflow Examples

### Example 1: Compare All Three Retrievers

**Goal:** Understand which retriever performs best for a specific query.

```python
# Run 1: Test BM25
QUERY_INDEX = 0
TEST_SINGLE_RETRIEVER = True
RETRIEVER_TYPE = "bm25"
# Run notebook, note results

# Run 2: Test E5
QUERY_INDEX = 0  # Same query
TEST_SINGLE_RETRIEVER = True
RETRIEVER_TYPE = "e5"
# Run notebook, note results

# Run 3: Test SPECTER
QUERY_INDEX = 0  # Same query
TEST_SINGLE_RETRIEVER = True
RETRIEVER_TYPE = "specter"
# Run notebook, note results

# Run 4: Test Fusion (Step 4)
QUERY_INDEX = 0  # Same query
TEST_SINGLE_RETRIEVER = False
# Run notebook, compare to individual retrievers
```

**Results table you can build:**

| Method | Recall@10 | Top Result Quality |
|--------|-----------|-------------------|
| BM25 + LLM | 100% | High |
| E5 + LLM | 50% | Medium |
| SPECTER + LLM | 75% | High |
| **Fusion + LLM** | **100%** | **Very High** |

---

### Example 2: Quantify LLM Reranking Impact

**Goal:** Show how much LLM reranking helps each retriever.

The output automatically shows this:

```
Comparison:
  BM25 alone (top 10): 0/1 = 0.00%
  BM25 + LLM (top 10): 1/1 = 100.00%
  ✅ Improvement: +1 citations found
```

You can collect this for all three retrievers:

| Retriever | Before LLM | After LLM | Improvement |
|-----------|-----------|-----------|-------------|
| BM25 | 0% | 100% | +100% |
| E5 | 25% | 50% | +25% |
| SPECTER | 50% | 75% | +25% |

---

### Example 3: Query Type Analysis

**Goal:** Understand which retriever works best for different query types.

```python
# Technical query
QUERY_INDEX = 0  # e.g., "transformer architecture for NLP"
RETRIEVER_TYPE = "bm25"
# Run, note: BM25 may perform well (keyword matching)

QUERY_INDEX = 5  # Conceptual query, e.g., "semantic understanding"
RETRIEVER_TYPE = "e5"
# Run, note: E5 may perform better (semantic matching)

QUERY_INDEX = 10  # Scientific citation query
RETRIEVER_TYPE = "specter"
# Run, note: SPECTER may excel (trained on citations)
```

---

## Output Explanation

### Before LLM Reranking

```
📋 Top 5 from BM25 (before LLM reranking):
1. Lite Transformer with Long-Short Range Attention... (score: 26.0770)
2. BERT: Pre-training... (score: 24.5123)
```

- Shows **original retriever ranking**
- Scores are **retriever-specific** (BM25 scores, E5 cosine similarities, etc.)

---

### After LLM Reranking

```
🏆 FINAL RESULTS: BM25 + LLM Reranking (Top 10)

1. BERT: Pre-training...
   LLM Score: 0.980
   Original BM25 Score: 24.5123
```

- Shows **LLM-reranked** order
- **LLM Score**: Relevance score from LLM (0.0-1.0)
- **Original Score**: Where this paper was in the original retriever's ranking

---

### Evaluation Summary

```
📊 EVALUATION: BM25 + LLM Reranking
Ground truth found in top 10: 1/1
Recall@10: 100.00%

Comparison:
  BM25 alone (top 10): 0/1 = 0.00%
  BM25 + LLM (top 10): 1/1 = 100.00%
  ✅ Improvement: +1 citations found
```

- **Ground truth found**: How many correct citations in top 10
- **Recall@10**: Percentage of ground truth citations found
- **Comparison**: Shows improvement from LLM reranking
- **Improvement**: Net change in citations found

---

## Use Cases for Your Paper

### Ablation Study Table

**Table X: Impact of LLM Reranking on Individual Retrievers**

| Retriever | Method | R@5 | R@10 | R@20 | MRR |
|-----------|--------|-----|------|------|-----|
| BM25 | Baseline | 16.2% | 18.2% | 27.2% | 10.5% |
| BM25 | + LLM Reranker | **22.5%** | **28.3%** | **35.1%** | **15.2%** |
| E5-Large | Baseline | 9.2% | 14.2% | 19.4% | 7.9% |
| E5-Large | + LLM Reranker | **14.8%** | **21.6%** | **28.7%** | **12.3%** |
| SPECTER-2 | Baseline | 11.3% | 18.3% | 22.3% | 10.8% |
| SPECTER-2 | + LLM Reranker | **18.7%** | **25.9%** | **31.2%** | **14.6%** |
| **Fusion** | RRF + LLM | **23.4%** | **29.7%** | **36.5%** | **16.1%** |

*Numbers shown are examples - you can generate actual numbers using the notebook*

---

### Qualitative Analysis

**Example insights from testing:**

> "We observe that BM25 benefits most from LLM reranking (+6.3% R@5), suggesting that while BM25 effectively retrieves relevant papers through keyword matching, it struggles to rank them optimally. In contrast, SPECTER-2 shows smaller gains (+3.2% R@5), indicating its domain-specific embeddings already provide strong ranking signals. The fusion approach (RRF + LLM) achieves the best overall performance, demonstrating the complementary strengths of sparse and dense retrieval methods."

---

## Tips for Running Experiments

### 1. Use Caching for Speed

```python
# Enable caching to speed up experiments
USE_CACHE = True
ENABLE_LLM_RERANKER = True
```

After first run, switching `RETRIEVER_TYPE` takes only ~30 seconds!

---

### 2. Test Multiple Queries

```python
# Test BM25 on queries 0-9
for i in range(10):
    QUERY_INDEX = i
    RETRIEVER_TYPE = "bm25"
    # Run cell, collect metrics

# Average results across 10 queries
```

---

### 3. Export Results

```python
# Collect results programmatically
results = {
    "bm25": [],
    "e5": [],
    "specter": []
}

for retriever in ["bm25", "e5", "specter"]:
    RETRIEVER_TYPE = retriever
    TEST_SINGLE_RETRIEVER = True
    # Run cell
    # Collect metrics from output
    results[retriever].append({
        "recall_at_10": found_count / len(relevant_ids),
        "improvement": improvement
    })

# Save to JSON or CSV for analysis
import json
with open("ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Comparison: Single vs Fusion

### When to Use Single Retriever Test:

✅ **Ablation studies** - understand individual components
✅ **Debugging** - identify which retriever is failing
✅ **Query analysis** - see which retriever fits query type best
✅ **Performance optimization** - maybe one retriever is enough?

### When to Use Fusion (Step 4):

✅ **Best performance** - fusion usually outperforms individuals
✅ **Robustness** - combines strengths of all retrievers
✅ **Production** - final system should use fusion
✅ **Paper results** - main results should show fusion

---

## Summary

The single retriever test (Step 4a) is:
- **Optional** - set `TEST_SINGLE_RETRIEVER = True` to enable
- **Configurable** - choose `bm25`, `e5`, or `specter`
- **Informative** - shows before/after LLM reranking comparison
- **Fast** - uses cached resources for quick iteration

Perfect for understanding your system and writing ablation studies! 🎯
