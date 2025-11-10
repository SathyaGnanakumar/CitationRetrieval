# BM25 Baseline

BM25 sparse retrieval baseline for citation retrieval.

## Overview

Traditional information retrieval using the BM25 ranking algorithm. This baseline ranks papers based on lexical overlap between the citation context and paper content (title + abstract).

## Quick Start

```bash
cd baselines/bm25
python bm25ScholarCopilot.py
```

## Files

- `bm25ScholarCopilot.py` - Main implementation
- `bm25_scholarcopilot_eval_v2_results.csv` - Evaluation results (v2)
- `bm25_scholarcopilot_full_results.csv` - Full results
- `bm25_scores.txt` - Score statistics
- `bm25_score_distribution.png` - Score distribution visualization

## How It Works

1. **Preprocessing**: Tokenizes and stems paper titles/abstracts
2. **Indexing**: Creates BM25 index from candidate corpus
3. **Ranking**: Scores papers based on BM25 similarity to citation context
4. **Retrieval**: Returns top-k papers

## Configuration

Key parameters:
- **k1**: Term frequency saturation (default: 1.5)
- **b**: Document length normalization (default: 0.75)
- **Stemming**: Porter stemmer enabled
- **Stopwords**: English stopwords removed

## Results

See `evaluation/` folder for comprehensive metrics comparison with other baselines.

Quick stats:
- **Dataset**: ScholarCopilot 1K evaluation set
- **Metrics**: Recall@k, MRR, Precision@k
- **Runtime**: ~2-5 seconds per query

## Integration with Evaluation Framework

This baseline is integrated into the unified evaluation framework:

```python
from evaluation.models import BM25Model

model = BM25Model(use_stemming=True, use_stopwords=True)
```

See `evaluation/README.md` for details on running comprehensive evaluations.
