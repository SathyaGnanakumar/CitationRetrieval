# Dense Retrieval Baseline

Dense embedding-based retrieval using SPECTER2 and E5-Large models.

## Overview

Neural embedding-based citation retrieval using pre-trained transformer models. Papers and citation contexts are encoded into dense vectors, and similarity is computed in the embedding space.

## Models

### SPECTER2
- **Source**: Allen AI
- **Model ID**: `allenai/specter2`
- **Description**: Scientific paper embeddings trained on citation graphs
- **Best for**: Academic papers and citations

### E5-Large
- **Source**: Microsoft
- **Model ID**: `intfloat/e5-large-v2`
- **Description**: General-purpose text embeddings
- **Best for**: General text similarity

## Quick Start

```bash
cd baselines/dense
jupyter notebook Dense_Retrieval.ipynb
```

## Files

- `Dense_Retrieval.ipynb` - Main implementation notebook
- Results available at: [Google Drive](https://drive.google.com/drive/folders/1L1Eo1dE77bOelBOvWEy466Hhir8OSYPE?usp=sharing)

## How It Works

1. **Encoding**: Encode all papers in corpus using pre-trained model
2. **Query Encoding**: Encode citation context with same model
3. **Similarity**: Compute cosine similarity between query and all papers
4. **Ranking**: Return top-k papers by similarity score

## Integration with Evaluation Framework

These models are integrated into the unified evaluation framework:

```python
from evaluation.models import DenseRetrievalModel

# SPECTER2
specter = DenseRetrievalModel('allenai/specter2')

# E5-Large
e5 = DenseRetrievalModel('intfloat/e5-large-v2')
```

## Requirements

```bash
pip install sentence-transformers torch
```

## Performance

Dense models typically outperform BM25 on:
- Semantic similarity (concepts expressed differently)
- Paraphrasing and synonyms
- Domain-specific terminology

But may underperform on:
- Exact keyword matching
- Short queries with specific terms
- Out-of-distribution data

## Results

See `evaluation/README.md` for comprehensive metrics comparison with other baselines.

External results: [Google Drive Folder](https://drive.google.com/drive/folders/1L1Eo1dE77bOelBOvWEy466Hhir8OSYPE?usp=sharing)
