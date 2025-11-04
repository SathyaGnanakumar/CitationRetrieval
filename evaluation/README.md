# Citation Retrieval Evaluation Framework

A comprehensive evaluation harness for citation retrieval models, designed for the ScholarCopilot dataset and CiteME benchmark.

## Features

- **Unified Evaluation Interface**: Consistent API for evaluating any retrieval model
- **Comprehensive Metrics**: Recall@k, Precision@k, MRR, Exact Match, Average Rank
- **Baseline Models**: BM25, SPECTER2, E5-Large
- **Error Analysis**: Automatic failure logging and categorization
- **Performance Tracking**: Latency measurement and cost estimation
- **Model Comparison**: Side-by-side comparison of multiple models

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from evaluation import CitationEvaluator, CitationDataLoader
from evaluation.models import BM25Model, DenseRetrievalModel

# Load data
loader = CitationDataLoader("data/scholar_copilot_eval_data_1k.json")
loader.load_data()
examples = loader.extract_examples()

# Initialize models
bm25 = BM25Model()
specter2 = DenseRetrievalModel("allenai/specter2")

# Evaluate
evaluator = CitationEvaluator()
results = evaluator.evaluate_model(bm25, examples, top_k=20)

print(results['metrics'])
```

### Running Full Baseline Evaluation

```bash
# Evaluate all baselines on 1K dataset
python run_baseline_evaluation.py \
    --data_path BM25/scholar_copilot_eval_data_1k.json \
    --output_dir results/baselines \
    --models bm25 specter2 e5-large

# Evaluate only BM25 (fast)
python run_baseline_evaluation.py --models bm25

# Evaluate on subset for testing
python run_baseline_evaluation.py --max_examples 100
```

## Architecture

### Directory Structure

```
evaluation/
├── __init__.py              # Package exports
├── evaluator.py             # Main evaluation harness
├── metrics.py               # Metrics calculation
├── data_loader.py           # Data loading and preprocessing
├── models/                  # Model wrappers
│   ├── base_model.py        # Abstract base class
│   ├── bm25_model.py        # BM25 baseline
│   ├── dense_model.py       # Dense retrieval (SPECTER2, E5)
│   └── citeagent_model.py   # CiteAgent (to be implemented)
└── README.md
```

### Key Components

#### 1. CitationEvaluator

Main evaluation engine that:
- Runs models on examples
- Calculates metrics
- Tracks latency
- Logs failures
- Compares multiple models

```python
evaluator = CitationEvaluator(
    fuzzy_match_threshold=0.85,  # Title matching threshold
    track_latency=True,          # Measure query latency
    log_failures=True            # Log failure cases
)
```

#### 2. MetricsCalculator

Computes retrieval metrics:
- **Recall@k**: Fraction of queries with correct answer in top-k
- **Precision@k**: Precision considering top-k results
- **MRR**: Mean Reciprocal Rank
- **Exact Match**: Recall@1
- **Average Rank**: Mean rank of correct answers

#### 3. CitationDataLoader

Handles ScholarCopilot dataset:
- Loads JSON data
- Extracts citation contexts
- Builds reference corpus
- Creates train/val/test splits

```python
loader = CitationDataLoader("data.json")
loader.load_data()
examples = loader.extract_examples()

# Create splits
splits = loader.create_splits(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

#### 4. Model Wrappers

All models implement `BaseRetrievalModel` interface:

```python
class MyModel(BaseRetrievalModel):
    def retrieve(self, query: str, corpus: List[Dict], k: int) -> List[Dict]:
        # Your retrieval logic
        return top_k_results
```

**Built-in models:**
- `BM25Model`: Sparse lexical retrieval
- `DenseRetrievalModel`: Dense embedding retrieval (SPECTER2, E5, etc.)

## Metrics Explained

### Recall@k
Percentage of queries where the correct citation appears in top-k results.

```
Recall@10 = 0.75 means 75% of queries have correct citation in top-10
```

### MRR (Mean Reciprocal Rank)
Average of reciprocal ranks of first correct answer.

```
If correct answer is at rank 1: RR = 1.0
If correct answer is at rank 5: RR = 0.2
If not found in top-k: RR = 0.0

MRR = average of all RRs
```

### Exact Match
Recall@1 - percentage of queries where correct citation is ranked #1.

## Evaluation Outputs

Running evaluation generates:

1. **Metrics JSON**: Overall performance metrics
```json
{
  "model": "BM25",
  "metrics": {
    "recall@1": 0.3214,
    "recall@5": 0.5892,
    "recall@10": 0.6745,
    "mrr": 0.4231
  }
}
```

2. **Comparison CSV**: Side-by-side model comparison
```
Model     R@1    R@5    R@10   MRR    Latency
BM25      0.32   0.59   0.67   0.42   15ms
SPECTER2  0.45   0.72   0.81   0.55   45ms
E5-Large  0.48   0.75   0.83   0.58   38ms
```

3. **Failure Log**: Error analysis
```json
[
  {
    "query_id": "q_123",
    "true_title": "Attention Is All You Need",
    "retrieved_titles": ["...", "..."],
    "category": "not_in_top_k",
    "corpus_size": 45
  }
]
```

## Adding Custom Models

### Example: Adding CiteAgent

```python
from evaluation.models import BaseRetrievalModel

class CiteAgentModel(BaseRetrievalModel):
    def __init__(self, retriever, reranker, selector):
        super().__init__("CiteAgent")
        self.retriever = retriever
        self.reranker = reranker
        self.selector = selector

    def retrieve(self, query, corpus, k=10):
        # Stage 1: Retrieval (get top-100)
        candidates = self.retriever.retrieve(query, corpus, k=100)

        # Stage 2: Reranking (narrow to top-20)
        reranked = self.reranker.rerank(query, candidates, k=20)

        # Stage 3: LLM Selection (final top-k)
        selected = self.selector.select(query, reranked, k=k)

        return selected

    def get_config(self):
        return {
            'model_name': 'CiteAgent',
            'retriever': 'SPECTER2',
            'reranker': 'cross-encoder',
            'selector': 'gpt-4'
        }

# Use it
citeagent = CiteAgentModel(retriever, reranker, selector)
results = evaluator.evaluate_model(citeagent, examples)
```

## Best Practices

### 1. Always Use Same Data Split

```python
# Save splits once
splits = loader.create_splits(seed=42)
loader.save_splits(splits, "data/splits/")

# Reuse for all experiments
test_examples = load_json("data/splits/test.json")
```

### 2. Track Everything

```python
evaluator = CitationEvaluator(
    track_latency=True,   # For performance analysis
    log_failures=True     # For error analysis
)
```

### 3. Batch Processing for Large Datasets

```python
# For 600K examples, process in batches
BATCH_SIZE = 10000
for i in range(0, len(examples), BATCH_SIZE):
    batch = examples[i:i+BATCH_SIZE]
    results = evaluator.evaluate_model(model, batch)
    save_checkpoint(f"batch_{i//BATCH_SIZE}", results)
```

### 4. Compare Models Fairly

```python
# Use same examples, same k, same evaluation
models = {
    'BM25': BM25Model(),
    'SPECTER2': DenseRetrievalModel('allenai/specter2'),
    'CiteAgent': CiteAgentModel()
}

comparison = evaluator.compare_models(
    models=models,
    examples=test_examples  # Same data for all
)
```

## Performance Tips

### For BM25
- Fast (10-20ms per query)
- No GPU needed
- Good for quick iterations

### For Dense Models
- Use GPU if available
- Batch encode corpus once, cache embeddings
- SPECTER2 is slower but better than E5 for scientific papers

### For Large-Scale Evaluation
```python
# Cache corpus embeddings
corpus_embeddings = model.encode([c['text'] for c in corpus])
save_embeddings(corpus_embeddings, "cache/corpus_emb.npy")

# Reuse in evaluation
corpus_embeddings = load_embeddings("cache/corpus_emb.npy")
```

## Troubleshooting

### ImportError: No module named 'bm25s'
```bash
pip install bm25s PyStemmer
```

### CUDA out of memory
```python
# Reduce batch size
model = DenseRetrievalModel(batch_size=8)

# Or use CPU
model = DenseRetrievalModel(device='cpu')
```

### Fuzzy matching too strict/loose
```python
# Adjust threshold (default=0.85)
evaluator = CitationEvaluator(fuzzy_match_threshold=0.80)
```

## Next Steps

1. **Run Baseline Evaluation**:
   ```bash
   python run_baseline_evaluation.py
   ```

2. **Implement CiteAgent**: Create `evaluation/models/citeagent_model.py`

3. **Compare Against Baselines**:
   ```python
   models = {'BM25': bm25, 'SPECTER2': specter2, 'CiteAgent': citeagent}
   comparison = evaluator.compare_models(models, examples)
   ```

4. **Scale to Full Dataset**: Use batch processing for 600K examples

5. **Benchmark on CiteME**: Run on official test set for paper-ready results

## Citation

If you use this framework, please cite:

```bibtex
@software{citation_eval_framework,
  title = {Citation Retrieval Evaluation Framework},
  author = {Gnanakumar, Sathya and Kalra, Ishaan and Suri, Dhruv and Kapoor, Kushal and Sreekanth, Vishnu and Singh, Vibhu},
  year = {2024}
}
```

## License

MIT License
