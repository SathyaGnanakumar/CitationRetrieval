# Citation Retrieval System

A multi-baseline citation retrieval system that combines BM25, dense retrieval (E5 and SPECTER), and LLM-based reranking to find relevant academic papers for citation contexts. Built with LangGraph for orchestration and designed for the ScholarCopilot dataset.

## Overview

This system provides a comprehensive solution for academic citation retrieval, combining multiple retrieval strategies to find the most relevant papers for a given query or citation context. It uses a LangGraph-based workflow to orchestrate query reformulation, parallel retrieval from multiple sources, result analysis, and reranking.

### Key Features

- **Multi-Stage Retrieval Pipeline**: Combines sparse (BM25) and dense (E5, SPECTER) retrieval methods
- **Query Reformulation**: Automatically expands and reformulates queries for better retrieval
- **Batch Processing**: Efficient GPU-accelerated batch processing for dense retrievers
- **LLM-Based Reranking**: Uses language models to rerank and improve final results
- **ScholarCopilot Integration**: Built-in support for the ScholarCopilot dataset format
- **Modular Architecture**: Class-based retrievers that work with any corpus source
- **Comprehensive Testing**: Full test suite with single and batch query processing

## Architecture

### Workflow Pipeline

The system uses a LangGraph-based workflow with the following stages:

```
Query Input
    ↓
Query Reformulator (expands queries)
    ↓
    ├──→ BM25 Retriever (sparse retrieval)
    ├──→ E5 Retriever (dense retrieval)
    └──→ SPECTER Retriever (academic paper embeddings)
    ↓
Analysis Agent (combines results)
    ↓
Reranker (LLM-based reranking)
    ↓
Final Ranked Results
```

### Components

#### Retrievers

1. **BM25 Agent** (`src/agents/retrievers/bm25_agent.py`)

   - Sparse retrieval using BM25 algorithm
   - Fast, keyword-based matching
   - No GPU required

2. **E5 Retriever** (`src/agents/retrievers/e5_agent.py`)

   - Dense retrieval using E5 embeddings
   - Class-based with `single_query()` and `batch_query()` methods
   - GPU-accelerated batch processing
   - Models: `intfloat/e5-base-v2`, `intfloat/e5-large-v2`

3. **SPECTER Retriever** (`src/agents/retrievers/specter_agent.py`)
   - Academic paper embeddings using SPECTER2
   - Optimized for scientific literature
   - Class-based with batch processing support
   - Models: `allenai/specter2_base`, `allenai/specter2`

#### Formulators

- **Query Reformulator**: Expands queries with keywords and academic-style rewrites
- **Reranker**: LLM-based reranking of retrieved results

#### Corpus Management

- **ScholarCopilot Corpus** (`src/corpus/scholarcopilot.py`): Handles loading and processing of ScholarCopilot dataset format

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for dense retrieval)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ disk space for models and embeddings

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd server
```

2. **Install dependencies**

```bash
uv sync
```

For development dependencies:

```bash
uv sync --extra dev
```

3. **Set up environment variables**

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Configure the following variables:

```env
# Dataset path
DATASET_DIR=/path/to/scholar_copilot_eval_data_1k.json

# Optional: LLM API keys for reranking
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
TOGETHER_API_KEY=your_key_here
```

4. **Download the dataset**

Place the ScholarCopilot dataset JSON file at the path specified in `DATASET_DIR` or use the default location: `../datasets/scholar_copilot_eval_data_1k.json`

## Quick Start

### Basic Usage

Run the retrieval workflow with a query:

```bash
uv run python main.py \
  --dataset /path/to/dataset.json \
  --query "transformer architecture for sequence modeling" \
  --k 5
```

### Using the Workflow Programmatically

```python
from src.workflow import RetrievalWorkflow
from src.corpus.scholarcopilot import load_dataset, build_citation_corpus
from src.resources.builders import build_inmemory_resources
from langchain_core.messages import HumanMessage

# Load dataset and build resources
dataset = load_dataset("/path/to/dataset.json")
corpus = build_citation_corpus(dataset)
resources = build_inmemory_resources(corpus)

# Initialize workflow
workflow = RetrievalWorkflow()

# Run query
result = workflow.run({
    "messages": [HumanMessage(content="transformer architecture")],
    "resources": resources,
    "config": {"k": 5}
})

# Access results
ranked_papers = result.get("ranked_papers", [])
for paper in ranked_papers[:5]:
    print(f"{paper['title']} (score: {paper.get('rerank_score', paper.get('score'))})")
```

### Using Individual Retrievers

#### E5 Retriever

```python
from src.agents.retrievers.e5_agent import E5Retriever
from src.resources.builders import build_e5_resources

# Build resources
resources = build_e5_resources(corpus, model_name="intfloat/e5-base-v2")

# Initialize retriever
retriever = E5Retriever(resources["model"], resources.get("device"))

# Single query
results = retriever.single_query(
    query="transformer architecture",
    corpus_embeddings=resources["corpus_embeddings"],
    ids=resources["ids"],
    titles=resources["titles"],
    k=5
)

# Batch queries
queries = ["query 1", "query 2", "query 3"]
results_per_query = retriever.batch_query(
    queries,
    resources["corpus_embeddings"],
    resources["ids"],
    resources["titles"],
    k=5
)
```

#### SPECTER Retriever

```python
from src.agents.retrievers.specter_agent import SPECTERRetriever
from src.resources.builders import build_specter_resources

# Build resources
resources = build_specter_resources(corpus, model_name="allenai/specter2_base")

# Initialize retriever
retriever = SPECTERRetriever(
    resources["model"],
    resources["tokenizer"],
    resources.get("device")
)

# Single or batch queries (same API as E5)
```

## Configuration

### Environment Variables

| Variable            | Description                                 | Default                                         |
| ------------------- | ------------------------------------------- | ----------------------------------------------- |
| `DATASET_DIR`       | Path to ScholarCopilot dataset              | `../datasets/scholar_copilot_eval_data_1k.json` |
| `OPENAI_API_KEY`    | OpenAI API key for reranking                | None                                            |
| `ANTHROPIC_API_KEY` | Anthropic API key for reranking             | None                                            |
| `TOGETHER_API_KEY`  | Together AI API key for reranking           | None                                            |
| `GRAPH_OUTPUT_DIR`  | Directory for workflow graph visualizations | `./graphs`                                      |

### Model Configuration

Models can be configured when building resources:

```python
# E5 models
build_e5_resources(corpus, model_name="intfloat/e5-base-v2")  # Faster, smaller
build_e5_resources(corpus, model_name="intfloat/e5-large-v2")  # More accurate

# SPECTER models
build_specter_resources(corpus, model_name="allenai/specter2_base")  # Base model
build_specter_resources(corpus, model_name="allenai/specter2")  # Larger model
```

## Testing

The project includes a comprehensive test suite. See [tests/README.md](tests/README.md) for detailed documentation.

### Quick Test Commands

```bash
# Run all tests
uv run python tests/test_retrievers_batch.py

# Run specific test types
uv run python tests/test_retrievers_batch.py --test-type single
uv run python tests/test_retrievers_batch.py --test-type batch
uv run python tests/test_retrievers_batch.py --test-type cli

# Test with custom models
uv run python tests/test_retrievers_batch.py \
  --test-type cli \
  --e5-model intfloat/e5-large-v2 \
  --specter-model allenai/specter2 \
  --num-queries 20

# Run with pytest
uv run pytest tests/ -v
```

### Test Coverage

- ✅ Single query processing (E5 & SPECTER)
- ✅ Batch query processing (10+ queries)
- ✅ Consistency verification (single vs batch)
- ✅ Performance benchmarking
- ✅ Corpus loading and processing
- ✅ Workflow integration

## Project Structure

```
server/
├── src/
│   ├── agents/
│   │   ├── retrievers/          # Retrieval agents (BM25, E5, SPECTER)
│   │   │   ├── bm25_agent.py
│   │   │   ├── e5_agent.py      # E5Retriever class with batch support
│   │   │   └── specter_agent.py # SPECTERRetriever class with batch support
│   │   └── formulators/          # Query reformulation and reranking
│   │       ├── query_reformulator.py
│   │       ├── reranker.py
│   │       └── dspy_prompt_generator/  # DSPy-based prompt generation
│   ├── corpus/                   # Corpus loading and processing
│   │   └── scholarcopilot.py
│   ├── models/                   # State models
│   │   └── state.py
│   ├── resources/                # Resource builders
│   │   └── builders.py
│   ├── services/                 # External service integrations
│   │   ├── semantic_scholar.py
│   │   └── arxiv_retriever.py
│   └── workflow.py              # Main LangGraph workflow
├── tests/                        # Test suite
│   ├── README.md                 # Testing documentation
│   ├── test_retrievers_batch.py # Comprehensive retriever tests
│   └── ...
├── main.py                       # CLI entry point
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Performance

### Batch Processing

The system is optimized for batch processing on GPU:

- **E5 Batch**: 5-10x faster than sequential single queries on GPU
- **SPECTER Batch**: 3-5x faster than sequential single queries on GPU
- **Memory Efficient**: Resources are cached and reused across queries

### Expected Performance

On GPU (CUDA):

- **E5 (base)**: ~0.2-0.3s per query in batch
- **SPECTER (base)**: ~0.3-0.4s per query in batch
- **BM25**: ~0.01s per query (CPU)

On CPU:

- **E5 (base)**: ~1-2s per query in batch
- **SPECTER (base)**: ~2-3s per query in batch

### Resource Building Time

First-time resource building (one-time cost):

- Corpus building: 1-2 minutes
- E5 embeddings: 2-5 minutes (depends on corpus size)
- SPECTER embeddings: 2-8 minutes (depends on corpus size)

Subsequent runs use cached resources.

## Development

### Code Style

The project uses:

- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **Type hints** throughout

Format code:

```bash
uv run black src/ tests/
uv run ruff check src/ tests/
```

### Adding New Retrievers

1. Create a new retriever class in `src/agents/retrievers/`
2. Implement `single_query()` and `batch_query()` methods
3. Add resource builder in `src/resources/builders.py`
4. Integrate into workflow in `src/workflow.py`
5. Add tests in `tests/`

Example structure:

```python
class NewRetriever:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device

    def single_query(self, query, corpus_embeddings, ids, titles, k=5):
        # Implementation
        pass

    def batch_query(self, queries, corpus_embeddings, ids, titles, k=5):
        # Implementation
        pass
```

### Running Development Server

```bash
# Fast mode (BM25 only, no dense embeddings)
uv run python main.py --dataset /path/to/dataset.json --bm25-only

# Full mode with all retrievers
uv run python main.py --dataset /path/to/dataset.json
```

## Troubleshooting

### Common Issues

**1. Dataset Not Found**

```bash
# Set DATASET_DIR environment variable
export DATASET_DIR="/path/to/dataset.json"

# Or use --dataset flag
uv run python main.py --dataset /path/to/dataset.json
```

**2. CUDA Out of Memory**

- Use smaller models (`e5-base-v2` instead of `e5-large-v2`)
- Reduce batch size in resource building
- Use CPU mode (slower but works)

**3. Model Download Issues**

- Check internet connection (models download from HuggingFace)
- Verify model names are correct
- Check HuggingFace Hub access

**4. Import Errors**

```bash
# Ensure you're in the server directory
cd server
uv run python main.py

# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## API Reference

### RetrievalState

The workflow state schema:

```python
class RetrievalState(MessagesState):
    query: str                    # Base query
    queries: List[str]            # Expanded queries
    resources: Dict[str, Any]     # Retrieval resources
    config: Dict[str, Any]        # Configuration (k, etc.)
```

### E5Retriever

```python
class E5Retriever:
    def __init__(self, model, device: Optional[str] = None)
    def single_query(query, corpus_embeddings, ids, titles, k=5) -> List[Dict]
    def batch_query(queries, corpus_embeddings, ids, titles, k=5) -> List[List[Dict]]
```

### SPECTERRetriever

```python
class SPECTERRetriever:
    def __init__(self, model, tokenizer, device: Optional[str] = None)
    def single_query(query, corpus_embeddings, ids, titles, k=5) -> List[Dict]
    def batch_query(queries, corpus_embeddings, ids, titles, k=5) -> List[List[Dict]]
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`uv run pytest tests/`)
5. Format code (`uv run black src/ tests/`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

[Add your license information here]

## Citation

If you use this system in your research, please cite:

```bibtex
@software{citation-retrieval,
  title = {Citation Retrieval System},
  author = {[Your Name/Organization]},
  year = {2024},
  url = {[Repository URL]}
}
```

## Acknowledgments

- **E5 Models**: [intfloat/e5](https://huggingface.co/intfloat/e5-base-v2)
- **SPECTER Models**: [allenai/specter2](https://huggingface.co/allenai/specter2_base)
- **LangGraph**: For workflow orchestration
- **ScholarCopilot**: For the evaluation dataset

## Support

For issues, questions, or contributions:

1. Check the [troubleshooting section](#troubleshooting)
2. Review [tests/README.md](tests/README.md) for testing documentation
3. Open an issue on GitHub
4. Check existing issues and discussions

---

**Version**: 0.1.0  
**Last Updated**: 2025
