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
Aggregator (combines and fuses results)
    ↓
Reranker (LLM-based reranking)
    ↓
DSPy Picker (optional, citation selection)
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
- **Aggregator**: Combines and fuses results from multiple retrievers using Reciprocal Rank Fusion (RRF) or simple max-score methods
- **Reranker**: LLM-based reranking of retrieved results

##### Aggregator Details

The Aggregator (`src/agents/formulators/aggregator.py`) merges results from all retrievers and supports two fusion methods:

1. **Reciprocal Rank Fusion (RRF)** (default, recommended):

   - Combines rankings using: `RRF_score = sum(1 / (k + rank_i))` across all retrievers
   - Papers appearing in multiple retrievers get higher scores
   - Robust to score scale differences between retrievers
   - Better than simple score averaging

2. **Simple Max-Score**:
   - Takes the highest normalized score for each paper
   - Useful for debugging or when retrievers are calibrated

Configuration:

```python
config = {
    "aggregation_method": "rrf",  # or "simple"
    "rrf_k": 60  # RRF constant (default: 60)
}
```

#### Corpus Management

- **ScholarCopilot Corpus** (`src/corpus/scholarcopilot.py`): Handles loading and processing of ScholarCopilot dataset format

## Installation

### Prerequisites

- **For Docker Setup**: Docker and Docker Compose (recommended)
- **For Local Setup**: Python 3.10 or higher with `uv` package manager
- CUDA-capable GPU (recommended for dense retrieval)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ disk space for models and embeddings

### Option 1: Docker Setup (Recommended)

The easiest way to get started is using Docker Compose, which handles all dependencies and setup automatically.

#### 1. Install Prerequisites

```bash
# Install Docker and Docker Compose
# Visit https://docs.docker.com/get-docker/ for installation instructions

# For GPU support (optional but recommended), install NVIDIA Container Toolkit
# Visit https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

#### 2. Clone the repository

```bash
git clone <repository-url>
cd Retrieval
```

#### 3. Set up environment variables

```bash
cd server
cp .env.example .env
```

Edit `.env` and configure your API keys and settings:

```env
# Dataset path (will be auto-mounted in Docker)
DATASET_DIR=/app/data/scholarcopilot/scholar_copilot_eval_data_1k.json

# Graph output directory
GRAPH_OUTPUT_DIR=/app/graphs

# Optional: LLM API keys for reranking
OPENAI_API_KEY=your_key_here
S2_API_KEY=your_semantic_scholar_key
hf_key=your_huggingface_token

# Local Ollama model (optional)
LOCAL_LLM=gemma3:4b
```

#### 4. Place your dataset

```bash
# Create corpus directory structure (if not exists)
mkdir -p corpus/scholarcopilot

# Place your dataset file
cp /path/to/scholar_copilot_eval_data_1k.json corpus/scholarcopilot/
```

#### 5. Build and run with Docker Compose

```bash
# Build and start containers (from the Retrieval root directory)
cd ..
docker-compose up -d

# Access the container shell
docker-compose exec api bash

# Inside the container, run your retrieval tasks
uv run python evaluate.py
```

#### 6. Enable GPU support (optional)

To use GPU acceleration in Docker, uncomment the GPU sections in `docker-compose.yml`:

```yaml
# In the app service section:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then rebuild and restart:

```bash
docker-compose down
docker-compose up -d --build
```

#### 7. Pull Ollama models (optional)

If using local LLM with Ollama:

```bash
# Access Ollama container
docker-compose exec ollama ollama pull gemma3:4b

# Or pull other models
docker-compose exec ollama ollama pull llama3.1:8b
docker-compose exec ollama ollama pull qwen3:8b
```

#### Docker Commands Reference

```bash
# Start containers
docker-compose up -d

# Stop containers
docker-compose down

# View logs
docker-compose logs -f api

# Access container shell
docker-compose exec api bash

# Rebuild containers after code changes
docker-compose up -d --build

# Remove all containers and volumes
docker-compose down -v
```

### Option 2: Local Setup

#### 1. Clone the repository

```bash
git clone <repository-url>
cd server
```

#### 2. Install dependencies

```bash
uv sync
```

#### 3. Set up environment variables

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Configure the following variables:

```env
# Dataset path (relative to server/ directory)
DATASET_DIR=corpus/scholarcopilot/scholar_copilot_eval_data_1k.json

# Optional: LLM API keys for reranking
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

#### 4. Dataset Location

The ScholarCopilot dataset should be placed at `corpus/scholarcopilot/scholar_copilot_eval_data_1k.json` (relative to the server/ directory).

**Download the dataset**: [scholar_copilot_eval_data_1k.json](https://drive.google.com/file/d/1_8aSJPcSS0U9-uWMo7SPfYkKSTg8ZgAT/view?usp=sharing)

Once downloaded, place it in the correct location:

```bash
# From the server/ directory
mkdir -p corpus/scholarcopilot
cp ~/Downloads/scholar_copilot_eval_data_1k.json corpus/scholarcopilot/
```

You can verify the dataset is in the correct location:

```bash
# From the server/ directory
ls corpus/scholarcopilot/scholar_copilot_eval_data_1k.json
```

## Quick Start

### Using Docker

If you're using Docker, first access the container:

```bash
# Start containers
docker-compose up -d

# Access the container shell
docker-compose exec api bash
```

Then run commands inside the container as shown in the sections below.

### Basic Usage

Run the retrieval workflow with a query:

```bash
# Local setup
uv run python main.py \
  --dataset /path/to/dataset.json \
  --query "transformer architecture for sequence modeling" \
  --k 5

# Docker setup (inside container)
uv run python evaluate.py
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

| Variable            | Description                                 | Default                                                   |
| ------------------- | ------------------------------------------- | --------------------------------------------------------- |
| `DATASET_DIR`       | Path to ScholarCopilot dataset              | `corpus/scholarcopilot/scholar_copilot_eval_data_1k.json` |
| `OPENAI_API_KEY`    | OpenAI API key for reranking                | None                                                      |
| `ANTHROPIC_API_KEY` | Anthropic API key for reranking             | None                                                      |
| `TOGETHER_API_KEY`  | Together AI API key for reranking           | None                                                      |
| `GRAPH_OUTPUT_DIR`  | Directory for workflow graph visualizations | `./graphs`                                                |

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

## Evaluation CLI

The system includes a comprehensive evaluation CLI (`evaluate.py`) that supports three evaluation modes:

### 1. Pipeline Evaluation Mode

Evaluate the full multi-agent pipeline with reranking:

```bash
# Basic evaluation
uv run evaluate.py --mode pipeline --num-queries 100

# With individual retriever comparison
uv run evaluate.py --mode pipeline --num-queries 100 --compare

# With custom dataset path
uv run evaluate.py --mode pipeline --dataset /path/to/dataset.json --num-queries 50 --k 20
```

#### Pipeline Mode Options

| Flag              | Description                               | Default |
| ----------------- | ----------------------------------------- | ------- |
| `--num-queries N` | Number of queries to evaluate             | All     |
| `--k N`           | Top-k results to retrieve                 | 20      |
| `--compare`       | Compare individual retrievers vs pipeline | False   |
| `--cross-encoder` | Use cross-encoder instead of LLM reranker | False   |
| `--bm25-only`     | Only use BM25 (disable dense retrievers)  | False   |
| `--no-e5`         | Disable E5 retriever                      | False   |
| `--no-specter`    | Disable SPECTER retriever                 | False   |
| `--no-cache`      | Rebuild indexes from scratch              | False   |
| `--debug`         | Enable debug logging                      | False   |

#### Example: Full Comparison

```bash
# Compare all retrieval methods and show improvements
uv run evaluate.py \
    --mode pipeline \
    --dataset corpus/scholarcopilot/scholar_copilot_eval_data_1k.json \
    --num-queries 100 \
    --k 20 \
    --compare
```

**Output with `--compare` flag:**

```
======================================================================
RETRIEVAL METHOD COMPARISON
======================================================================
Method                           R@5     R@10     R@20      MRR
----------------------------------------------------------------------
BM25 (Keyword)                0.0234   0.0345   0.0456   0.8123
E5 (Dense)                    0.0312   0.0423   0.0534   0.8567
SPECTER (Dense)               0.0289   0.0398   0.0512   0.8345
Multi-Agent (RRF + Reranking) 0.0456   0.0589   0.0712   0.9234

----------------------------------------------------------------------
IMPROVEMENTS vs BEST INDIVIDUAL RETRIEVER:
----------------------------------------------------------------------
  • Recall@5:  +46.15%
  • Recall@10: +39.24%
  • Recall@20: +33.33%
  • MRR:       +7.79%
======================================================================
```

### 2. Baselines Evaluation Mode

Evaluate individual retrievers separately:

```bash
# Evaluate all retrievers
uv run evaluate.py --mode baselines --dataset PATH --retrievers bm25,e5,specter

# Evaluate specific retrievers only
uv run evaluate.py --mode baselines --dataset PATH --retrievers bm25,e5

# Custom k values
uv run evaluate.py --mode baselines --dataset PATH --k 10
```

**Output:**

- Recall@k for each retriever
- Latency comparison
- Performance plots (saved to `results/`)

### 3. Retrievers Testing Mode

Test and benchmark retriever implementations:

```bash
# Run all tests
uv run evaluate.py --mode retrievers --test-type all

# Test batch processing
uv run evaluate.py --mode retrievers --test-type batch --num-queries 20

# Test with custom models
uv run evaluate.py \
    --mode retrievers \
    --test-type cli \
    --e5-model intfloat/e5-large-v2 \
    --specter-model allenai/specter2 \
    --num-queries 10
```

### LLM Reranker Configuration

The system uses LLM-based reranking by default. You can choose between three inference engines:

#### Inference Engine Options

**1. Ollama (Default - Local, Easy Setup)**

Best for local development. Requires Ollama to be installed and running.

```bash
# .env file
INFERENCE_ENGINE=ollama
LOCAL_LLM=gemma3:4b
USE_OPENAI_RERANKER=false
```

Available Ollama models:
- `gemma3:4b` (default, lightweight, fast)
- `llama3.1:8b` (more capable)
- `qwen3:8b` (good performance)
- `mistral:7b` (balanced)

**2. HuggingFace (GPU Clusters Without Ollama)**

Best for GPU clusters where Ollama is difficult to run. Uses HuggingFace Transformers directly.

```bash
# .env file
INFERENCE_ENGINE=huggingface
LOCAL_LLM=meta-llama/Llama-3.2-3B-Instruct
USE_OPENAI_RERANKER=false
```

Available HuggingFace models:
- `meta-llama/Llama-3.2-3B-Instruct` (recommended, good performance)
- `google/gemma-2-2b-it` (lightweight, fast)
- `mistralai/Mistral-7B-Instruct-v0.3` (more capable)
- Any HuggingFace causal LM model

**CLI Override:**

```bash
# Use HuggingFace for this run only
uv run evaluate.py --mode pipeline --num-queries 10 \
  --inference-engine huggingface \
  --local-model meta-llama/Llama-3.2-3B-Instruct

# Use Ollama with different model
uv run evaluate.py --mode pipeline --num-queries 10 \
  --inference-engine ollama \
  --local-model qwen3:8b
```

**3. OpenAI (Cloud-based)**

Best for production use or when you need the highest quality results.

```bash
# .env file
INFERENCE_ENGINE=openai  # or USE_OPENAI_RERANKER=true
OPENAI_RERANKER_MODEL=gpt-5-mini-2025-08-07
OPENAI_API_KEY=sk-your-key-here
```

Available OpenAI models:
- `gpt-5-mini-2025-08-07` (default, recommended, cost-effective)
- `gpt-5-2025-08-07` (most capable, higher cost)
- `gpt-4o-mini` (older generation model)
- `gpt-4o` (older generation model)

**CLI Override:**

```bash
# Use cross-encoder instead of LLM
uv run evaluate.py --mode pipeline --num-queries 10 --cross-encoder

# Force OpenAI for this run
uv run evaluate.py --mode pipeline --num-queries 10 \
  --inference-engine openai
```

#### Quick Reference: When to Use Each Engine

| Engine | Best For | Pros | Cons |
|--------|----------|------|------|
| **Ollama** | Local development, testing | Fast setup, easy to use | Requires Ollama installed |
| **HuggingFace** | GPU clusters, research | Direct GPU access, flexible | Slower first load, more memory |
| **OpenAI** | Production, quality | Highest quality, no setup | Costs money, requires internet |

### DSPy Prompt Optimization (Optional)

Enable prompt-optimized citation picker:

```bash
# .env file
ENABLE_DSPY_PICKER=true
DSPY_MODEL=gpt-5-mini-2025-08-07
DSPY_TOP_N=10
```

When enabled, the comparison output will show:

```
Multi-Agent + DSPy          0.0523   0.0645   0.0789   0.9456
```

### Configuration Priority

Settings are applied in order (later overrides earlier):

1. Hardcoded defaults
2. `.env` file values
3. CLI flags (highest priority)

### Performance Notes

- **Individual retriever**: ~1-2 seconds per query
- **Full pipeline**: ~3-5 seconds per query with LLM reranking
- **With `--compare`**: ~10-15 seconds per query (evaluates each retriever separately)

For quick iterations:

```bash
# Quick test (5 queries)
uv run evaluate.py --mode pipeline --num-queries 5 --compare

# Full evaluation (100+ queries)
uv run evaluate.py --mode pipeline --num-queries 100 --compare
```

## Model Comparison

The system includes a dedicated model comparison tool (`compare_models.py`) that allows you to evaluate and compare different LLM models for reranking performance.

### Quick Start

Compare the default models (gpt-5-mini-2025-08-07 and gpt-4o-mini):

```bash
# Compare default models on 20 queries
uv run python compare_models.py --num-queries 20

# Compare specific models
uv run python compare_models.py \
  --models gpt-5-mini-2025-08-07 gpt-5-2025-08-07 gpt-4o-mini \
  --num-queries 50

# Compare with custom dataset
uv run python compare_models.py \
  --dataset corpus/scholarcopilot/scholar_copilot_eval_data_1k.json \
  --num-queries 100 \
  --output results/model_comparison.json
```

### Available Models

The comparison tool supports any OpenAI-compatible model. Common options include:

- **gpt-5-mini-2025-08-07** (default) - Most cost-effective, good performance
- **gpt-5-2025-08-07** - Most capable GPT-5 model (higher cost)
- **gpt-4o-mini** - Previous generation mini model
- **gpt-4o** - Previous generation capable model (higher cost)
- **gpt-4-turbo** - Previous generation fast model (higher cost)

### Comparison Output

The tool generates a detailed comparison table:

```
=====================================================================================================
MODEL PERFORMANCE COMPARISON
=====================================================================================================
Model                                R@5     R@10     R@20      MRR  Latency (s)   Est. Cost ($)
-----------------------------------------------------------------------------------------------------
gpt-5-mini-2025-08-07              0.0456   0.0589   0.0712   0.9234        3.42          0.0153
gpt-5-2025-08-07                   0.0478   0.0612   0.0745   0.9312        4.21          0.8500
gpt-4o-mini                        0.0445   0.0578   0.0698   0.9187        3.38          0.0153
=====================================================================================================

🏆 Best Model: gpt-5-2025-08-07
   Recall@20: 0.0745
   MRR: 0.9312
   Avg Latency: 4.21s
   Est. Cost: $0.8500
```

### Metrics Explained

- **R@5, R@10, R@20**: Recall at top 5, 10, and 20 results - measures how often the correct citation appears in the top-k results
- **MRR**: Mean Reciprocal Rank - measures how high the correct citation is ranked (1.0 = first position)
- **Latency**: Average time per query in seconds
- **Est. Cost**: Estimated API cost for the evaluation (based on current pricing)

### Command-Line Options

```bash
--dataset PATH          # Path to dataset (default: from DATASET_DIR env var)
--models MODEL [MODEL]  # Models to compare (default: gpt-5-mini-2025-08-07 gpt-4o-mini)
--num-queries N         # Number of queries to evaluate (default: 20)
--k N                   # Number of results to retrieve (default: 20)
--output PATH           # Save results to JSON file
--seed N                # Random seed for reproducibility (default: 42)
```

### Best Practices

1. **Start Small**: Use `--num-queries 20` for quick comparisons
2. **Scale Up**: Use `--num-queries 100+` for statistically significant results
3. **Save Results**: Use `--output` to save results for later analysis
4. **Cost Awareness**: Check estimated costs before running large evaluations
5. **Reproducibility**: Use `--seed` for consistent query sampling

### Example Workflows

**Quick test to choose a model:**
```bash
uv run python compare_models.py --num-queries 20
```

**Comprehensive evaluation:**
```bash
uv run python compare_models.py \
  --models gpt-5-mini-2025-08-07 gpt-5-2025-08-07 gpt-4o-mini \
  --num-queries 100 \
  --output results/full_comparison.json
```

**Compare cost vs. performance tradeoff:**
```bash
# Test on small sample first
uv run python compare_models.py --models gpt-5-mini-2025-08-07 gpt-5-2025-08-07 --num-queries 10

# If gpt-5-2025-08-07 shows significant improvement, run larger test
uv run python compare_models.py --models gpt-5-2025-08-07 --num-queries 100
```

## API Server

The system can be run as a REST API server for integration with web applications.

### Starting the API Server

```bash
# Development mode with auto-reload
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Production mode with multiple workers
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:

- **Base URL**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **Health Check**: http://localhost:8000/health

### API Endpoints

#### `GET /` - Root

Returns API information and status.

**Response:**

```json
{
  "message": "Citation Retrieval API",
  "version": "0.1.0",
  "docs": "/docs"
}
```

#### `GET /health` - Health Check

Returns health status and corpus information.

**Response:**

```json
{
  "status": "healthy",
  "corpus_size": 9740,
  "retrievers": ["bm25", "e5", "specter"]
}
```

#### `POST /api/find-citation` - Find Citation

Find the most relevant citations for a given context.

**Request Body:**

```json
{
  "context": "Transformers [CITATION] revolutionized NLP by introducing attention mechanisms.",
  "k": 5,
  "use_llm_reranker": true
}
```

**Parameters:**

- `context` (string, required): The citation context with `[CITATION]` placeholder
- `k` (integer, optional): Number of results to return (default: 5)
- `use_llm_reranker` (boolean, optional): Use LLM-based reranking (default: true)

**Response:**

```json
{
  "results": [
    {
      "citation": {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani, A.", "Shazeer, N.", "..."],
        "year": 2017,
        "source": "NeurIPS",
        "doi": "10.48550/arXiv.1706.03762",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."
      },
      "confidence": 99.5,
      "reasoning": "This paper introduced the Transformer architecture, which revolutionized NLP through self-attention mechanisms.",
      "score": 0.98,
      "formatted": {
        "apa": "Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.",
        "mla": "Vaswani, A., et al. \"Attention Is All You Need.\" NeurIPS, 2017.",
        "bibtex": "@inproceedings{vaswani2017attention,\n  title={Attention Is All You Need},\n  author={Vaswani, A. and ...},\n  booktitle={NeurIPS},\n  year={2017}\n}"
      }
    }
  ],
  "query": "Transformers revolutionized NLP by introducing attention mechanisms.",
  "expanded_queries": [
    "Transformers revolutionized NLP by introducing attention mechanisms.",
    "transformers revolutionized nlp introducing attention mechanisms",
    "paper discussing transformers, revolutionized, nlp, introducing, attention, mechanisms"
  ],
  "num_results": 5
}
```

### API Usage Examples

#### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Find citation
curl -X POST http://localhost:8000/api/find-citation \
  -H "Content-Type: application/json" \
  -d '{
    "context": "The architecture [CITATION] introduced attention mechanisms.",
    "k": 5,
    "use_llm_reranker": true
  }'
```

#### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Find citation
response = requests.post(
    "http://localhost:8000/api/find-citation",
    json={
        "context": "The transformer [CITATION] revolutionized NLP.",
        "k": 5,
        "use_llm_reranker": True
    }
)

results = response.json()
for i, result in enumerate(results["results"], 1):
    citation = result["citation"]
    print(f"{i}. {citation['title']} ({citation['year']})")
    print(f"   Confidence: {result['confidence']}%")
    print(f"   Score: {result['score']:.4f}")
    print()
```

#### Using JavaScript/TypeScript

```typescript
// Health check
const health = await fetch("http://localhost:8000/health");
const healthData = await health.json();
console.log(healthData);

// Find citation
const response = await fetch("http://localhost:8000/api/find-citation", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    context: "The transformer [CITATION] revolutionized NLP.",
    k: 5,
    use_llm_reranker: true,
  }),
});

const data = await response.json();
data.results.forEach((result, i) => {
  console.log(`${i + 1}. ${result.citation.title} (${result.citation.year})`);
  console.log(`   Confidence: ${result.confidence}%`);
  console.log(`   Formatted (APA): ${result.formatted.apa}`);
});
```

### API Configuration

Configure the API server via environment variables in `.env`:

```bash
# API Server
HOST=0.0.0.0
PORT=8000
USE_LLM_RERANKER=true

# CORS Settings (for frontend integration)
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Ollama Settings (for local LLM)
OLLAMA_HOST=http://localhost:11434
LOCAL_LLM=gemma3:4b
```

### API Performance

Expected response times:

- **With LLM reranking**: 3-5 seconds
- **Without LLM reranking**: 1-2 seconds
- **BM25 only**: <500ms

For production deployment:

- Use `--workers 4` or more for better throughput
- Consider using a reverse proxy (nginx, caddy)
- Enable CORS for frontend integration
- Set up proper logging and monitoring

## Testing

The project includes a comprehensive test suite. See [tests/README.md](tests/README.md) for detailed documentation.

For evaluation and testing, use the `evaluate.py` script as described in the [Evaluation CLI](#evaluation-cli) section above.

## Project Structure

```
server/
├── src/
│   ├── agents/
│   │   ├── retrievers/          # Retrieval agents (BM25, E5, SPECTER)
│   │   │   ├── bm25_agent.py
│   │   │   ├── e5_agent.py      # E5Retriever class with batch support
│   │   │   └── specter_agent.py # SPECTERRetriever class with batch support
│   │   └── formulators/          # Query reformulation, aggregation, and reranking
│   │       ├── query_reformulator.py
│   │       ├── aggregator.py    # Result fusion with RRF
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

---

**Version**: 0.1.0  
**Last Updated**: 2025
