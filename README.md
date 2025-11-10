# Citation Retrieval Project

**CMSC473 - Natural Language Processing**

**Team**: Sathya Gnanakumar, Ishaan Kalra, Dhruv Suri, Kushal Kapoor, Vishnu Sreekanth, Vibhu Singh

## 🎯 Project Overview

This project develops and evaluates multiple approaches for **automatic citation retrieval** - the task of identifying which paper is cited given a citation context from a scientific document. We implement and compare three baseline approaches and plan to develop a novel multi-agent system using LangGraph.

## 📁 Project Structure

```
CitationRetrieval/
├── datasets/                    # Centralized datasets
│   ├── scholar_copilot_eval_data_1k.json  # ScholarCopilot benchmark (1K examples)
│   └── CiteME.tsv                          # CiteME benchmark (100 examples)
│
├── baselines/                   # All baseline implementations
│   ├── bm25/                   # BM25 sparse retrieval baseline
│   ├── dense/                  # Dense retrieval (SPECTER2, E5-Large)
│   └── cite_agent/             # LLM-based agent baseline
│       ├── src/                # Source code
│       ├── CiteME/             # CiteME dataset metadata
│       └── README.md           # Agent documentation
│
├── evaluation/                  # Unified evaluation framework
│   ├── evaluator.py           # Main evaluation harness
│   ├── metrics.py             # Metrics (Recall@k, MRR, etc.)
│   ├── data_loader.py         # Dataset loading
│   ├── models/                # Model wrappers for all baselines
│   └── README.md              # Evaluation documentation
│
├── multi_agent_pipeline/       # 🚧 Future: Multi-agent system (LangGraph)
│   ├── src/                   # Source code (placeholder)
│   ├── configs/               # Configuration files (placeholder)
│   ├── tests/                 # Tests (placeholder)
│   └── README.md              # Architecture plan
│
├── data_processing/            # Data cleaning and processing scripts
│   ├── data_cleaning.py       # Data cleaning utilities
│   └── *.csv, *.json          # Processed data files
│
├── example_evaluation.py       # Quick start evaluation script
└── README.md                   # This file
```

## 📊 Datasets

### ScholarCopilot Eval Data (Primary)
- **Location**: `datasets/scholar_copilot_eval_data_1k.json`
- **Size**: 1,000 papers, ~5,000-10,000 citation instances
- **Format**: JSON with full paper text and citation markers
- **Source**: [ScholarCopilot Training Dataset](https://huggingface.co/datasets/ubowang/ScholarCopilot-TrainingData)
- **Use**: Primary evaluation benchmark

### CiteME (Secondary)
- **Location**: `datasets/CiteME.tsv`
- **Size**: ~100 citation instances
- **Format**: TSV with curated citation excerpts
- **Source**: [CiteME Dataset](https://huggingface.co/datasets/bethgelab/CiteME)
- **Use**: Secondary benchmark for comparison

### Full Training Data
- **ScholarCopilot Database**: [600K papers](https://huggingface.co/datasets/TIGER-Lab/ScholarCopilot-Data-v1)
- **Note**: Currently using 1K subset due to compute constraints

### Additional Resources
- **All Datasets (Google Drive)**: [Download Link](https://drive.google.com/drive/folders/1BsRwzx_CGlXdSRT_at1LPdgmwXB8CUcC?usp=sharing)

## 🚀 Quick Start

### Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Clone the repository
git clone <repo-url>
cd CitationRetrieval

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

### Environment Setup

1. **Copy the environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Add your API keys** to `.env`:
   ```bash
   # Required for CiteAgent baseline
   OPENAI_API_KEY=your_openai_api_key_here          # For GPT-4o models
   ANTHROPIC_API_KEY=your_anthropic_api_key_here    # For Claude models
   TOGETHER_API_KEY=your_together_api_key_here      # For open-source models

   # Required for paper search
   S2_API_KEY=your_semantic_scholar_api_key_here
   ```

3. **Get your API keys**:
   - **OpenAI**: https://platform.openai.com/api-keys
   - **Anthropic**: https://console.anthropic.com/
   - **Together AI**: https://api.together.xyz/settings/api-keys
   - **Semantic Scholar**: https://www.semanticscholar.org/product/api#api-key (free tier available)

> **Note**: BM25 and Dense Retrieval baselines don't require API keys. Only CiteAgent needs them.

### Run Quick Evaluation

```bash
# Test on 50 examples (quick) - No API keys needed
uv run main.py

# Test CiteAgent (requires API keys)
cd baselines/cite_agent
uv run src/main.py
```

## 📈 Baseline Methods

### 1. BM25 (Sparse Retrieval)
**Location**: `baselines/bm25/`

Traditional information retrieval using BM25 ranking algorithm.

```bash
cd baselines/bm25
uv run bm25ScholarCopilot.py
```

**Results**: See `baselines/bm25/bm25_scholarcopilot_full_results.csv`

### 2. Dense Retrieval
**Location**: `baselines/dense/`

Neural embedding-based retrieval using:
- **SPECTER2**: Scientific paper embeddings
- **E5-Large**: General-purpose dense embeddings

```bash
cd baselines/dense
uv run jupyter notebook Dense_Retrieval.ipynb
```

**Results**: [Google Drive Results](https://drive.google.com/drive/folders/1L1Eo1dE77bOelBOvWEy466Hhir8OSYPE?usp=sharing)

### 3. CiteAgent (LLM Agent)
**Location**: `baselines/cite_agent/`

LLM-powered agent using structured actions:
- Search by relevance/citation count
- Read paper abstracts
- Select best match

```bash
cd baselines/cite_agent
uv run src/main.py
```

**Features**:
- Multiple LLM backends (GPT-4, Claude, Together AI)
- Semantic Scholar API integration
- Pydantic-structured outputs
- Few-shot prompting

**Requirements**: Requires API keys (see Environment Setup above)

See [baselines/cite_agent/README.md](baselines/cite_agent/README.md) for details.

## 📊 Evaluation Framework

**Location**: `evaluation/`

Unified evaluation harness supporting all baselines with comprehensive metrics:

### Metrics
- **Recall@k**: Citation found in top-k results
- **Precision@k**: Precision in top-k results
- **MRR**: Mean Reciprocal Rank
- **Exact Match**: Perfect match rate

### Usage

```python
from evaluation import CitationEvaluator, CitationDataLoader
from evaluation.models import BM25Model, DenseRetrievalModel

# Load data
loader = CitationDataLoader("datasets/scholar_copilot_eval_data_1k.json")
examples = loader.extract_examples()

# Initialize models
models = {
    'BM25': BM25Model(),
    'SPECTER2': DenseRetrievalModel('allenai/specter2'),
}

# Run comparison
evaluator = CitationEvaluator()
comparison = evaluator.compare_models(models, examples)
```

See [evaluation/README.md](evaluation/README.md) for detailed documentation.

## 🤖 Future: Multi-Agent Pipeline

**Location**: `multi_agent_pipeline/` (Not yet implemented)

Planned multi-agent system using **LangGraph** and **Context7**:

### Planned Architecture
1. **Query Reformulation Agent**: Enhances citation context
2. **Search Agent**: Multi-source paper retrieval
3. **Ranking Agent**: Candidate scoring
4. **Verification Agent**: Citation validation
5. **Coordinator Agent**: Pipeline orchestration

### Technology Stack
- LangGraph for agent workflow
- Context7 for coordination
- Semantic Scholar API
- Fine-tuning on ScholarCopilot subset

See [multi_agent_pipeline/README.md](multi_agent_pipeline/README.md) for the detailed plan.

## 📅 Project Timeline

### ✅ Completed

**Week 1**: Initial exploration of CiteME paper and baseline planning

**Weeks 2-4**:
- Pivoted to BM25 and Dense Retrieval baselines
- Selected ScholarCopilot as primary dataset
- Designed multi-agent pipeline architecture

**Week 5**:
- Implemented BM25 baseline
- Implemented Dense Retrieval (SPECTER2, E5-Large)
- Collected baseline results

**Week 6+**:
- Built unified evaluation framework
- Implemented CiteAgent LLM baseline
- Comprehensive metrics and error analysis

### 🚧 In Progress

- Full evaluation on 1K ScholarCopilot examples
- CiteAgent performance optimization
- Error analysis and failure case studies

### 📋 Planned

- Multi-agent pipeline implementation (LangGraph)
- Training on full 600K ScholarCopilot dataset
- Final benchmark on CiteME test set
- Comparative analysis vs. baselines

## 🎯 Research Goals

1. **Establish Baselines**: Benchmark BM25, Dense Retrieval, and LLM agents
2. **Multi-Agent Innovation**: Develop novel multi-agent retrieval system
3. **Scalability**: Scale to 600K training examples
4. **Generalization**: Test on multiple benchmarks (ScholarCopilot, CiteME)
5. **Real-world Application**: Deploy as citation recommendation service

## 📊 Current Results

### BM25 Baseline
- Evaluation on ScholarCopilot 1K subset
- Results: `baselines/bm25/bm25_scholarcopilot_full_results.csv`

### Dense Retrieval
- SPECTER2 and E5-Large embeddings
- Results: [Google Drive](https://drive.google.com/drive/folders/1L1Eo1dE77bOelBOvWEy466Hhir8OSYPE?usp=sharing)

### CiteAgent
- LLM-powered agent with search capabilities
- Evaluation in progress

## 🔧 Technical Details

### Project Structure
- **Unified uv project**: Single `pyproject.toml` at root manages all dependencies
- **Python 3.12+**: Required for all baselines
- **Environment variables**: Managed via `.env` file (see `.env.example`)

### APIs Used
- **Semantic Scholar API**: Paper search and metadata
- **OpenAI API**: GPT-4o for LLM agent
- **Anthropic API**: Claude for alternative LLM backend
- **Together AI**: Open-source LLM hosting

### Key Dependencies
- `uv`: Fast Python package installer and resolver
- `langchain`: LLM orchestration
- `sentence-transformers`: Dense embeddings
- `bm25s`: BM25 implementation
- `pydantic`: Structured outputs
- `pandas`, `numpy`: Data processing

See `pyproject.toml` for complete dependency list.

## 📚 Resources

- [ScholarCopilot Paper](https://arxiv.org/abs/2305.11041)
- [CiteME Paper](https://arxiv.org/abs/2310.04685)
- [SPECTER2 Paper](https://arxiv.org/abs/2004.07180)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Semantic Scholar API Docs](https://api.semanticscholar.org/)

## 📄 License

Academic use only - CMSC473 course project
