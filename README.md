# Citation Retrieval System

A multi-baseline citation retrieval system that combines BM25, dense retrieval (E5 and SPECTER), and LLM-based reranking to find relevant academic papers for citation contexts. Built with LangGraph for orchestration and designed for the ScholarCopilot dataset.

## Overview

This system provides a comprehensive solution for academic citation retrieval, combining multiple retrieval strategies to find the most relevant papers for a given query or citation context. It uses a LangGraph-based workflow to orchestrate query reformulation, parallel retrieval from multiple sources, result aggregation, and LLM-based reranking.

The project consists of:

1. **Server** (Python): The core retrieval pipeline, evaluation scripts, and REST API
2. **Client** (Next.js): Optional web interface for finding citations (not required for evaluation)

## Getting Started

**For detailed setup instructions, evaluation guide, and API documentation, see [server/README.md](server/README.md)**

The server README contains:
- Complete installation instructions (local setup or Docker)
- Evaluation CLI usage and examples
- API server documentation
- Configuration options
- Performance benchmarks
- Troubleshooting guide

## Quick Links

- **Evaluation & Setup**: [server/README.md](server/README.md) - Start here for running evaluations on the ScholarCopilot corpus
- **Frontend UI** (optional): [client/README.md](client/README.md) - Web interface for interactive citation finding

## Key Features

- **Multi-Stage Retrieval Pipeline**: Combines sparse (BM25) and dense (E5, SPECTER) retrieval methods
- **Query Reformulation**: Automatically expands and reformulates queries for better retrieval
- **LLM-Based Reranking**: Uses language models (Ollama or OpenAI) to rerank results
- **Reciprocal Rank Fusion**: Intelligently combines results from multiple retrievers
- **ScholarCopilot Integration**: Built-in support for the ScholarCopilot dataset format
- **GPU-Accelerated**: Batch processing for efficient dense retrieval
- **Comprehensive Evaluation**: Compare individual retrievers vs full pipeline with detailed metrics

## Architecture Overview

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

For detailed information about each component, see [server/README.md](server/README.md).
