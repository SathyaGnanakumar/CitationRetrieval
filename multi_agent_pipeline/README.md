# Multi-Agent Pipeline (LangGraph + Context7)

**Status**: 🚧 In Progress - Core agents scaffolding underway

## Overview

This folder will contain the multi-agent citation retrieval pipeline built using **LangGraph** and **Context7** framework. The multi-agent system will combine multiple specialized agents to achieve better citation retrieval accuracy than individual baselines.

## Planned Architecture

The multi-agent pipeline will consist of:

### Agent Roles
1. **Query Reformulation Agent**: Enhances the citation context for better search
2. **Search Agent**: Retrieves candidate papers from multiple sources
3. **Ranking Agent**: Scores and ranks candidate papers
4. **Verification Agent**: Validates the selected citation
5. **Coordinator Agent**: Orchestrates the pipeline and makes final decisions

## Implemented Components (current)

- **EntityRecognitionAgent** (`src/agents/entity_recognition.py`): wraps a Hugging Face
  token-classification pipeline (defaults to `dslim/bert-base-NER`) and annotates the
  citation context with `RecognizedEntity` instances stored on the shared `AgentContext`.
- **VerificationAgent** (`src/agents/verification.py`): consumes recognized entities and
  retrieved citation candidates, producing a `VerificationResult` with heuristic scoring
  over title/abstract matches, fuzzy alignment, and retrieval confidence.
- **Shared Types** (`src/types.py`): dataclasses for `AgentContext`, `CandidatePaper`,
  `RecognizedEntity`, and `VerificationResult` enabling consistent data exchange between
  pipeline stages.
- **ScholarCopilotDataset utilities** (`src/data.py`): lightweight wrapper around the
  evaluation `CitationDataLoader` that yields `TrainingExample` objects with populated
  `AgentContext`, providing a consistent entry-point for training loops.

### Usage Example

```python
from multi_agent_pipeline.src import AgentContext, CandidatePaper
from multi_agent_pipeline.src.agents import EntityRecognitionAgent, VerificationAgent

context = AgentContext(
    citation_context="Recent work by Smith et al. (2024) introduced the GraphAlign model...",
    retrieved_candidates=[
        CandidatePaper(title="GraphAlign: Alignment Models for Graph Data", score=0.72),
        CandidatePaper(title="Neural Citation Matching with Aligners", score=0.55),
    ],
)

ner_agent = EntityRecognitionAgent()
verify_agent = VerificationAgent()

context = ner_agent(context)
verification = verify_agent(context)

print(verification.selected_candidate.title)
print(verification.justification)
```

### Loading Training Data

```python
from multi_agent_pipeline.src import ScholarCopilotDataset

dataset = ScholarCopilotDataset()  # defaults to datasets/scholar_copilot_eval_data_1k.json
examples = dataset.sample(limit=100)

for example in examples:
    context = example.context
    # feed context to retrieval/agent pipeline or fine-tuning loop
```

### Run Verification Workflow

Use the CLI utility to run EntityRecognitionAgent + VerificationAgent (with optional Semantic Scholar lookup) on any ScholarCopilot example:

```bash
uv run python -m multi_agent_pipeline.src.verify_citation \
    --dataset datasets/scholar_copilot_eval_data_1k.json \
    --example-index 0 \
    --external-title-threshold 0.9

# Disable external lookup if you want local-only scoring
uv run python -m multi_agent_pipeline.src.verify_citation --disable-external-lookup
```

> Ensure `S2_API_KEY` is set in your environment when Semantic Scholar verification is enabled.

> **Note**: The NER agent lazily loads the transformers pipeline; ensure the
> `transformers` package is installed (added to `pyproject.toml`) and optionally
> pass `device="cuda:0"` to leverage a GPU. Execution on CPU is supported and is
> sufficient for prototyping.

### Technology Stack
- **LangGraph**: For building the agent workflow graph
- **Context7**: For agent coordination and context management
- **LangChain**: For LLM interactions and tooling
- **Semantic Scholar API**: For paper search and retrieval

## Training Strategy

The multi-agent system will be:
1. Trained on a subset of the ScholarCopilot dataset (600K examples)
2. Fine-tuned to narrow down possible citation candidates
3. Optimized for accuracy using reinforcement learning or reward modeling

## Evaluation

The pipeline will be evaluated against the three baselines:
- BM25 (sparse retrieval)
- Dense Retrieval (SPECTER2, E5-Large)
- CiteAgent (single LLM agent)

Using both test datasets:
- **ScholarCopilot eval data** (1K examples)
- **CiteME benchmark** (100 examples)

## Folder Structure (Planned)

```
multi_agent_pipeline/
├── src/
│   ├── agents/           # Individual agent implementations
│   ├── graph/            # LangGraph workflow definitions
│   ├── tools/            # Custom tools for agents
│   └── orchestrator.py   # Main pipeline coordinator
├── configs/
│   ├── agent_configs.yaml
│   └── pipeline_configs.yaml
├── tests/
│   └── test_pipeline.py
├── notebooks/
│   └── pipeline_exploration.ipynb
└── README.md
```

## Getting Started (Future)

```bash
# Install dependencies
pip install langgraph langchain context7

# Run the pipeline
python -m multi_agent_pipeline.src.orchestrator \
    --dataset datasets/scholar_copilot_eval_data_1k.json \
    --config configs/pipeline_configs.yaml
```

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Context7 Framework](https://github.com/context7)
- [ScholarCopilot Paper](https://arxiv.org/abs/...)
- [CiteME Paper](https://arxiv.org/abs/...)

## Timeline

- **Phase 1**: Design agent architecture and workflow
- **Phase 2**: Implement individual agents
- **Phase 3**: Integrate with LangGraph
- **Phase 4**: Train on ScholarCopilot subset
- **Phase 5**: Evaluate and iterate
- **Phase 6**: Scale to full dataset

## Notes

This is a placeholder for the future multi-agent implementation. The current focus is on baseline evaluations (BM25, Dense Retrieval, CiteAgent) to establish performance benchmarks.
