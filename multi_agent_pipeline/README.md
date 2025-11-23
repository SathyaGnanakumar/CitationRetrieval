# Multi-Agent Pipeline (LangGraph + Context7)

**Status**: 🚧 Not Yet Implemented - Planned for Future Development

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
