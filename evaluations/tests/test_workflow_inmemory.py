import sys
from pathlib import Path
from typing import List

# Ensure `import src...` works when running pytest from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage

from src.resources.builders import build_bm25_resources
from src.workflow import RetrievalWorkflow


class FakeReranker:
    """Simple deterministic reranker for tests (no model downloads)."""

    def compute_score(self, pairs: List[List[str]], normalize: bool = True):
        # Higher score for earlier items, deterministically
        n = len(pairs)
        return [float(n - i) for i in range(n)]


def test_workflow_runs_with_injected_resources_and_returns_ranked_papers():
    docs = [
        {
            "id": "c1",
            "title": "Attention Is All You Need",
            "abstract": "Transformers...",
            "text": "Attention Is All You Need. Transformers...",
        },
        {
            "id": "c2",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "Masked LM...",
            "text": "BERT: Pre-training of Deep Bidirectional Transformers. Masked LM...",
        },
        {
            "id": "c3",
            "title": "EfficientNet",
            "abstract": "Scaling CNNs...",
            "text": "EfficientNet. Scaling CNNs...",
        },
    ]

    resources = {
        "bm25": build_bm25_resources(docs),
        # Leave out dense retrievers to ensure workflow handles missing resources gracefully
        "reranker_model": FakeReranker(),
    }

    workflow = RetrievalWorkflow()
    final_state = workflow.run(
        {
            "messages": [HumanMessage(content="transformer architecture for sequence modeling")],
            "resources": resources,
            "config": {"k": 2},
        }
    )

    assert "ranked_papers" in final_state
    assert isinstance(final_state["ranked_papers"], list)
    assert len(final_state["ranked_papers"]) > 0
    assert "rerank_score" in final_state["ranked_papers"][0]
