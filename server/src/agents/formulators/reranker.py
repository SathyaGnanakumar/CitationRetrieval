from typing import List, Dict, Any
from langgraph.graph import MessagesState
from FlagEmbedding import FlagReranker


def processed_list(query: str, candidate_papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs = []
    for paper in candidate_papers:
        pairs.append([query, paper.get("title", "")])
    return pairs


# src/agents/ranking_agent.py
def reranker(state: MessagesState, model_name: str = "BAAI/bge-reranker-v2-m3"):
    print(f"📊 Ranking results using {model_name}...")
    reranker_model = FlagReranker(model_name, use_fp16=True)
    pairs = processed_list(state.query, state.candidate_papers)
    score = reranker_model.compute_score(pairs, normalize=True)
    ranked = sorted(zip(state.candidate_papers, score), key=lambda x: x[1], reverse=True)
    return {"ranked_papers": ranked}
