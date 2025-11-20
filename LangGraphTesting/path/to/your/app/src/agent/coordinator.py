# coordinator.py

def coordinator(state: dict):
    """
    Coordinator Agent:
    - decomposes user query into sub tasks
    - orchestrates which retrieval paths to run
    - returns short structured plan (markov memory)
    """

    query = state.get("query")

    # Simple heuristic for now
    # later we upgrade this to a model-based planner
    if len(query) < 250:
        plan = ["bm25_first", "dense_second", "rerank_final"]
    else:
        plan = ["dense_first", "bm25_support", "rerank_final"]

    return {
        "plan": plan,
        "query": query,
        "memory": state.get("memory", {})  # markov short memory
    }
