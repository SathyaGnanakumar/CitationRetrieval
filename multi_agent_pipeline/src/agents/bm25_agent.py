from langgraph.graph import MessagesState

def bm25_agent(state: MessagesState):
    query = getattr(state, "query", None)

    # placeholder
    results = [
        {"title": f"BM25 hit for: {query}", "score": 10.0},
        {"title": f"BM25 alt hit for: {query}", "score": 9.5}
    ]

    return {"bm25_results": results}
