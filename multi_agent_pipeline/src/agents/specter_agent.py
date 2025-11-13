from langgraph.graph import MessagesState

def specter_agent(state: MessagesState):
    query = getattr(state, "query", None)

    # placeholder — use your real specter2 embeddings later
    results = [
        {"title": f"Specter2 result for {query}", "score": 0.77},
        {"title": f"Specter2 alt result for {query}", "score": 0.71}
    ]

    return {"specter_results": results}
