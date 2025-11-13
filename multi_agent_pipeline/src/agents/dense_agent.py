from langgraph.graph import MessagesState

def dense_agent(state: MessagesState):
    query = getattr(state, "query", None)

    # placeholder until we embed with E5
    results = [
        {"title": f"E5 result for {query}", "score": 0.83},
        {"title": f"E5 alt result for {query}", "score": 0.79}
    ]

    return {"dense_results": results}
