from langgraph.graph import MessagesState

def coordinator(state):
    # Safely extract the query
    query = getattr(state, "query", None)
    if not query:
        query = ""  # avoid KeyError cases

    return {
        "bm25_query": query,
        "e5_query": query,
        "specter_query": query,
        "cite_query": query,
    }
