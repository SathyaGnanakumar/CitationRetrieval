from langgraph.graph import MessagesState

def cite_agent(state: MessagesState):
    query = getattr(state, "query", None)


    # Final version: call LLM with tool for Semantic Scholar, etc.
    # Current stub:
    results = [
        {"title": f"CiteAgent reasoning result for {query}", "score": 0.92},
        {"title": f"CiteAgent secondary result for {query}", "score": 0.64}
    ]

    return {"citeagent_results": results}
