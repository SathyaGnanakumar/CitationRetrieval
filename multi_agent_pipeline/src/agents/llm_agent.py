from langgraph.graph import MessagesState

def llm_agent(state: MessagesState):
    query = getattr(state, "query", None)


    # Final version: call LLM with tool for Semantic Scholar, etc.
    # Current stub:
    results = [
        {"title": f"LLM Agent reasoning result for {query}", "score": 0.92},
        {"title": f"LLM Agent secondary result for {query}", "score": 0.64}
    ]

    return {"llmagent_results": results}
