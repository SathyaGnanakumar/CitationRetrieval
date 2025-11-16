## ANALYSIS AGENT OUTLINE:

from langgraph.graph import MessagesState

def analysis_agent(state: MessagesState):
    query = getattr(state, "query", None)

    # placeholder
    results = [
        {"title": f"Analysis Agent hit for: {query}", "score": 10.0},
        {"title": f"Analysis Agent alt hit for: {query}", "score": 9.5}
    ]

    return {"analysis_agent_results": results}