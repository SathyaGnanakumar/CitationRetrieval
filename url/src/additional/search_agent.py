# src/agents/search_agent.py
def search_agent(state):
    print("🔍 Searching for papers...")
    results = [f"Paper about {q}" for q in state.expanded_queries]
    return {"candidate_papers": results}
