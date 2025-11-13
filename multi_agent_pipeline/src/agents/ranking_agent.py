from langgraph.graph import MessagesState

# src/agents/ranking_agent.py
def ranking_agent(state: MessagesState):
    print("📊 Ranking results...")
    ranked = sorted(state.candidate_papers)
    return {"ranked_papers": ranked}
