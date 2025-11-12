# src/agents/ranking_agent.py
def ranking_agent(state):
    print("📊 Ranking results...")
    ranked = sorted(state.candidate_papers)
    return {"ranked_papers": ranked}
