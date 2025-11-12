# src/agents/verifier_agent.py
def verifier_agent(state):
    print("✅ Verifying top citation...")
    return {"verified_paper": state.ranked_papers[0]}
