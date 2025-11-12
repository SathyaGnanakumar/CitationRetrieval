# src/agents/query_reformulator.py
def query_reformulator(state):
    print("🧠 Reformulating query...")
    expanded = [state.query, f"{state.query} paper citation"]
    return {"expanded_queries": expanded}
