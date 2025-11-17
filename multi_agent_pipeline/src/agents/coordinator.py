# src/agents/coordinator.py

import ast
from langchain_core.messages import AIMessage

def coordinator(state):
    # from reformulator agent
    msg = state["messages"][-1].content

    try:
        queries = ast.literal_eval(msg)
    except:
        queries = [msg]

    routing = {
        "run_bm25": True,
        "run_dense": True,
        "run_specter": True,
        "run_cite": True,
        "queries": queries
    }

    return {
        "messages": [
            AIMessage(
                name="coordinator",
                content=str(routing)
            )
        ]
    }
