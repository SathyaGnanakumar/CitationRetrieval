# orchestrator.py

from langgraph.graph import StateGraph, MessagesState, START, END

# Import agent functions
from src.agents.coordinator import coordinator
from src.agents.bm25_agent import bm25_agent
from src.agents.dense_agent import dense_agent
from src.agents.specter_agent import specter_agent
from src.agents.cite_agent import cite_agent


############################################################
# 1️⃣  BUILD THE STATE GRAPH (use MessagesState!)
############################################################

# IMPORTANT: Studio REQUIRES MessagesState or Annotated fields
graph = StateGraph(MessagesState)

############################################################
# 2️⃣  ADD AGENTS (nodes)
############################################################

graph.add_node("coordinator", coordinator, tags=["agent"])
graph.add_node("bm25", bm25_agent, tags=["retriever"])
graph.add_node("e5", dense_agent, tags=["retriever"])
graph.add_node("specter", specter_agent, tags=["retriever"])
graph.add_node("citeagent", cite_agent, tags=["retriever"])

############################################################
# 3️⃣  ADD EDGES
############################################################

# Start → Coordinator
graph.add_edge(START, "coordinator")

# Coordinator fans out to all retrieval agents
graph.add_edge("coordinator", "bm25")
graph.add_edge("coordinator", "e5")
graph.add_edge("coordinator", "specter")
graph.add_edge("coordinator", "citeagent")

# Each retrieval agent flows to END for now
graph.add_edge("bm25", END)
graph.add_edge("e5", END)
graph.add_edge("specter", END)
graph.add_edge("citeagent", END)

############################################################
# 4️⃣ EXPOSE PIPELINE FOR LANGGRAPH STUDIO
############################################################

pipeline = graph.compile()

############################################################
# 5️⃣ LOCAL TESTING
############################################################

if __name__ == "__main__":
    out = pipeline.invoke({
        "messages": [{"role": "user", "content": "Transformer models for citation retrieval"}]
    })
    print("\n=== Pipeline Output ===")
    print(out)
