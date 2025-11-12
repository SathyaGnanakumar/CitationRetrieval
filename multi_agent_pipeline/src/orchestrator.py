from langgraph.graph import StateGraph, START, END
from src.state import PipelineState
from src.agents.query_reformulator import query_reformulator
from src.agents.search_agent import search_agent
from src.agents.ranking_agent import ranking_agent
from src.agents.verifier_agent import verifier_agent
from src.agents.coordinator import coordinator

graph = StateGraph(PipelineState)

# Add all agents as nodes
graph.add_node("coordinator", coordinator, tags=["agent"])
graph.add_node("query_reformulator", query_reformulator, tags=["agent"])
graph.add_node("search_agent", search_agent, tags=["agent"])
graph.add_node("ranking_agent", ranking_agent, tags=["agent"])
graph.add_node("verifier_agent", verifier_agent, tags=["agent"])

# Define the pipeline order
graph.add_edge(START, "coordinator")
graph.add_edge("coordinator", "query_reformulator")
graph.add_edge("query_reformulator", "search_agent")
graph.add_edge("search_agent", "ranking_agent")
graph.add_edge("ranking_agent", "verifier_agent")
graph.add_edge("verifier_agent", END)

app = graph.compile()

if __name__ == "__main__":
    out = app.invoke({"query": "Transformer models for efficient citation retrieval"})
    print(out)
