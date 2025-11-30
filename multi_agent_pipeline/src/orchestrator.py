# orchestrator.py

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from pydantic import Field
from typing import List, Dict, Any

# Import agent functions
from src.agents.coordinator import coordinator
from src.agents.bm25_agent import bm25_agent
from src.agents.dense_agent import dense_agent
from src.agents.specter_agent import specter_agent
from src.agents.cite_agent import cite_agent
from src.agents.query_reformulator import query_reformulator
from src.agents.analysis_agent import analysis_agent
from src.agents.entity_recognition_agent import entity_recognition_agent


############################################################
# 1️⃣  DEFINE GLOBAL STATE (extends MessagesState)
############################################################

class RetrievalState(MessagesState):
    bm25_results: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_docs: List[str] = Field(default_factory=list)


############################################################
# 2️⃣  BUILD THE STATE GRAPH
############################################################

graph = StateGraph(RetrievalState)

############################################################
# 3️⃣  ADD AGENTS (nodes)
############################################################

graph.add_node("reformulator", query_reformulator)
graph.add_node("coordinator", coordinator)

graph.add_node("bm25", bm25_agent)
graph.add_node("e5", dense_agent)
graph.add_node("specter", specter_agent)
graph.add_node("citeagent", cite_agent)

graph.add_node("entityrecognition", entity_recognition_agent)
graph.add_node("analysis", analysis_agent)

############################################################
# 4️⃣  ADD EDGES
############################################################

# Core query flow
graph.add_edge(START, "reformulator")
graph.add_edge("reformulator", "coordinator")

# Coordinator → all retrievers
graph.add_edge("coordinator", "bm25")
graph.add_edge("coordinator", "e5")
graph.add_edge("coordinator", "specter")
graph.add_edge("coordinator", "citeagent")

# All retrievers → entity-recognition
graph.add_edge("bm25", "entityrecognition")
graph.add_edge("e5", "entityrecognition")
graph.add_edge("specter", "entityrecognition")
graph.add_edge("citeagent", "entityrecognition")

# Entity recognition → final analysis
graph.add_edge("entityrecognition", "analysis")

# Analysis → End
graph.add_edge("analysis", END)

############################################################
# 5️⃣  COMPILE PIPELINE
############################################################

pipeline = graph.compile()

############################################################
# 6️⃣  LOCAL TESTING HARNESS
############################################################

def pretty_print_messages(output):
    print("\n==================== PIPELINE RESULTS ====================\n")
    messages = output.get("messages", [])

    for msg in messages:
        role = msg.__class__.__name__
        name = getattr(msg, "name", None)
        title = f"{role}" if not name else f"{role} ({name})"

        print(f"--- {title} ---")
        print(msg.content)
        print()

    print("==========================================================\n")


if __name__ == "__main__":
    out = pipeline.invoke({
        "messages": [{"role": "user", "content": "Building on top of them, modern state-of-the-art models, such as BERT<|cite_1|>, are able to learn powerful language representations from unlabeled text."}]
    })
    pretty_print_messages(out)
