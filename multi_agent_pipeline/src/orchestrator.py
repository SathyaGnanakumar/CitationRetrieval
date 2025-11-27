# orchestrator.py

from langgraph.graph import StateGraph, MessagesState, START, END

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
# 1️⃣  BUILD THE STATE GRAPH (use MessagesState!)
############################################################

# IMPORTANT: Studio REQUIRES MessagesState or Annotated fields
graph = StateGraph(MessagesState)

############################################################
# 2️⃣  ADD AGENTS (nodes)
############################################################

graph.add_node("reformulator", query_reformulator, tags=["agent"])
graph.add_node("coordinator", coordinator, tags=["agent"])
graph.add_node("bm25", bm25_agent, tags=["retriever"])
graph.add_node("e5", dense_agent, tags=["retriever"])
graph.add_node("specter", specter_agent, tags=["retriever"])
graph.add_node("citeagent", cite_agent, tags=["retriever"])
graph.add_node("entityrecognition", entity_recognition_agent, tags=["agent"])
graph.add_node("analysis", analysis_agent, tags=["agent"])  # Placeholder analysis agent

############################################################
# 3️⃣  ADD EDGES
############################################################

# Start → Query Reformulator -> Coordinator
graph.add_edge(START, "reformulator")
graph.add_edge("reformulator", "coordinator")

# Coordinator fans out to all retrieval agents
graph.add_edge("coordinator", "bm25")
graph.add_edge("coordinator", "e5")
graph.add_edge("coordinator", "specter")
graph.add_edge("coordinator", "citeagent")

# All retrieval agents → entity recognition agent
graph.add_edge("bm25", "entityrecognition")
graph.add_edge("e5", "entityrecognition")
graph.add_edge("specter", "entityrecognition")
graph.add_edge("citeagent", "entityrecognition")

# Entity recognition agent → analysis agent
graph.add_edge("entityrecognition", "analysis")

# Analysis agent → End for now (placeholder)
graph.add_edge("analysis", END) 

############################################################
# 4️⃣ EXPOSE PIPELINE FOR LANGGRAPH STUDIO
############################################################

pipeline = graph.compile()

############################################################
# 5️⃣ LOCAL TESTING
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
        "messages": [{"role": "user", "content": "Building on top of them, modern state-of-the-art models, such as BERT<|cite_1|>, are able to learn powerful language representations from unlabeled text and even surpass the human performance on the challenging question answering task."}]
    })
    print("\n=== Pipeline Output ===")
    pretty_print_messages(out)




