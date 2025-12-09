# orchestrator.py

import os
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END

# Load environment variables
load_dotenv()

# Import agent functions
from src.agents.coordinator import coordinator

# Import Retriever agent functions
from src.agents.retrievers.bm25_agent import bm25_agent
from src.agents.retrievers.e5_agent import e5_agent
from src.agents.retrievers.specter_agent import specter_agent

# from src.agents.llm_agent import llm_agent
from src.agents.formulators.query_reformulator import query_reformulator
from src.agents.formulators.analysis_agent import analysis_agent
from src.agents.formulators.reranker import reranker

# from agents.verifier_agent import verifier_agent


class RetrievalWorkflow:
    def __init__(self):
        """Initialize the workflow and build the graph pipeline."""
        self.pipeline = self._build_workflow()

    ############################################################
    # 1️⃣  BUILD THE STATE GRAPH (use MessagesState!)
    ############################################################

    def _build_workflow(self):
        """Build and compile the LangGraph workflow."""
        # IMPORTANT: Studio REQUIRES MessagesState or Annotated fields
        graph = StateGraph(MessagesState)

        ############################################################
        # 2️⃣  ADD AGENTS (nodes)
        ############################################################

        graph.add_node("reformulator", query_reformulator)
        # graph.add_node("coordinator", coordinator)
        graph.add_node("bm25", bm25_agent)
        graph.add_node("e5", e5_agent)
        graph.add_node("specter", specter_agent)
        # graph.add_node("llm", llm_agent, tags=["retriever"])
        graph.add_node("analysis", analysis_agent)  # Placeholder analysis agent
        # graph.add_node("verifier", verifier_agent, tags=["agent"])
        graph.add_node("reranking", reranker)

        ############################################################
        # 3️⃣  ADD EDGES
        ############################################################

        # Start → Query Reformulator -> Coordinator
        graph.add_edge(START, "reformulator")
        # graph.add_edge("reformulator", "coordinator")

        # Coordinator fans out to all retrieval agents
        graph.add_edge("reformulator", "bm25")
        graph.add_edge("reformulator", "e5")
        graph.add_edge("reformulator", "specter")
        # graph.add_edge("coordinator", "llm")

        # All retrieval agents → analysis agent
        graph.add_edge("bm25", "analysis")
        graph.add_edge("e5", "analysis")
        graph.add_edge("specter", "analysis")
        # graph.add_edge("llm", "analysis")

        # Analysis -> Verifier agent -> END
        graph.add_edge("analysis", "reranking")
        graph.add_edge("reranking", END)
        # graph.add_edge("verifier", END)

        ############################################################
        # 4️⃣ COMPILE AND RETURN PIPELINE
        ############################################################

        return graph.compile()

    ############################################################
    # 5️⃣ RUN METHOD - Takes initial state, returns final state
    ############################################################

    def run(self, initial_state: MessagesState) -> MessagesState:
        """
        Execute the workflow pipeline with an initial state.

        Args:
            initial_state: The initial state containing messages and any other state data

        Returns:
            The final state after the pipeline completes execution
        """
        final_state = self.pipeline.invoke(initial_state)
        return final_state

    def get_pipeline(self):
        return self.pipeline

    ############################################################
    # 6️⃣ UTILITY METHODS
    ############################################################

    @staticmethod
    def pretty_print_messages(output):
        """Pretty print the messages from pipeline output."""
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

    def visualize_graph(self, filename: str = "workflow_graph.png"):
        """
        Visualize the workflow graph as a Mermaid diagram and save to file.

        Args:
            filename: Name of the output file (default: "workflow_graph.png")

        Returns:
            IPython.display.Image object that can be displayed in Jupyter notebooks
        """
        from IPython.display import Image

        # Get output directory from environment variable
        output_dir = os.getenv("GRAPH_OUTPUT_DIR", "./graphs")
        output_path = Path(output_dir)

        # Create directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate graph image
        graph_image = self.pipeline.get_graph().draw_mermaid_png()

        # Save to file
        file_path = output_path / filename
        with open(file_path, "wb") as f:
            f.write(graph_image)

        print(f"Graph visualization saved to: {file_path}")

        return Image(graph_image)
