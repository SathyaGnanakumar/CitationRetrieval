# orchestrator.py
"""
LangGraph workflow with conditional routing for self-evolution.

When evolution is enabled, the workflow uses conditional edges to route
to optimized versions of the reformulator and picker nodes.
"""

import logging
import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Import Retriever agent functions
from src.agents.retrievers.bm25_agent import bm25_agent
from src.agents.retrievers.e5_agent import e5_agent
from src.agents.retrievers.specter_agent import specter_agent

# Import Formulator agent functions (default and optimized versions)
from src.agents.formulators.query_reformulator import (
    query_reformulator,
    query_reformulator_optimized,
)
from src.agents.formulators.aggregator import aggregator
from src.agents.formulators.reranker import reranker
from src.agents.formulators.llm_agent import llm_reranker
from src.agents.formulators.dspy_picker import dspy_picker, dspy_picker_optimized
from src.models.state import RetrievalState


class RetrievalWorkflow:
    def __init__(self, use_llm_reranker: bool = False, enable_evolution: bool = False):
        """
        Initialize the workflow and build the graph pipeline.

        Args:
            use_llm_reranker: If True, use LLM reranker instead of cross-encoder reranker
            enable_evolution: If True, enable DSPy self-evolution features with conditional routing
        """
        self.use_llm_reranker = use_llm_reranker

        # Check evolution flag from environment if not specified
        if not enable_evolution:
            enable_evolution = os.getenv("ENABLE_DSPY_EVOLUTION", "false").lower() in {
                "true",
                "1",
                "yes",
                "on",
            }

        self.enable_evolution = enable_evolution
        self._optimized_modules = {}  # Store optimized modules

        if self.enable_evolution:
            logger.info("🔄 DSPy Evolution ENABLED - using conditional routing")

        self.pipeline = self._build_workflow()

    ############################################################
    # 1️⃣  ROUTING FUNCTIONS FOR CONDITIONAL EDGES
    ############################################################

    def _route_reformulator(
        self, state
    ) -> Literal["reformulator_default", "reformulator_optimized"]:
        """
        Route to optimized reformulator based on config flag or evolution setting.
        """
        config = state.get("config", {}) or {}

        # Check explicit config flag first (for manual control)
        use_dspy = config.get("use_dspy_reformulator", False) or config.get(
            "use_optimized_reformulator", False
        )

        if use_dspy:
            # User explicitly wants DSPy, try to use it
            try:
                from src.agents.self_evolve.version_tracker import VersionTracker

                tracker = VersionTracker("reformulator")
                if tracker.get_best() is not None:
                    return "reformulator_optimized"
            except Exception:
                pass
            # If no optimized version, still use optimized node (will fall back internally)
            return "reformulator_optimized"

        # Fall back to evolution-based routing
        if not self.enable_evolution:
            return "reformulator_default"

        if config.get("use_optimized_reformulator", True):
            try:
                from src.agents.self_evolve.version_tracker import VersionTracker

                tracker = VersionTracker("reformulator")
                if tracker.get_best() is not None:
                    return "reformulator_optimized"
            except Exception:
                pass

        return "reformulator_default"

    def _route_picker(self, state) -> Literal["picker_default", "picker_optimized"]:
        """
        Route to optimized picker based on config flag or evolution setting.
        """
        config = state.get("config", {}) or {}

        # Check explicit config flag first (for manual control)
        enable_dspy_picker = config.get("enable_dspy_picker", False) or config.get(
            "use_optimized_picker", False
        )

        if enable_dspy_picker:
            # User explicitly wants DSPy picker, try to use it
            try:
                from src.agents.self_evolve.version_tracker import VersionTracker

                tracker = VersionTracker("picker")
                if tracker.get_best() is not None:
                    return "picker_optimized"
            except Exception:
                pass
            # If no optimized version, still use optimized node (will fall back internally)
            return "picker_optimized"

        # Fall back to evolution-based routing
        if not self.enable_evolution:
            return "picker_default"

        if config.get("use_optimized_picker", True):
            try:
                from src.agents.self_evolve.version_tracker import VersionTracker

                tracker = VersionTracker("picker")
                if tracker.get_best() is not None:
                    return "picker_optimized"
            except Exception:
                pass

        return "picker_default"

    ############################################################
    # 2️⃣  BUILD THE STATE GRAPH
    ############################################################

    def _build_workflow(self):
        """Build and compile the LangGraph workflow with conditional routing."""
        graph = StateGraph(RetrievalState)

        ############################################################
        # 3️⃣  ADD NODES
        ############################################################

        # Reformulator nodes (default and optimized)
        graph.add_node("reformulator_default", query_reformulator)
        graph.add_node("reformulator_optimized", query_reformulator_optimized)

        # Retriever nodes
        graph.add_node("bm25", bm25_agent)
        graph.add_node("e5", e5_agent)
        graph.add_node("specter", specter_agent)

        # Aggregator
        graph.add_node("aggregator", aggregator)

        # Reranker (choose type based on init flag)
        if self.use_llm_reranker:
            logger.info("Using LLM-based reranker")
            graph.add_node("reranking", llm_reranker)
        else:
            logger.debug("Using cross-encoder reranker")
            graph.add_node("reranking", reranker)

        # Picker nodes (default and optimized)
        graph.add_node("picker_default", dspy_picker)
        graph.add_node("picker_optimized", dspy_picker_optimized)

        ############################################################
        # 4️⃣  ADD EDGES WITH CONDITIONAL ROUTING
        ############################################################

        # START → Reformulator (conditional: default or optimized)
        graph.add_conditional_edges(
            START,
            self._route_reformulator,
            {
                "reformulator_default": "reformulator_default",
                "reformulator_optimized": "reformulator_optimized",
            },
        )

        # Both reformulator variants fan-out to retrievers in parallel
        for reformulator_node in ["reformulator_default", "reformulator_optimized"]:
            graph.add_edge(reformulator_node, "bm25")
            graph.add_edge(reformulator_node, "e5")
            graph.add_edge(reformulator_node, "specter")

        # Aggregator (fan-in from all retrievers)
        graph.add_edge("bm25", "aggregator")
        graph.add_edge("e5", "aggregator")
        graph.add_edge("specter", "aggregator")

        # Aggregator → Reranking
        graph.add_edge("aggregator", "reranking")

        # Reranking → Picker (conditional: default or optimized)
        graph.add_conditional_edges(
            "reranking",
            self._route_picker,
            {
                "picker_default": "picker_default",
                "picker_optimized": "picker_optimized",
            },
        )

        # Both picker variants → END
        graph.add_edge("picker_default", END)
        graph.add_edge("picker_optimized", END)

        ############################################################
        # 5️⃣ COMPILE AND RETURN PIPELINE
        ############################################################

        return graph.compile()

    ############################################################
    # 5️⃣ RUN METHOD - Takes initial state, returns final state
    ############################################################

    def run(self, initial_state: RetrievalState) -> RetrievalState:
        """
        Execute the workflow pipeline with an initial state.

        Args:
            initial_state: The initial state containing messages and any other state data

        Returns:
            The final state after the pipeline completes execution
        """
        logger.info("🚀 Starting workflow execution...")
        logger.debug(f"Initial state keys: {list(initial_state.keys())}")

        final_state = self.pipeline.invoke(initial_state)

        logger.info("✅ Workflow execution completed")
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

    def visualize_graph(self, save_file: bool = False, filename: str = "workflow_graph.png"):
        """
        Visualize the workflow graph as a Mermaid diagram and save to file.

        Args:
            filename: Name of the output file (default: "workflow_graph.png")

        Returns:
            IPython.display.Image object that can be displayed in Jupyter notebooks
        """
        from IPython.display import Image

        # Generate graph image
        graph_image = self.pipeline.get_graph().draw_mermaid_png()

        if save_file:
            # Get output directory from environment variable
            output_dir = os.getenv("GRAPH_OUTPUT_DIR", "./graphs")
            output_path = Path(output_dir)

            # Create directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)

            # Save to file
            file_path = output_path / filename
            with open(file_path, "wb") as f:
                f.write(graph_image)

            print(f"Graph visualization saved to: {file_path}")

        else:
            return Image(graph_image)

    ############################################################
    # 7️⃣  SELF-EVOLUTION METHODS
    ############################################################

    def is_evolution_enabled(self) -> bool:
        """Check if DSPy evolution is enabled."""
        return self.enable_evolution

    def update_dspy_module(self, module_name: str, new_module) -> None:
        """
        Hot-swap a DSPy module.

        Args:
            module_name: Name of module ('picker', 'reformulator')
            new_module: New DSPy module instance
        """
        if not self.enable_evolution:
            logger.warning("Evolution not enabled, cannot update module")
            return

        self._optimized_modules[module_name] = new_module
        logger.info(f"✓ Updated module: {module_name}")

    def update_reformulator(self, new_reformulator) -> None:
        """
        Update query reformulator module.

        Args:
            new_reformulator: New reformulator module
        """
        self.update_dspy_module("reformulator", new_reformulator)

    def get_module_versions(self) -> dict:
        """
        Return current versions of all modules.

        Returns:
            Dict mapping module names to version info
        """
        if not self.enable_evolution:
            return {}

        versions = {}
        try:
            from src.agents.self_evolve.version_tracker import VersionTracker

            # Get picker version
            picker_tracker = VersionTracker("picker")
            picker_stats = picker_tracker.get_statistics()
            if picker_stats.get("count", 0) > 0:
                versions["picker"] = {
                    "current_version": picker_stats.get("latest_version"),
                    "best_version": picker_stats.get("best_version"),
                    "best_score": picker_stats.get("best_score"),
                }

            # Get reformulator version
            reformulator_tracker = VersionTracker("reformulator")
            reformulator_stats = reformulator_tracker.get_statistics()
            if reformulator_stats.get("count", 0) > 0:
                versions["reformulator"] = {
                    "current_version": reformulator_stats.get("latest_version"),
                    "best_version": reformulator_stats.get("best_version"),
                    "best_score": reformulator_stats.get("best_score"),
                }
        except Exception as e:
            logger.error(f"Could not get module versions: {e}")

        return versions
