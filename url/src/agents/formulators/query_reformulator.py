# src/agents/query_reformulator.py
"""
Query reformulator with support for both rule-based and optimized DSPy versions.

The workflow routes to either:
- query_reformulator_default: Rule-based keyword expansion
- query_reformulator_optimized: DSPy-based with evolution support
"""

import logging
import os
import re
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

ACADEMIC_EXPANSIONS = {
    "transformer": ["self-attention", "encoder-decoder", "attention mechanism"],
    "efficient": ["lightweight", "mobile inference", "low-compute", "compressed models"],
    "retrieval": ["information retrieval", "citation matching", "document ranking"],
    "model": ["architecture", "neural network", "deep learning model"],
    "citation": ["scientific reference", "scholarly work", "academic publication"],
}


def extract_keywords(query: str):
    tokens = re.findall(r"[A-Za-z]+", query.lower())
    return [t for t in tokens if len(t) > 3]


def expand_keywords(keywords):
    expansions = []
    for kw in keywords:
        if kw in ACADEMIC_EXPANSIONS:
            expansions.extend(ACADEMIC_EXPANSIONS[kw])
    return expansions


def academic_style_rewrite(query, keywords, expansions):
    joined = ", ".join(keywords + expansions[:3])
    return f"paper discussing {joined} in the context of citation retrieval"


def _get_user_query(state):
    """Extract user query from messages."""
    msgs = state.get("messages", [])
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            return m.content
    return None


def query_reformulator(state):
    """
    Default rule-based reformulator.
    Reads the LAST human message from MessagesState.
    """
    logger.info("🔄 Query reformulator (default) starting...")

    user_msg = _get_user_query(state)
    if not user_msg:
        logger.error("❌ No HumanMessage found")
        return {"messages": [AIMessage(name="reformulator", content="[]")]}

    base_query = user_msg.strip()
    logger.info(f"Original query: {base_query}")

    keywords = extract_keywords(base_query)
    expansions = expand_keywords(keywords)

    expanded_queries = [
        base_query,
        " ".join(keywords + expansions),
        academic_style_rewrite(base_query, keywords, expansions),
        f"{base_query} scientific paper architecture analysis",
    ]

    logger.info(f"✅ Generated {len(expanded_queries)} query variations")
    for i, q in enumerate(expanded_queries, 1):
        logger.debug(f"  {i}. {q[:80]}...")

    return {
        "query": base_query,
        "queries": expanded_queries,
        "messages": [AIMessage(name="reformulator", content=str(expanded_queries))],
    }


def query_reformulator_optimized(state):
    """
    Optimized DSPy-based reformulator.
    Uses evolution-trained module if available, otherwise falls back to default.
    """
    logger.info("🔄 Query reformulator (optimized) starting...")

    user_msg = _get_user_query(state)
    if not user_msg:
        logger.error("❌ No HumanMessage found")
        return {"messages": [AIMessage(name="reformulator", content="[]")]}

    base_query = user_msg.strip()

    # Try to load optimized module from version tracker
    try:
        from src.agents.self_evolve.version_tracker import VersionTracker

        tracker = VersionTracker("reformulator")
        module = tracker.get_best()

        if module is not None:
            logger.info(f"✓ Using optimized reformulator v{tracker.get_best_version_number()}")

            # Configure DSPy
            import dspy

            resources = state.get("resources", {}) or {}
            lm = resources.get("dspy_lm")
            if lm is None:
                model = os.getenv("DSPY_MODEL", "gpt-4o-mini")
                lm = dspy.LM(model=model, temperature=0.0, max_tokens=500)
            dspy.configure(lm=lm)

            # Run optimized module
            result = module(query=base_query)
            expanded_queries = getattr(result, "query_list", None) or [base_query]

            # Ensure original query is included
            if base_query not in expanded_queries:
                expanded_queries.insert(0, base_query)

            logger.info(f"✅ Optimized reformulator generated {len(expanded_queries)} queries")

            return {
                "query": base_query,
                "queries": expanded_queries,
                "reformulator_version": tracker.get_best_version_number(),
                "messages": [
                    AIMessage(name="reformulator_optimized", content=str(expanded_queries))
                ],
            }

    except Exception as e:
        logger.warning(f"Could not use optimized reformulator: {e}, falling back to default")

    # Fall back to default rule-based reformulator
    logger.info("Falling back to default reformulator")
    return query_reformulator(state)
