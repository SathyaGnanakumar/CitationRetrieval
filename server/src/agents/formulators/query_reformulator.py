# src/agents/query_reformulator.py

import re
from langchain_core.messages import HumanMessage, AIMessage

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


def query_reformulator(state):
    """Reads the LAST human message from MessagesState."""

    msgs = state["messages"]
    user_msg = None
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            user_msg = m.content
            break

    if not user_msg:
        # Nothing to reformulate
        return {"messages": [AIMessage(name="reformulator", content="[]")]}

    base_query = user_msg.strip()
    keywords = extract_keywords(base_query)
    expansions = expand_keywords(keywords)

    expanded_queries = [
        base_query,
        " ".join(keywords + expansions),
        academic_style_rewrite(base_query, keywords, expansions),
        f"{base_query} scientific paper architecture analysis",
    ]

    return {
        # Structured keys (preferred by downstream nodes)
        "query": base_query,
        "queries": expanded_queries,
        # Message for Studio visibility / backwards-compat
        "messages": [AIMessage(name="reformulator", content=str(expanded_queries))],
    }
