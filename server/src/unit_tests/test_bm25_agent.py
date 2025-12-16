from server.src.agents.retrievers.bm25_agent import bm25_agent

ACADEMIC_EXPANSIONS = {
    "transformer": ["self-attention", "encoder-decoder", "attention mechanism"],
    "efficient": ["lightweight", "mobile inference", "low-compute", "compressed models"],
    "retrieval": ["information retrieval", "citation matching", "document ranking"],
    "model": ["architecture", "neural network", "deep learning model"],
    "citation": ["scientific reference", "scholarly work", "academic publication"],
}

base_query = "hello, world!"
keywords = ["hello", "world"]

# utilities from query_reformulator.py
def expand_keywords(keywords):
    expansions = []
    for kw in keywords:
        if kw in ACADEMIC_EXPANSIONS:
            expansions.extend(ACADEMIC_EXPANSIONS[kw])
    return expansions

def academic_style_rewrite(query, keywords, expansions):
    joined = ", ".join(keywords + expansions[:3])
    return f"paper discussing {joined} in the context of citation retrieval"

def test_bm25_returns_dict():
    state = {"query": base_query,
             "queries": [
                 "hello, world!",
                 " ".join(keywords + expand_keywords(keywords)),
                 academic_style_rewrite(base_query, keywords, expand_keywords(keywords)),
                 f"{base_query} scientific paper architecture analysis",
                ]}

    new_state = bm25_agent(state)
    
    # tests that the returned state is a dictionary
    assert isinstance(new_state, dict)
    
def test_bm25_has_all_parts():
    state = {"query": base_query,
             "queries": [
                 "hello, world!",
                 " ".join(keywords + expand_keywords(keywords)),
                 academic_style_rewrite(base_query, keywords, expand_keywords(keywords)),
                 f"{base_query} scientific paper architecture analysis",
                ]}

    new_state = bm25_agent(state)
    
    # tests that the returned dict has all requirements
    assert isinstance(new_state, dict)
    assert "bm25results" in new_state
    assert "messages" in new_state

def test_bm25_results_contain_all_details():
    state = {"query": base_query,
             "queries": [
                 "hello, world!",
                 " ".join(keywords + expand_keywords(keywords)),
                 academic_style_rewrite(base_query, keywords, expand_keywords(keywords)),
                 f"{base_query} scientific paper architecture analysis",
                ]}

    new_state = bm25_agent(state)
    
    # tests that the returned results dict has all requirements
    assert isinstance(new_state, dict)
    assert "bm25results" in new_state
    assert isinstance(new_state["bm25results"], list)
    for elt in new_state["bm25results"]:
        assert "id" in elt
        assert "title" in elt
        assert "score" in elt
        assert "source" in elt
        assert "source" == "bm25"