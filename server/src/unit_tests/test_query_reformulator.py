from server.src.agents.formulators.query_reformulator import query_reformulator

def test_query_reformulator_is_dict():
    state = {"messages": "hello, world!"}

    new_state = query_reformulator(state)
    
    # tests that the returned state is a dictionary
    assert isinstance(new_state, dict)

def test_query_reformulator_includes_all_parts():
    state = {"messages": "hello, world!"}

    new_state = query_reformulator(state)
    
    # tests that the returned state includes all parts
    assert isinstance(new_state, dict)
    assert "query" in new_state
    assert "queries" in new_state
    assert "messages" in new_state

def test_query_reformulator_includes_original_query():
    state = {"messages": "hello, world!"}

    new_state = query_reformulator(state)
    
    # tests that the returned state includes the original query
    assert isinstance(new_state, dict)
    assert "query" in new_state
    
def test_query_reformulator_original_query_matches():
    state = {"messages": "hello, world!"}

    new_state = query_reformulator(state)
    
    # tests that the returned original query is actually equal
    assert isinstance(new_state, dict)
    assert "query" in new_state
    assert state["messages"] == new_state["query"]
    
def test_query_reformulator_adds_expanded_query():
    state = {"messages": "hello, world!"}

    new_state = query_reformulator(state)
    
    # tests that the returned state includes the expanded query
    assert isinstance(new_state, dict)
    assert "queries" in new_state
    
def test_query_reformulator_expanded_query_is_expanded():
    state = {"messages": "hello, world!"}

    new_state = query_reformulator(state)
    
    # tests that the expanded query includes all the proper keys
    assert isinstance(new_state, dict)
    assert "queries" in new_state
    assert "transformer" in new_state["queries"]
    assert "efficient" in new_state["queries"]
    assert "retrieval" in new_state["queries"]
    assert "model" in new_state["queries"]
    assert "citation" in new_state["queries"]