# test.py
from langgraph.graph import StateGraph, START, END

def step1(state):
    return {"value": state.get("value", 0) + 1}

def step2(state):
    return {"value": state["value"] * 10}

graph = StateGraph(dict)

# label nodes so studio knows to display them
graph.add_node("step1", step1, tags=["agent"])
graph.add_node("step2", step2, tags=["agent"])

graph.add_edge(START, "step1")
graph.add_edge("step1", "step2")
graph.add_edge("step2", END)

app = graph.compile()

if __name__ == "__main__":
    out = app.invoke({"value": 5})
    print(out)
