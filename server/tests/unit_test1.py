# Tests 1 Query

from ..src.workflow import RetrievalWorkflow

# Example usage
if __name__ == "__main__":
    workflow = RetrievalWorkflow()

    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": "Automatic neural architecture search<|cite_2|> is a choice for high accuracy model design, but the massive search cost (GPU hours and $CO_2$ emission) raises severe environmental concerns<|cite_3|>, shown in \\fig{fig:teaser:b}.",
            }
        ]
    }

    final_state = workflow.run(initial_state)

    print("\n=== Pipeline Output ===")
    RetrievalWorkflow.pretty_print_messages(final_state)
