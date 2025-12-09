from src.workflow import RetrievalWorkflow


# Example usage
if __name__ == "__main__":
    workflow = RetrievalWorkflow()
    workflow.get_pipeline()
    workflow.visualize_graph()
