from langgraph.graph import Graph
from agents.verification import VerificationAgent
from agents.analysis import AnalysisAgent

class CiteAgent:
    def __init__(self):
        self.graph = Graph()
        self.graph.add_node("bm25", BM25Node())
        self.graph.add_node("dense", DenseNode())
        self.graph.add_node("analysis", AnalysisAgent())
        self.graph.add_node("verify", VerificationAgent())

        # flow: retrieval → analysis → verification
        self.graph.add_edge("bm25", "analysis")
        self.graph.add_edge("dense", "analysis")
        self.graph.add_edge("analysis", "verify")

    def run(self, query):
        bm25_out = self.graph.nodes["bm25"].run(query)
        dense_out = self.graph.nodes["dense"].run(query)

        merged = list(set(bm25_out + dense_out))
        ranked = self.graph.nodes["analysis"].run(query, merged)

        verified = []
        for title, score in ranked:
            paper = self.graph.nodes["verify"].verify(title)
            if paper:
                paper["relevance"] = score
                verified.append(paper)

        return verified
