import sys
from pathlib import Path
import json
import argparse
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.formulators.reranker import reranker, processed_list


def clean_text(text):
    """Remove LaTeX commands and extra whitespace"""
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_query_from_paper(paper_text, max_length=200):
    """Extract a query from the paper text (first few sentences)"""
    sentences = paper_text.split(". ")[:2]
    query = ". ".join(sentences)
    if len(query) > max_length:
        query = query[:max_length]
    return clean_text(query)


def run_retriever(retriever_name, query, dataset_path):
    """Run the specified retriever and return candidate papers"""

    if retriever_name == "bm25":
        from src.agents.retrievers.bm25_agent import bm25, corpus_titles, corpus_texts, stemmer
        import bm25s

        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer)
        doc_ids, scores = bm25.retrieve(query_tokens, k=10)
        doc_ids = doc_ids[0]
        scores = scores[0]

        results = [
            {"title": corpus_titles[i], "score": float(scores[idx])}
            for idx, i in enumerate(doc_ids)
        ]

    elif retriever_name == "e5":
        from src.agents.retrievers.e5_agent import model, corpus_embeddings, corpus_titles
        from sentence_transformers import util
        import torch

        query_embedding = model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_10 = torch.topk(scores, k=10)

        results = [
            {"title": corpus_titles[idx], "score": float(score)}
            for score, idx in zip(top_10.values, top_10.indices)
        ]

    elif retriever_name == "specter":
        from src.agents.retrievers.specter_agent import (
            model, tokenizer, corpus_embeddings, corpus_titles, device
        )
        import torch

        if device == "cuda":
            model.to("cuda")

        inputs = tokenizer(
            [query], padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        if device == "cuda":
            inputs = inputs.to("cuda")

        with torch.no_grad():
            q_emb = model(**inputs).last_hidden_state.mean(dim=1).cpu()

        model.to("cpu")

        scores = torch.cosine_similarity(q_emb.expand_as(corpus_embeddings), corpus_embeddings, dim=1)
        top_indices = torch.topk(scores, k=10).indices.tolist()

        results = [
            {"title": corpus_titles[i], "score": float(scores[i])}
            for i in top_indices
        ]

    else:
        raise ValueError(f"Unknown retriever: {retriever_name}. Choose from: bm25, e5, specter")

    return results


def test_reranker_with_dataset(retriever_name, reranker_model, dataset_path, example_idx=0):
    """Test reranker using real dataset and retriever"""

    print("\n" + "="*70)
    print(f"Testing Reranker with Real Dataset")
    print("="*70)

    # Load dataset
    with open(dataset_path) as f:
        data = json.load(f)

    print(f"\n📚 Loaded {len(data)} papers from dataset")

    # Get example paper
    if example_idx >= len(data):
        print(f"⚠️  Example index {example_idx} out of range. Using index 0.")
        example_idx = 0

    example = data[example_idx]
    print(f"📄 Using example {example_idx}: {example['title']}")

    # Extract query from paper text
    query = extract_query_from_paper(example['paper'])
    print(f"\n🔍 Query (extracted from paper):\n   {query}")

    # Run retriever
    print(f"\n🔎 Running {retriever_name.upper()} retriever...")
    candidate_papers = run_retriever(retriever_name, query, dataset_path)

    print(f"\n📋 Retrieved {len(candidate_papers)} candidate papers:")
    for i, paper in enumerate(candidate_papers[:5], 1):
        print(f"   {i}. {paper['title'][:80]}... (score: {paper['score']:.4f})")

    # Create state for reranker
    class SimpleState:
        def __init__(self, query, papers):
            self.query = query
            self.candidate_papers = papers

    state = SimpleState(query, candidate_papers)

    # Run reranker
    print(f"\n🎯 Running reranker with model: {reranker_model}")
    result = reranker(state, model_name=reranker_model)

    print(f"\n✨ Top 5 papers after reranking:")
    for i, (paper, score) in enumerate(result['ranked_papers'][:5], 1):
        print(f"   {i}. {paper['title'][:80]}...")
        print(f"      Retriever score: {paper['score']:.4f} | Reranker score: {score:.4f}")

    # Verify results
    assert 'ranked_papers' in result
    assert len(result['ranked_papers']) == len(candidate_papers)

    print("\n✅ Test completed successfully!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Test reranker with different retrievers")
    parser.add_argument(
        "--retriever",
        type=str,
        default="bm25",
        choices=["bm25", "e5", "specter"],
        help="Which retriever to use (default: bm25)"
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
        help="Reranker model name (default: BAAI/bge-reranker-v2-m3)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/Users/ishaankalra/Dev/Retrieval/datasets/scholar_copilot_eval_data_1k.json",
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--example-idx",
        type=int,
        default=0,
        help="Which example from dataset to use (default: 0)"
    )

    args = parser.parse_args()

    # Run test
    test_reranker_with_dataset(
        retriever_name=args.retriever,
        reranker_model=args.reranker_model,
        dataset_path=args.dataset,
        example_idx=args.example_idx
    )


if __name__ == "__main__":
    main()
