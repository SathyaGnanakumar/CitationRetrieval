
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.models import DenseRetrievalModel, BM25Model

def create_dummy_data(num_docs=100, num_queries=10):
    """Create dummy corpus and queries."""
    corpus = []
    for i in range(num_docs):
        corpus.append({
            'title': f"Paper Title {i}",
            'abstract': f"This is the abstract for paper {i}. It contains some keywords like machine learning and retrieval.",
            'text': f"Paper Title {i} This is the abstract for paper {i}. It contains some keywords like machine learning and retrieval."
        })
    
    queries = [f"query about machine learning {i}" for i in range(num_queries)]
    return queries, corpus

def test_dense_optimization():
    print("\n🧪 Testing Dense Retrieval Optimization...")
    
    # Initialize model (use small model for speed)
    try:
        model = DenseRetrievalModel(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    except Exception as e:
        print(f"Skipping Dense test due to model load error: {e}")
        return

    queries, corpus = create_dummy_data(num_docs=50, num_queries=20)
    
    # 1. Test Correctness
    print("   Verifying correctness...")
    results_sequential = [model.retrieve(q, corpus, k=5) for q in queries]
    results_batch = model.retrieve_batch(queries, corpus, k=5)
    
    # Check if results match
    mismatch = 0
    for i in range(len(queries)):
        seq_titles = [r['title'] for r in results_sequential[i]]
        batch_titles = [r['title'] for r in results_batch[i]]
        if seq_titles != batch_titles:
            mismatch += 1
            
    if mismatch == 0:
        print("   ✅ Batch results match sequential results exactly.")
    else:
        print(f"   ❌ Mismatch in {mismatch} queries!")

    # 2. Test Speedup
    print("   Measuring speedup...")
    # Increase data for meaningful timing
    queries, corpus = create_dummy_data(num_docs=200, num_queries=50)
    
    # Sequential
    start = time.time()
    for q in queries:
        model.retrieve(q, corpus, k=5)
    seq_time = time.time() - start
    
    # Batch
    start = time.time()
    model.retrieve_batch(queries, corpus, k=5)
    batch_time = time.time() - start
    
    print(f"   Sequential time: {seq_time:.4f}s")
    print(f"   Batch time:      {batch_time:.4f}s")
    print(f"   Speedup:         {seq_time / batch_time:.2f}x")

def test_bm25_optimization():
    print("\n🧪 Testing BM25 Optimization...")
    model = BM25Model()
    
    queries, corpus = create_dummy_data(num_docs=500, num_queries=50)
    
    # 1. Test Correctness
    print("   Verifying correctness...")
    results_sequential = [model.retrieve(q, corpus, k=5) for q in queries]
    results_batch = model.retrieve_batch(queries, corpus, k=5)
    
    mismatch = 0
    for i in range(len(queries)):
        seq_titles = [r['title'] for r in results_sequential[i]]
        batch_titles = [r['title'] for r in results_batch[i]]
        # BM25 scores might have slight float diffs, but ranks should be stable for distinct scores
        if seq_titles != batch_titles:
             # Check if scores are very close
            seq_scores = [r['score'] for r in results_sequential[i]]
            batch_scores = [r['score'] for r in results_batch[i]]
            if not np.allclose(seq_scores, batch_scores):
                mismatch += 1
            
    if mismatch == 0:
        print("   ✅ Batch results match sequential results.")
    else:
        print(f"   ❌ Mismatch in {mismatch} queries!")

    # 2. Test Speedup
    print("   Measuring speedup...")
    queries, corpus = create_dummy_data(num_docs=1000, num_queries=100)
    
    # Sequential
    start = time.time()
    for q in queries:
        model.retrieve(q, corpus, k=5)
    seq_time = time.time() - start
    
    # Batch
    start = time.time()
    model.retrieve_batch(queries, corpus, k=5)
    batch_time = time.time() - start
    
    print(f"   Sequential time: {seq_time:.4f}s")
    print(f"   Batch time:      {batch_time:.4f}s")
    print(f"   Speedup:         {seq_time / batch_time:.2f}x")

if __name__ == "__main__":
    test_dense_optimization()
    test_bm25_optimization()
