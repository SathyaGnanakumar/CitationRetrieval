"""
DSPy optimization metrics for self-evolving agents.

Metrics are used by optimizers to evaluate and improve module performance.
"""

from typing import List, Set, Optional
import dspy


def citation_retrieval_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Optional[str] = None
) -> float:
    """
    Metric for citation retrieval optimization.
    
    Combines multiple signals:
    - Correct selection (40% weight)
    - Recall@5 (30% weight)
    - Recall@10 (20% weight)
    - MRR (10% weight)
    
    Args:
        example: DSPy Example with ground truth
        prediction: DSPy Prediction from module
        trace: Optional trace information
        
    Returns:
        Score between 0.0 and 1.0
    """
    # Get ground truth
    ground_truth_ids = set(getattr(example, 'ground_truth_ids', []))
    
    if not ground_truth_ids:
        return 0.0
    
    # Get predicted/selected paper
    selected_paper = getattr(prediction, 'selected_paper', None)
    
    # Handle different prediction formats
    if isinstance(selected_paper, dict):
        selected_id = selected_paper.get('id')
    elif hasattr(selected_paper, 'id'):
        selected_id = selected_paper.id
    else:
        selected_id = None
    
    # Check if selected paper is correct
    correct_selection = 1.0 if selected_id in ground_truth_ids else 0.0
    
    # Check ranking quality (if we have ranked papers)
    ranked_papers = getattr(prediction, 'ranked_papers', [])
    
    if ranked_papers:
        # Extract IDs from ranked papers (handle tuples and dicts)
        retrieved_ids = []
        for p in ranked_papers[:20]:
            if isinstance(p, tuple):
                paper = p[0]
            else:
                paper = p
            
            if isinstance(paper, dict):
                pid = paper.get('id')
            elif hasattr(paper, 'id'):
                pid = paper.id
            else:
                pid = None
            
            if pid:
                retrieved_ids.append(pid)
        
        # Calculate Recall@5
        recall_5 = len(set(retrieved_ids[:5]) & ground_truth_ids) / len(ground_truth_ids)
        
        # Calculate Recall@10
        recall_10 = len(set(retrieved_ids[:10]) & ground_truth_ids) / len(ground_truth_ids)
        
        # Calculate MRR
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in ground_truth_ids:
                mrr = 1.0 / rank
                break
        
        # Weighted combination
        score = (
            0.4 * correct_selection +  # 40% weight on final selection
            0.3 * recall_5 +            # 30% on Recall@5
            0.2 * recall_10 +           # 20% on Recall@10
            0.1 * mrr                   # 10% on MRR
        )
    else:
        # If no ranked papers, only use correct selection
        score = correct_selection
    
    return score


def query_reformulation_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Optional[str] = None
) -> float:
    """
    Metric for query reformulation quality.
    
    Measures diversity and quality of reformulated queries.
    
    Args:
        example: DSPy Example with input query
        prediction: DSPy Prediction with reformulated queries
        trace: Optional trace information
        
    Returns:
        Score between 0.0 and 1.0
    """
    queries = getattr(prediction, 'queries', [])
    
    if not queries:
        return 0.0
    
    # Measure diversity: reward unique queries
    unique_queries = set(q.lower().strip() for q in queries if isinstance(q, str))
    diversity_score = min(1.0, len(unique_queries) / 4.0)  # Target: 4 unique queries
    
    # Measure length: queries should be substantial (at least 5 words)
    avg_length = sum(len(q.split()) for q in queries) / len(queries)
    length_score = min(1.0, avg_length / 10.0)  # Target: 10 words average
    
    # Combined score
    score = 0.7 * diversity_score + 0.3 * length_score
    
    return score


def retrieval_quality_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Optional[str] = None
) -> float:
    """
    General retrieval quality metric.
    
    Focus on Recall@K and MRR.
    
    Args:
        example: DSPy Example with ground truth
        prediction: DSPy Prediction with retrieved results
        trace: Optional trace information
        
    Returns:
        Score between 0.0 and 1.0
    """
    ground_truth_ids = set(getattr(example, 'ground_truth_ids', []))
    
    if not ground_truth_ids:
        return 0.0
    
    # Get retrieved papers
    retrieved = getattr(prediction, 'retrieved_papers', [])
    ranked_papers = getattr(prediction, 'ranked_papers', retrieved)
    
    if not ranked_papers:
        return 0.0
    
    # Extract IDs
    retrieved_ids = []
    for p in ranked_papers[:20]:
        if isinstance(p, tuple):
            paper = p[0]
        else:
            paper = p
        
        if isinstance(paper, dict):
            pid = paper.get('id')
        elif hasattr(paper, 'id'):
            pid = paper.id
        else:
            pid = None
        
        if pid:
            retrieved_ids.append(pid)
    
    if not retrieved_ids:
        return 0.0
    
    # Calculate Recall@10
    recall_10 = len(set(retrieved_ids[:10]) & ground_truth_ids) / len(ground_truth_ids)
    
    # Calculate MRR
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            mrr = 1.0 / rank
            break
    
    # Weighted combination
    score = 0.7 * recall_10 + 0.3 * mrr
    
    return score
