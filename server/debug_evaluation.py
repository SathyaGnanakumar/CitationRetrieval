#!/usr/bin/env python3
"""
Debug script to diagnose evaluation issues.
Checks for ID format mismatches and ground truth problems.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus_loaders.scholarcopilot import load_dataset, build_citation_corpus
from evaluate import get_ground_truth_ids, extract_query_from_paper
from src.resources.builders import build_inmemory_resources
from src.resources.cache import load_resources

load_dotenv()


def debug_single_example():
    """Debug a single example in detail."""
    print("=" * 80)
    print("EVALUATION DEBUG - Single Example")
    print("=" * 80)

    # Load dataset
    dataset_path = os.getenv("DATASET_DIR")
    if not dataset_path:
        print("❌ Error: DATASET_DIR not set")
        return

    print(f"\n📚 Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"   ✓ Loaded {len(dataset)} papers")

    # Extract first citation context (proper evaluation)
    import re

    cite_pattern = re.compile(r"<\|cite_\d+\|>")

    citation_example = None
    for paper in dataset[:10]:
        paper_text = paper.get("paper", "")
        if not paper_text:
            continue

        bib_info = paper.get("bib_info", {})

        # Find first citation marker
        for match in cite_pattern.finditer(paper_text):
            cite_token = match.group(0)

            if cite_token not in bib_info:
                continue

            refs = bib_info[cite_token]
            if not refs:
                continue

            # Get ground truth for this specific citation
            relevant_ids = set()
            for ref in refs:
                # Use citation_key (or paper_id as fallback) - this matches the corpus builder!
                # See: corpus_loaders/scholarcopilot/loader.py line 84-86
                ref_id = ref.get("citation_key") or ref.get("paper_id")
                if ref_id:
                    relevant_ids.add(str(ref_id))

            if not relevant_ids:
                continue

            # Extract local context around citation
            pos = match.start()
            words_before = paper_text[:pos].split()[-100:]
            words_after = paper_text[match.end() :].split()[:100]

            context = " ".join(words_before + words_after)
            context = re.sub(r"<\|cite_\d+\|>", "", context)
            context = " ".join(context.split())

            if len(context.split()) < 10:
                continue

            citation_example = {
                "query": context,
                "relevant_ids": relevant_ids,
                "paper": paper,
                "cite_token": cite_token,
            }
            break

        if citation_example:
            break

    if not citation_example:
        print("❌ No citations found in first 10 papers")
        return

    paper = citation_example["paper"]
    query = citation_example["query"]
    relevant_ids = citation_example["relevant_ids"]

    print(f"\n📄 Paper: {paper.get('title', 'No title')[:80]}...")
    print(f"   ID: {paper.get('paper_id', 'unknown')}")
    print(f"   Citation: {citation_example['cite_token']}")

    print(f"\n✅ Ground Truth for This Citation: {len(relevant_ids)}")
    print(f"   IDs:")
    for i, gid in enumerate(list(relevant_ids), 1):
        print(f"   {i}. {gid!r} (type: {type(gid).__name__})")

    # Build resources
    print(f"\n🔧 Building resources...")
    resources = load_resources(dataset_path, enable_bm25=True, enable_e5=True, enable_specter=True)

    if not resources:
        corpus = build_citation_corpus(dataset)
        resources = build_inmemory_resources(
            corpus, enable_bm25=True, enable_e5=True, enable_specter=True
        )

    print(f"   ✓ Resources ready")

    # Check corpus IDs
    print(f"\n📦 Corpus IDs (from BM25 index):")
    bm25_ids = resources.get("bm25", {}).get("ids", [])
    print(f"   Total corpus docs: {len(bm25_ids)}")
    print(f"   Sample corpus IDs (first 5):")
    for i, cid in enumerate(bm25_ids[:5], 1):
        print(f"   {i}. {cid!r} (type: {type(cid).__name__}, len: {len(str(cid))})")

    # Run BM25 retrieval
    print(f"\n🔍 Running BM25 retrieval...")
    print(f"   Query (citation context, {len(query.split())} words):")
    print(f"   {query[:150]}...")

    import bm25s

    bm25_res = resources.get("bm25")
    q_tokens = bm25s.tokenize(query, stopwords="en", stemmer=bm25_res["stemmer"])
    doc_indices, scores = bm25_res["index"].retrieve(q_tokens, k=20)

    retrieved_ids = [bm25_res["ids"][idx] for idx in doc_indices[0]]

    print(f"\n📊 Retrieved IDs (top 20):")
    for i, rid in enumerate(retrieved_ids[:5], 1):
        print(f"   {i}. {rid!r} (type: {type(rid).__name__}, len: {len(str(rid))})")

    # Check for matches
    print(f"\n🎯 ID Matching Analysis:")
    print(f"   Ground truth IDs: {len(relevant_ids)}")
    print(f"   Retrieved IDs: {len(retrieved_ids)}")

    # Exact match
    exact_matches = set(retrieved_ids[:20]) & relevant_ids
    print(f"\n   ✓ Exact matches in top-20: {len(exact_matches)}")
    if exact_matches:
        print(f"     Examples: {list(exact_matches)[:3]}")

    # Check if formats differ
    print(f"\n🔬 Format Analysis:")

    # Try string conversion
    retrieved_str = {str(rid) for rid in retrieved_ids[:20]}
    relevant_str = {str(gid) for gid in relevant_ids}
    str_matches = retrieved_str & relevant_str
    print(f"   As strings: {len(str_matches)} matches")
    num_str_matches = len(str_matches)

    # Try removing prefixes
    def normalize_id(id_val):
        """Try to normalize ID by removing common prefixes."""
        s = str(id_val)
        # Remove common prefixes
        for prefix in ["arxiv-", "arxiv.", "paper-", "corpus-"]:
            if s.startswith(prefix):
                s = s[len(prefix) :]
        return s

    retrieved_norm = {normalize_id(rid) for rid in retrieved_ids[:20]}
    relevant_norm = {normalize_id(gid) for gid in relevant_ids}
    norm_matches = retrieved_norm & relevant_norm
    num_norm_matches = len(norm_matches)
    print(f"   Normalized (no prefix): {num_norm_matches} matches")
    if norm_matches:
        print(f"     Examples: {list(norm_matches)[:3]}")

    # Check if issue is in corpus
    print(f"\n🔍 Checking if ground truth papers are in corpus...")
    corpus_id_set = set(bm25_ids)
    gt_in_corpus_exact = len(relevant_ids & corpus_id_set)
    print(f"   Ground truth papers in corpus (exact): {gt_in_corpus_exact}/{len(relevant_ids)}")

    corpus_id_str = {str(cid) for cid in bm25_ids}
    gt_in_corpus_str = len(relevant_str & corpus_id_str)
    print(f"   Ground truth papers in corpus (as strings): {gt_in_corpus_str}/{len(relevant_ids)}")

    corpus_id_norm = {normalize_id(cid) for cid in bm25_ids}
    gt_in_corpus_norm = len(relevant_norm & corpus_id_norm)
    print(f"   Ground truth papers in corpus (normalized): {gt_in_corpus_norm}/{len(relevant_ids)}")

    # Recommendations
    print(f"\n💡 Diagnosis:")
    if exact_matches:
        print(f"   ✅ IDs match correctly - evaluation should work!")
        print(f"   ⚠️  Low recall might be real performance issue")
    elif num_str_matches > 0:
        print(f"   ⚠️  ID type mismatch (int vs string)")
        print(f"   🔧 Fix: Convert all IDs to strings in evaluation")
    elif num_norm_matches > 0:
        print(f"   ⚠️  ID format mismatch (different prefixes)")
        print(f"   🔧 Fix: Normalize IDs by removing prefixes")
    elif gt_in_corpus_exact == 0 and gt_in_corpus_norm > 0:
        print(f"   ⚠️  Ground truth papers ARE in corpus, but with different ID format")
        print(f"   🔧 Fix: Update get_ground_truth_ids() to match corpus ID format")
    elif gt_in_corpus_norm == 0:
        print(f"   ❌ Ground truth papers NOT in corpus!")
        print(f"   🔧 Fix: Check dataset loading and corpus building")
    else:
        print(f"   ❌ Ground truth paper IS in corpus but NOT being retrieved!")
        print(f"   🔧 This suggests:")
        print(f"      1. Query quality issue (query doesn't match paper content)")
        print(f"      2. Corpus text issue (paper text doesn't contain relevant terms)")
        print(f"      3. Tokenization/preprocessing mismatch")
        print(f"\n   📝 Debug suggestions:")
        print(f"      - Check what text is indexed for ground truth paper")
        print(f"      - Try searching for the paper title directly")
        print(f"      - Inspect BM25 scores (are they just too low?)")

    # Additional debugging: Check what's in the corpus for the ground truth paper
    if gt_in_corpus_exact > 0 and not exact_matches:
        print(f"\n🔍 Additional Investigation:")
        print(f"   Ground truth paper '{list(relevant_ids)[0]}' IS in corpus but NOT retrieved.")
        print(f"   Let's check what's indexed for it...")

        gt_id = list(relevant_ids)[0]
        if gt_id in corpus_id_set:
            idx = bm25_ids.index(gt_id)
            print(f"   Corpus index: {idx}")

            # Try to get the title if available
            titles = bm25_res.get("titles", [])
            if idx < len(titles):
                print(f"   Title: {titles[idx][:80]}...")

            # Show BM25 score for this paper
            import bm25s

            q_tokens = bm25s.tokenize(query, stopwords="en", stemmer=bm25_res["stemmer"])
            doc_indices, scores = bm25_res["index"].retrieve(q_tokens, k=len(bm25_ids))

            # Find position of ground truth paper
            gt_indices = [
                i for i, did in enumerate(doc_indices[0]) if bm25_res["ids"][did] == gt_id
            ]
            if gt_indices:
                gt_rank = gt_indices[0] + 1
                gt_score = float(scores[0][gt_indices[0]])
                print(f"   BM25 rank: {gt_rank}/{len(bm25_ids)}")
                print(f"   BM25 score: {gt_score:.4f}")
                print(f"   Top-1 score: {float(scores[0][0]):.4f}")
                print(
                    f"\n   💡 The ground truth paper IS being found, just ranked at position {gt_rank}!"
                )
                print(f"      This is a retrieval quality issue, not an ID mismatch.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    debug_single_example()
