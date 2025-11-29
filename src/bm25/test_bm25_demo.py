"""
Interactive BM25 Evaluation Demo
Shows step-by-step what happens during evaluation
"""

import json
import re
import bm25s
import Stemmer
import difflib
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
DATA_FILE = "datasets/scholar_copilot_eval_data_1k.json"
TOP_K = 10
MAX_PAPERS = 3  # Limit to first 3 papers for demo
MAX_QUERIES = 5  # Limit to first 5 queries for demo

# -----------------------------
# 1️⃣ LOAD DATA
# -----------------------------
print("=" * 80)
print("📂 STEP 1: Loading dataset...")
print("=" * 80)
data_path = Path(__file__).parent.parent.parent / DATA_FILE
with open(data_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"✅ Loaded {len(dataset)} papers")
print(f"🔬 Using first {MAX_PAPERS} papers for demo\n")

# Limit dataset for demo
dataset = dataset[:MAX_PAPERS]

# -----------------------------
# 2️⃣ BUILD CORPUS
# -----------------------------
print("=" * 80)
print("🧱 STEP 2: Building reference corpus...")
print("=" * 80)

corpus_texts, corpus_titles, corpus_ids = [], [], []

for paper_idx, paper in enumerate(dataset):
    bib = paper.get("bib_info", {})
    paper_title = paper.get("title", "Unknown")
    print(f"\n📄 Paper {paper_idx + 1}: {paper_title[:60]}...")

    ref_count = 0
    for cite_tag, refs in bib.items():
        for ref in refs:
            title = ref.get("title", "") or ""
            abstract = ref.get("abstract", "") or ""
            text = (title + " " + abstract).strip()
            if text:
                corpus_texts.append(text)
                corpus_titles.append(title.strip())
                corpus_ids.append(ref.get("citation_key", cite_tag))
                ref_count += 1

    print(f"   Added {ref_count} references to corpus")

print(f"\n✅ Total reference entries indexed: {len(corpus_texts)}")
print(f"   Sample titles:")
for i, title in enumerate(corpus_titles[:5], 1):
    print(f"   {i}. {title[:70]}...")

# -----------------------------
# 3️⃣ EXTRACT CITATION CONTEXTS
# -----------------------------
print("\n" + "=" * 80)
print("🔍 STEP 3: Extracting citation contexts...")
print("=" * 80)


def extract_contexts(paper_text):
    """Extract short contexts around each <|cite_X|> tag."""
    contexts = []
    for match in re.finditer(r"<\|cite_(\d+)\|>", paper_text):
        start, end = match.span()
        context = paper_text[max(0, start - 100) : min(len(paper_text), end + 100)]
        contexts.append((match.group(1), context))
    return contexts


query_contexts, query_true_titles, query_paper_ids = [], [], []

for paper_idx, paper in enumerate(dataset):
    paper_text = paper.get("paper", "")
    bib = paper.get("bib_info", {})
    paper_title = paper.get("title", "Unknown")

    for cite_idx, context in extract_contexts(paper_text):
        tag = f"<|cite_{cite_idx}|>"
        refs = bib.get(tag, [])
        if not refs:
            continue
        true_title = refs[0].get("title", "")
        query_contexts.append(context)
        query_true_titles.append(true_title)
        query_paper_ids.append(paper_idx)

print(f"✅ Extracted {len(query_contexts)} citation contexts")
print(f"🔬 Using first {min(MAX_QUERIES, len(query_contexts))} queries for demo\n")

# Limit queries for demo
query_contexts = query_contexts[:MAX_QUERIES]
query_true_titles = query_true_titles[:MAX_QUERIES]
query_paper_ids = query_paper_ids[:MAX_QUERIES]

# Show sample contexts
print("📝 Sample citation contexts:")
for i, (ctx, true_title) in enumerate(zip(query_contexts[:3], query_true_titles[:3]), 1):
    print(f"\n   Query {i}:")
    print(f"   Context: ...{ctx[:100]}...")
    print(f"   True Title: {true_title[:70]}...")

# -----------------------------
# 4️⃣ BUILD BM25 INDEX
# -----------------------------
print("\n" + "=" * 80)
print("🔨 STEP 4: Building BM25 index...")
print("=" * 80)

stemmer = Stemmer.Stemmer("english")
print("   Tokenizing corpus...")
corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=stemmer)
print("   Indexing documents...")
retriever = bm25s.BM25()
retriever.index(corpus_tokens)
print(f"✅ BM25 index built successfully ({len(corpus_texts)} documents)")


# -----------------------------
# 5️⃣ FUZZY MATCHING FUNCTION
# -----------------------------
def match_title(true_title, retrieved_titles, threshold=0.85):
    """Check if true title matches any retrieved title (fuzzy)."""
    true_lower = true_title.lower()
    best_match = None
    best_ratio = 0.0

    for t in retrieved_titles:
        ratio = difflib.SequenceMatcher(None, true_lower, t.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = t

    return best_ratio > threshold, best_match, best_ratio


# -----------------------------
# 6️⃣ RETRIEVE & EVALUATE (WITH DETAILED OUTPUT)
# -----------------------------
print("\n" + "=" * 80)
print("🎯 STEP 5: Retrieval & Evaluation (Real-time)")
print("=" * 80)

results = []
hits_at_5 = 0
hits_at_10 = 0
reciprocal_ranks = []

for query_idx, (context, true_title) in enumerate(zip(query_contexts, query_true_titles), 1):
    print(f"\n{'─' * 80}")
    print(f"🔍 QUERY {query_idx}/{len(query_contexts)}")
    print(f"{'─' * 80}")
    print(f"📝 Context: ...{context[:120]}...")
    print(f"✅ True Title: {true_title}")

    # Tokenize and retrieve
    query_tokens = bm25s.tokenize(context, stopwords="en", stemmer=stemmer)
    doc_ids, scores = retriever.retrieve(query_tokens, k=TOP_K)
    doc_ids, scores = doc_ids[0], scores[0]

    retrieved_titles = [corpus_titles[doc_id] for doc_id in doc_ids]

    # Check for match
    hit_fuzzy, best_match, best_ratio = match_title(true_title, retrieved_titles)

    print(f"\n📊 Top {TOP_K} Retrieved Results:")
    print(f"{'Rank':<6} {'BM25 Score':<12} {'Title':<60} {'Match':<8}")
    print(f"{'-' * 90}")

    found_rank = None
    for rank, (doc_id, score) in enumerate(zip(doc_ids, scores), start=1):
        retrieved_title = corpus_titles[doc_id]
        # Check if this specific title matches
        ratio = difflib.SequenceMatcher(None, true_title.lower(), retrieved_title.lower()).ratio()
        is_match = ratio > 0.85

        if is_match and found_rank is None:
            found_rank = rank
            match_indicator = f"✅ {ratio:.2f}"
        else:
            match_indicator = f"❌ {ratio:.2f}" if rank <= 5 else ""

        title_display = (
            retrieved_title[:57] + "..." if len(retrieved_title) > 60 else retrieved_title
        )
        print(f"{rank:<6} {float(score):<12.4f} {title_display:<60} {match_indicator:<8}")

    # Evaluation results
    print(f"\n🎯 Evaluation:")
    if hit_fuzzy:
        print(f"   ✅ MATCH FOUND! (Best similarity: {best_ratio:.3f})")
        if found_rank:
            print(f"   📍 Found at rank {found_rank}")
            if found_rank <= 5:
                hits_at_5 += 1
            if found_rank <= 10:
                hits_at_10 += 1
            reciprocal_ranks.append(1.0 / found_rank)
            print(f"   📈 MRR contribution: {1.0 / found_rank:.4f}")
    else:
        print(f"   ❌ NO MATCH (Best similarity: {best_ratio:.3f}, threshold: 0.85)")
        reciprocal_ranks.append(0.0)

    # Store results
    for rank, (doc_id, score) in enumerate(zip(doc_ids, scores), start=1):
        results.append(
            {
                "query_id": query_idx - 1,
                "rank": rank,
                "bm25_score": float(score),
                "fuzzy_hit": hit_fuzzy,
            }
        )

# -----------------------------
# 7️⃣ FINAL METRICS
# -----------------------------
print("\n" + "=" * 80)
print("📊 FINAL EVALUATION METRICS")
print("=" * 80)

total_queries = len(query_contexts)
recall_at_5 = hits_at_5 / total_queries if total_queries > 0 else 0
recall_at_10 = hits_at_10 / total_queries if total_queries > 0 else 0
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

print(f"\n📈 Results Summary:")
print(f"   Total Queries: {total_queries}")
print(f"   Hits in Top-5:  {hits_at_5}")
print(f"   Hits in Top-10: {hits_at_10}")
print(f"\n🎯 Metrics:")
print(f"   Recall@5:  {recall_at_5:.3f} ({hits_at_5}/{total_queries})")
print(f"   Recall@10: {recall_at_10:.3f} ({hits_at_10}/{total_queries})")
print(f"   MRR:        {mrr:.3f}")

print("\n" + "=" * 80)
print("✅ Demo Complete!")
print("=" * 80)
