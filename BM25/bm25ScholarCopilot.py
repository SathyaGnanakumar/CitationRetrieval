import json
import re
from tqdm import tqdm
import bm25s
import Stemmer
import pandas as pd
import difflib
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
DATA_FILE = "scholar_copilot_eval_data_1k.json"
OUTPUT_FILE = "bm25_scholarcopilot_eval_v2_results.csv"
TOP_K = 10

# -----------------------------
# 1️⃣ LOAD DATA
# -----------------------------
print("📂 Loading dataset...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)
print(f"✅ Loaded {len(dataset)} papers")

# -----------------------------
# 2️⃣ BUILD CORPUS (all cited papers)
# -----------------------------
print("🧱 Building reference corpus...")

corpus_texts, corpus_titles, corpus_ids = [], [], []

for paper in dataset:
    bib = paper.get("bib_info", {})
    for cite_tag, refs in bib.items():
        for ref in refs:
            title = ref.get("title", "") or ""
            abstract = ref.get("abstract", "") or ""
            text = (title + " " + abstract).strip()
            if text:
                corpus_texts.append(text)
                corpus_titles.append(title.strip())
                corpus_ids.append(ref.get("citation_key", cite_tag))

print(f"✅ Total reference entries indexed: {len(corpus_texts)}")

# -----------------------------
# 3️⃣ EXTRACT CITATION CONTEXTS
# -----------------------------
def extract_contexts(paper_text):
    """Extract short contexts around each <|cite_X|> tag."""
    contexts = []
    for match in re.finditer(r"<\|cite_(\d+)\|>", paper_text):
        start, end = match.span()
        context = paper_text[max(0, start - 100):min(len(paper_text), end + 100)]
        contexts.append((match.group(1), context))
    return contexts

query_contexts, query_true_titles = [], []

for paper in dataset:
    paper_text = paper.get("paper", "")
    bib = paper.get("bib_info", {})

    for cite_idx, context in extract_contexts(paper_text):
        tag = f"<|cite_{cite_idx}|>"
        refs = bib.get(tag, [])
        if not refs:
            continue
        true_title = refs[0].get("title", "")
        query_contexts.append(context)
        query_true_titles.append(true_title)

print(f"✅ Extracted {len(query_contexts)} citation contexts")

# -----------------------------
# 4️⃣ BUILD BM25 INDEX
# -----------------------------
stemmer = Stemmer.Stemmer("english")
corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=stemmer)
retriever = bm25s.BM25()
retriever.index(corpus_tokens)
print("✅ BM25 index built successfully")

# -----------------------------
# 5️⃣ FUZZY MATCHING FUNCTION
# -----------------------------
def match_title(true_title, retrieved_titles, threshold=0.85):
    true_lower = true_title.lower()
    for t in retrieved_titles:
        ratio = difflib.SequenceMatcher(None, true_lower, t.lower()).ratio()
        if ratio > threshold:
            return True
    return False

# -----------------------------
# 6️⃣ RETRIEVE & EVALUATE
# -----------------------------
results = []

for i, (context, true_title) in enumerate(tqdm(zip(query_contexts, query_true_titles), total=len(query_contexts))):
    query_tokens = bm25s.tokenize(context, stopwords="en", stemmer=stemmer)
    doc_ids, scores = retriever.retrieve(query_tokens, k=TOP_K)
    doc_ids, scores = doc_ids[0], scores[0]

    retrieved_titles = [corpus_titles[doc_id] for doc_id in doc_ids]
    hit_fuzzy = match_title(true_title, retrieved_titles)

    for rank, (doc_id, score) in enumerate(zip(doc_ids, scores), start=1):
        results.append({
            "query_id": i,
            "citation_context": context,
            "true_title": true_title,
            "retrieved_title": corpus_titles[doc_id],
            "retrieved_id": corpus_ids[doc_id],
            "bm25_score": float(score),
            "rank": rank,
            "fuzzy_hit": hit_fuzzy
        })

# -----------------------------
# 7️⃣ SAVE RESULTS
# -----------------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)
print(f"💾 Results saved to {OUTPUT_FILE}")

# -----------------------------
# 8️⃣ METRICS
# -----------------------------
recall_at_5 = df[df["rank"] <= 5].groupby("query_id")["fuzzy_hit"].max().mean()
recall_at_10 = df[df["rank"] <= 10].groupby("query_id")["fuzzy_hit"].max().mean()

df["reciprocal_rank"] = df.apply(lambda r: 1/r["rank"] if r["fuzzy_hit"] else 0, axis=1)
mrr = df.groupby("query_id")["reciprocal_rank"].max().mean()

print("\n📊 BM25 Evaluation Summary:")
print(f"🎯 Recall@5:  {recall_at_5:.3f}")
print(f"🎯 Recall@10: {recall_at_10:.3f}")
print(f"⭐ MRR:        {mrr:.3f}")

# -----------------------------
# 9️⃣ VISUALIZATION
# -----------------------------
plt.figure(figsize=(7,4))
plt.hist(df["bm25_score"], bins=50, color="skyblue", edgecolor="black")
plt.title("Distribution of BM25 Scores Across Retrieved Docs")
plt.xlabel("BM25 Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("bm25_score_distribution.png")
print("📈 Saved histogram: bm25_score_distribution.png")
