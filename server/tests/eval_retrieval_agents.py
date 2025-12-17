import sys
from pathlib import Path
import json
import re
import time
from typing import Dict, List, Any

import bm25s
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Fix Python path
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.resources.builders import (
    build_bm25_resources,
    build_e5_resources,
    build_specter_resources,
)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DATA_FILE = "../datasets/scholar_copilot_eval_data_1k.json"
WINDOW = 200
KS = (1, 5, 10)
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

E5_BATCH_SIZE = 32
SPECTER_BATCH_SIZE = 32
SPECTER_MAX_LENGTH = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# Load dataset (JSON or JSONL)
# ------------------------------------------------------------------
with open(DATA_FILE) as f:
    content = f.read().strip()
    raw_data = json.loads(content) if content.startswith("[") else [
        json.loads(line) for line in content.splitlines() if line.strip()
    ]

print(f"Loaded {len(raw_data)} source papers")

# ------------------------------------------------------------------
# 🔑 Build mapping: paper_id → citation_corpus_id
# ------------------------------------------------------------------
paper_to_corpus_id: Dict[str, str] = {}

for entry in raw_data:
    for records in entry["bib_info"].values():
        for rec in records:
            cid = rec.get("citation_corpus_id")
            if cid:
                paper_to_corpus_id[entry["paper_id"]] = cid

print(f"Mapped {len(paper_to_corpus_id)} papers to citation_corpus_id")

# ------------------------------------------------------------------
# Build corpus docs (ID-ALIGNED)
# ------------------------------------------------------------------
docs: List[Dict[str, Any]] = []

for entry in raw_data:
    cid = paper_to_corpus_id.get(entry["paper_id"])
    if cid is None:
        continue

    docs.append({
        "id": cid,                       # 🔑 CRITICAL FIX
        "title": entry.get("title", ""),
        "text": entry["paper"],
    })

print(f"Built corpus with {len(docs)} documents")

# ------------------------------------------------------------------
# Citation extractor (correct schema)
# ------------------------------------------------------------------
CITE_PATTERN = re.compile(r"<\|cite_(\d+)\|>")

def extract_examples(entry: Dict[str, Any], window: int):
    text = entry["paper"]
    bib = entry["bib_info"]
    examples = []

    for match in CITE_PATTERN.finditer(text):
        cite_token = match.group(0)
        if cite_token not in bib:
            continue

        gold_ids = [
            rec["citation_corpus_id"]
            for rec in bib[cite_token]
            if "citation_corpus_id" in rec
        ]
        if not gold_ids:
            continue

        s, e = match.span()
        ctx = (text[max(0, s - window):s] + text[e:e + window]).strip()
        if ctx:
            examples.append({
                "query": ctx,
                "gold_ids": gold_ids,
            })

    return examples

eval_examples = []
for entry in raw_data:
    eval_examples.extend(extract_examples(entry, WINDOW))

if not eval_examples:
    raise RuntimeError("No evaluation queries extracted")

queries = [e["query"] for e in eval_examples]
gold_ids_list = [e["gold_ids"] for e in eval_examples]
N = len(eval_examples)

print(f"Evaluating on {N} citation queries")

# ------------------------------------------------------------------
# Recall helper (multi-gold)
# ------------------------------------------------------------------
def recall_at_k(retrieved_ids, gold_ids, k):
    return int(any(g in retrieved_ids[:k] for g in gold_ids))

# ------------------------------------------------------------------
# Build retrieval resources (timed)
# ------------------------------------------------------------------
latency = {}

t0 = time.time()
bm25_res = build_bm25_resources(docs)
latency["BM25_build"] = time.time() - t0

t0 = time.time()
e5_res = build_e5_resources(docs)
latency["E5_build"] = time.time() - t0

t0 = time.time()
specter_res = build_specter_resources(docs)
specter_res["model"].to(DEVICE)
latency["SPECTER_build"] = time.time() - t0

# ------------------------------------------------------------------
# BM25 Evaluation
# ------------------------------------------------------------------
print("Evaluating BM25...")
t0 = time.time()

bm25_hits = {k: 0 for k in KS}
for ex in eval_examples:
    q_tokens = bm25s.tokenize(
        ex["query"], stopwords="en", stemmer=bm25_res["stemmer"]
    )
    doc_ids, _ = bm25_res["index"].retrieve(q_tokens, k=max(KS))
    retrieved = [bm25_res["ids"][i] for i in doc_ids[0]]
    for k in KS:
        bm25_hits[k] += recall_at_k(retrieved, ex["gold_ids"], k)

latency["BM25_eval"] = time.time() - t0

# ------------------------------------------------------------------
# E5 Evaluation (BATCHED)
# ------------------------------------------------------------------
print("Evaluating E5 (batched)...")
t0 = time.time()

with torch.no_grad():
    q_embs = e5_res["model"].encode(
        queries,
        batch_size=E5_BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

scores = q_embs @ e5_res["corpus_embeddings"].T

e5_hits = {k: 0 for k in KS}
for i in range(N):
    topk = torch.topk(scores[i], k=max(KS)).indices.tolist()
    retrieved = [e5_res["ids"][j] for j in topk]
    for k in KS:
        e5_hits[k] += recall_at_k(retrieved, gold_ids_list[i], k)

latency["E5_eval"] = time.time() - t0

# ------------------------------------------------------------------
# SPECTER Evaluation (BATCHED + GPU)
# ------------------------------------------------------------------
print("Evaluating SPECTER (batched, GPU)...")
t0 = time.time()

q_embs = []
with torch.no_grad():
    for i in range(0, N, SPECTER_BATCH_SIZE):
        batch = queries[i:i + SPECTER_BATCH_SIZE]
        inputs = specter_res["tokenizer"](
            batch,
            padding=True,
            truncation=True,
            max_length=SPECTER_MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        emb = specter_res["model"](**inputs).last_hidden_state.mean(dim=1)
        q_embs.append(emb.cpu())

q_embs = torch.nn.functional.normalize(torch.cat(q_embs), dim=1)
corpus_embs = torch.nn.functional.normalize(
    specter_res["corpus_embeddings"], dim=1
)

scores = q_embs @ corpus_embs.T

specter_hits = {k: 0 for k in KS}
for i in range(N):
    topk = torch.topk(scores[i], k=max(KS)).indices.tolist()
    retrieved = [specter_res["ids"][j] for j in topk]
    for k in KS:
        specter_hits[k] += recall_at_k(retrieved, gold_ids_list[i], k)

latency["SPECTER_eval"] = time.time() - t0

# ------------------------------------------------------------------
# Aggregate Metrics
# ------------------------------------------------------------------
results = {
    "BM25": bm25_hits,
    "E5": e5_hits,
    "SPECTER": specter_hits,
}

recall = {
    name: {f"Recall@{k}": results[name][k] / N for k in KS}
    for name in results
}

print("\n=== FINAL RESULTS ===")
for name, scores in recall.items():
    print(name, scores)

# ------------------------------------------------------------------
# Save Tables
# ------------------------------------------------------------------
recall_df = pd.DataFrame(recall).T
recall_df.to_csv(OUT_DIR / "retrieval_recall_table.csv")

lat_df = pd.DataFrame.from_dict(latency, orient="index", columns=["seconds"])
lat_df.to_csv(OUT_DIR / "retrieval_latency.csv")

# ------------------------------------------------------------------
# Latency vs Recall Plot (Recall@5)
# ------------------------------------------------------------------
plt.figure()
methods = ["BM25", "E5", "SPECTER"]
x = [latency[f"{m}_eval"] for m in methods]
y = [recall[m]["Recall@5"] for m in methods]

plt.scatter(x, y)
for i, m in enumerate(methods):
    plt.annotate(m, (x[i], y[i]))

plt.xlabel("Evaluation Latency (seconds)")
plt.ylabel("Recall@5")
plt.title("Latency vs Recall@5")
plt.grid(True)

plt.savefig(OUT_DIR / "latency_vs_recall.png", dpi=300)
plt.close()

print("\nSaved:")
print(" - retrieval_recall_table.csv")
print(" - retrieval_latency.csv")
print(" - latency_vs_recall.png")
