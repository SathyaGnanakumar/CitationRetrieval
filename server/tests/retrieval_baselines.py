import sys
from pathlib import Path
import json
import re
import time
from typing import Dict, List, Any
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import bm25s

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.resources.builders import (
    build_bm25_resources,
    build_e5_resources,
    build_specter_resources,
)

DATA_FILE = "../datasets/scholar_copilot_eval_data_1k.json"

# Extract papers from JSON
with open(DATA_FILE) as f:
    data = json.load(f)
print(f"Loaded 1000 papers from ScholarCopilot dataset.")

# Text cleaning function
def clean_text(text):
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Title normalization function for accurate comparison
def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[{}]", "", title)
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()

# Capture local context and build queries from sentences which used citations
queries, true_titles = [], []
cite_pattern = re.compile(r"<\|cite_\d+\|>")

for d in tqdm(data, desc="Extracting citation contexts"):
    paper_text = clean_text(d.get("paper", ""))
    bib_info = d.get("bib_info", {})

    found_cites = cite_pattern.findall(paper_text)
    if not found_cites:
        continue

    for cite_token in found_cites:
        refs = bib_info.get(cite_token, [])
        if not refs:
            continue

        pos = paper_text.find(cite_token)
        if pos == -1:
            continue

        # Extract local window of 400 characters around the citation
        start, end = max(0, pos - 400), min(len(paper_text), pos + 400)
        queries.append(clean_text(paper_text[start:end]))
        true_titles.append(refs[0]["title"].strip("{} "))

print(f"Built {len(queries)} queries from sentences with contexts.")
if len(queries) == 0:
    raise RuntimeError("Error: No queries extracted")

# Save paper id, title, and text for builders to use
docs: List[Dict[str, Any]] = []
for paper in data:
    for bib in paper.get("bib_info", {}).values():
        for citation in bib:
            docs.append({
                "id": citation.get("citation_key"),
                "title": citation.get("title", ""),
                "text": citation.get("abstract", ""),
            })

# Set k below to get recall@k scores
k = 5

# BM25 Evaluation
bm25_results = build_bm25_resources(docs)
tokenized_queries = bm25s.tokenize(
    queries,
    stopwords="en",
    stemmer=bm25_results["stemmer"],
)
results = bm25_results["index"].retrieve(tokenized_queries, k=k)
bm25_topk_indices = results.documents
bm25_hits = []
for i, doc_indices in enumerate(bm25_topk_indices):
    correct_title = normalize_title(true_titles[i])
    retrieved_titles = [
        normalize_title(bm25_results["titles"][idx]) for idx in doc_indices
    ]
    bm25_hits.append(correct_title in retrieved_titles)
bm25_recall_at_k = sum(bm25_hits) / len(bm25_hits)
print(f"BM25 Recall@{k}: {bm25_recall_at_k:.4f}")

# E5 Evaluation
e5_results = build_e5_resources(docs)
e5_query_embeddings = e5_results["model"].encode(
    queries,                 
    convert_to_tensor=True,
    normalize_embeddings=True,
    batch_size=16
)
scores = e5_query_embeddings @ e5_results["corpus_embeddings"].T
e5_topk_indices = torch.topk(scores, k=k, dim=1).indices
e5_hits = []
for i, doc_indices in enumerate(e5_topk_indices):
    correct_title = normalize_title(true_titles[i])
    retrieved_titles = [
        normalize_title(e5_results["titles"][idx]) for idx in doc_indices
    ]
    e5_hits.append(correct_title in retrieved_titles)
e5_recall_at_k = sum(e5_hits) / len(e5_hits)
print(f"E5 Recall@{k}: {e5_recall_at_k:.4f}")

# -------------------------------
# SPECTER Evaluation
# -------------------------------
print("Evaluating SPECTER...")

specter_results = build_specter_resources(docs)

device = "cuda" if torch.cuda.is_available() else "cpu"
specter_results["model"].to(device)

specter_hits = []

with torch.no_grad():
    query_embeddings = []

    # Batch inference for queries
    for i in tqdm(range(0, len(queries), 16), desc="SPECTER query encoding"):
        batch_queries = queries[i:i + 16]
        inputs = specter_results["tokenizer"](
            batch_queries,
            padding=True,
            truncation=True,
            max_length=256,          # critical for speed
            return_tensors="pt"
        ).to(device)

        emb = specter_results["model"](**inputs).last_hidden_state.mean(dim=1)
        query_embeddings.append(emb.cpu())

    query_embeddings = torch.cat(query_embeddings, dim=0)

# Normalize embeddings
query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=1)
corpus_embeddings = torch.nn.functional.normalize(
    specter_results["corpus_embeddings"], dim=1
)

# Similarity search
scores = query_embeddings @ corpus_embeddings.T
topk_indices = torch.topk(scores, k=k, dim=1).indices

# Compute Recall@k
for i, doc_indices in enumerate(topk_indices):
    correct_title = normalize_title(true_titles[i])
    retrieved_titles = [
        normalize_title(specter_results["titles"][idx])
        for idx in doc_indices.tolist()
    ]
    specter_hits.append(correct_title in retrieved_titles)

specter_recall_at_k = sum(specter_hits) / len(specter_hits)
print(f"SPECTER Recall@{k}: {specter_recall_at_k:.4f}")