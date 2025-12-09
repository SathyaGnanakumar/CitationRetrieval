# src/agents/bm25_agent.py

import json
import bm25s
import Stemmer
from langchain_core.messages import AIMessage

# --------------------------------------
# 1. Load corpus once (fast + shared)
# --------------------------------------

import os
from dotenv import load_dotenv

load_dotenv()

DATA_FILE = os.getenv("DATASET_DIR")

corpus_texts = []
corpus_titles = []

print("🔧 Loading BM25 corpus...")

with open(DATA_FILE, "r") as f:
    data = json.load(f)

for entry in data:
    bib = entry["bib_info"]
    for tag, entries in bib.items():
        for e in entries:
            title = e.get("title", "")
            abstract = e.get("abstract", "")

            if title:
                corpus_titles.append(title)
                corpus_texts.append(f"{title} {abstract}")

# --------------------------------------
# 2. Preprocess using SAME baseline stemmer
# --------------------------------------

stemmer = Stemmer.Stemmer("english")

tokenized_corpus = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=stemmer)

# --------------------------------------
# 3. Build BM25 retriever (shared index)
# --------------------------------------

bm25 = bm25s.BM25()
bm25.index(tokenized_corpus)

print(f"📚 Indexed {len(corpus_titles)} documents for BM25.")


# --------------------------------------
# 4. LangGraph Agent
# --------------------------------------


def bm25_agent(state):
    """
    BM25 retrieval agent.
    Expects coordinator to send:
        {"queries": [...]}  inside the last AIMessage.
    Returns top-5 document titles.
    """

    # Get last message content (the coordinator routing dict)
    msg = state["messages"][-1].content

    # Parse JSON dictionary inside the message
    try:
        routing = eval(msg) if isinstance(msg, str) else msg
    except:
        return {"messages": [AIMessage(content="BM25_ERROR: Could not parse coordinator message.")]}

    # Extract expanded queries
    queries = routing.get("queries", [])
    if not queries:
        return {"messages": [AIMessage(content="BM25 received no queries")]}

    q0 = queries[0]  # BM25 only uses first query for now

    # -----------------------------------
    # BM25 SCORING (same as baseline)
    # -----------------------------------

    query_tokens = bm25s.tokenize(q0, stopwords="en", stemmer=stemmer)

    doc_ids, scores = bm25.retrieve(query_tokens, k=5)
    doc_ids = doc_ids[0]  # unwrap list-of-lists
    scores = scores[0]

    results = [
        {"title": corpus_titles[i], "score": float(scores[idx])} for idx, i in enumerate(doc_ids)
    ]

    # -----------------------------------
    # Return formatted result as AIMessage
    # -----------------------------------

    return {"messages": [AIMessage(content=str(results), name="bm25")]}
