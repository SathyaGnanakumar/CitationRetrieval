# src/agents/bm25_agent.py

import json
import bm25s
import Stemmer
from langchain_core.messages import AIMessage

DATA_FILE = "../datasets/scholar_copilot_eval_data_1k.json"

corpus_texts = []
corpus_titles = []
corpus_abstracts = []  

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
                corpus_abstracts.append(abstract)
                corpus_texts.append(f"{title} {abstract}")

stemmer = Stemmer.Stemmer("english")

tokenized_corpus = bm25s.tokenize(
    corpus_texts,
    stopwords="en",
    stemmer=stemmer
)

bm25 = bm25s.BM25()
bm25.index(tokenized_corpus)

print(f"📚 Indexed {len(corpus_titles)} documents for BM25.")


def bm25_agent(state):

    msg = state["messages"][-1].content

    try:
        routing = eval(msg) if isinstance(msg, str) else msg
    except:
        return {
            "messages": [
                AIMessage(content="BM25_ERROR: Could not parse coordinator message.")
            ]
        }

    queries = routing.get("queries", [])
    if not queries:
        return {"messages": [AIMessage(content="BM25 received no queries")]}

    q0 = queries[0]

    query_tokens = bm25s.tokenize(
        q0,
        stopwords="en",
        stemmer=stemmer
    )

    doc_ids, scores = bm25.retrieve(query_tokens, k=5)
    doc_ids = doc_ids[0]
    scores = scores[0]

    results = []
    for idx, i in enumerate(doc_ids):
        results.append({
            "doc_id": int(i),
            "title": corpus_titles[i],
            #"abstract": corpus_abstracts[i],  
            "score": float(scores[idx])
        })

    return {
        "messages": [
            AIMessage(
                name="bm25",
                content=json.dumps(results)
            )
        ],
        "bm25_results": results     
    }
