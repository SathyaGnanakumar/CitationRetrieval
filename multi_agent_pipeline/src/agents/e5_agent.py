from langgraph.graph import MessagesState
from sentence_transformers import SentenceTransformer, util
import json
import torch
import re
from langchain_core.messages import AIMessage

DATA_FILE = "../datasets/scholar_copilot_eval_data_1k.json"

def clean_text(text):
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Extract papers from JSON
with open(DATA_FILE) as f:
    data = json.load(f)
print(f"Loaded {len(data)} papers from ScholarCopilot dataset.")

# Build corpus from papers' bibliographies.
corpus_ids = []
corpus_titles = []
corpus_abstracts = []
corpus_texts = []

"""
Note: Normally the corpus should be built from the citations in every paper's 
bib_info, but in this version I built it from just the papers since I'm testing
this locally. When we move this code to the cluster, the loop needs to be
changed to the loop inside this comment so that it iterates through every
citation in bib_info for every paper.

for paper in data:
    for bib in paper.get("bib_info", {}).values():
        for citation in bib:
            citation_key = citation.get("citation_key")
            title = citation.get("title", "")
            abstract = citation.get("abstract", "")

            corpus_ids.append(citation_key)
            corpus_titles.append(title)
            corpus_abstracts.append(abstract)
            corpus_texts.append(f"{title}. {abstract}")
"""

for paper in data:
    citation_key = paper.get("paper_id")
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    corpus_ids.append(citation_key)
    corpus_titles.append(title)
    corpus_abstracts.append(abstract)
    corpus_texts.append(f"{title}. {abstract}")

print(f"Corpus size: {len(corpus_texts)} papers")

# Initialize E5-Large Model
model_name = "intfloat/e5-large-v2"
print(f"Loading dense retriever model: {model_name}")
model = SentenceTransformer(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on device: {device}")

# Create embeddings of cited papers in corpus and query sentences
print("Creating corpus embeddings:")
corpus_embeddings = model.encode(
    corpus_texts,
    batch_size=16,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

def e5_agent(state: MessagesState):
    """
    E5 dense retrieval agent.
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
        return {
            "messages": [
                AIMessage(content="BM25_ERROR: Could not parse coordinator message.")
            ]
        }

    # Extract expanded queries
    queries = routing.get("queries", [])
    if not queries:
        return {"messages": [AIMessage(content="E5 received no queries")]}
    
    query_embedding = model.encode(
        queries[0],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Get top 5 results from corpus
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_5 = torch.topk(scores, k=5)

    results = []
    for _, (score, idx) in enumerate(zip(top_5.values, top_5.indices)):
        results.append({
            "title": corpus_titles[idx],
            "score": float(score),
        })

    # Return formatted result as AIMessage
    return {
        "messages": [
            AIMessage(
                content=str(results),
                name="e5"
            )
        ]
    }