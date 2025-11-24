from langgraph.graph import MessagesState
from transformers import AutoTokenizer, AutoModel
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

# Initialize SPECTER2 Model
model_name = "allenai/specter2_base"
print(f"Loading dense retriever model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on device: {device}")

def encode_texts(texts):
    # Tokenize texts and get embeddings
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Create embeddings of cited papers in corpus and query sentences
print("Creating corpus embeddings:")
corpus_embeddings = encode_texts(corpus_texts)

def specter_agent(state: MessagesState):
    """
    SPECTER2 dense retrieval agent.
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
        return {"messages": [AIMessage(content="BM25 received no queries")]}
    
    query_embedding = encode_texts(queries[0])

    # Get top 5 results from corpus
    scores = torch.cosine_similarity(query_embedding, corpus_embeddings)[0]
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
                name="specter2"
            )
        ]
    }