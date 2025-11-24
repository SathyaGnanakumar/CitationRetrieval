# specter_agent.py (SAFE FOR LANGGRAPH STUDIO)
from langgraph.graph import MessagesState
from transformers import AutoTokenizer, AutoModel
import json
import torch
import re
from langchain_core.messages import AIMessage

DATA_FILE = "../datasets/scholar_copilot_eval_data_1k.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = None
_model = None
_corpus_embeddings = None
_corpus_titles = None


def clean_text(text):
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


############################################################
# 1️⃣  Lazy-loading helper functions
############################################################

def load_data():
    global _corpus_titles

    if _corpus_titles is not None:
        return

    with open(DATA_FILE) as f:
        data = json.load(f)

    titles = []
    texts = []

    for paper in data:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        titles.append(title)
        texts.append(f"{title}. {abstract}")

    _corpus_titles = titles
    return texts


def load_model():
    global _tokenizer, _model

    if _model is not None:
        return

    model_name = "allenai/specter2_base"
    print(f"[SPECTER] Loading tokenizer/model on CPU: {model_name}")

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModel.from_pretrained(model_name)

    # ALWAYS LOAD ON CPU FIRST
    _model.to("cpu")


def build_corpus_embeddings(texts):
    global _corpus_embeddings

    if _corpus_embeddings is not None:
        return _corpus_embeddings

    print("[SPECTER] Encoding corpus (lazy, first-time only)...")

    # Move model to GPU temporarily if available
    if device == "cuda":
        _model.to("cuda")

    encodings = []
    batch_size = 8  # safe for 10GB GPU

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        if device == "cuda":
            inputs = inputs.to("cuda")

        with torch.no_grad():
            outputs = _model(**inputs).last_hidden_state.mean(dim=1)

        encodings.append(outputs.cpu())

    _corpus_embeddings = torch.cat(encodings, dim=0)

    # Move model *back to CPU* after embedding to keep GPU memory low
    _model.to("cpu")

    return _corpus_embeddings


############################################################
# 2️⃣  The actual LangGraph agent
############################################################

def specter_agent(state: MessagesState):
    global _corpus_embeddings

    # Lazy-load data + model
    load_model()
    corpus_texts = load_data()

    # Build corpus embeddings if missing
    embeddings = build_corpus_embeddings(corpus_texts)

    # Parse coordinator message
    msg = state["messages"][-1].content
    routing = eval(msg) if isinstance(msg, str) else msg
    queries = routing.get("queries", [])

    if not queries:
        return {"messages": [AIMessage(content="SPECTER received no queries")]}

    query = queries[0]

    # Encode query on GPU if available
    if device == "cuda":
        _model.to("cuda")

    inputs = _tokenizer(
        [query],
        padding=True, truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    if device == "cuda":
        inputs = inputs.to("cuda")

    with torch.no_grad():
        q_emb = _model(**inputs).last_hidden_state.mean(dim=1).cpu()

    # Move model back to CPU (free GPU)
    _model.to("cpu")

    # Compute similarity
    scores = torch.cosine_similarity(q_emb, embeddings)[0]
    top_indices = torch.topk(scores, k=5).indices.tolist()

    results = [
        {"title": _corpus_titles[i], "score": float(scores[i])}
        for i in top_indices
    ]

    return {
        "messages": [
            AIMessage(content=str(results), name="specter2")
        ]
    }
