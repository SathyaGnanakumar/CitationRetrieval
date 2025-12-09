# specter_agent.py (SAFE FOR LANGGRAPH STUDIO)
from langgraph.graph import MessagesState
from transformers import AutoTokenizer, AutoModel
import json
import torch
import re
from langchain_core.messages import AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

DATA_FILE = os.getenv("DATASET_DIR")

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


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

for paper in data:
    citation_key = paper.get("paper_id")
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    corpus_ids.append(citation_key)
    corpus_titles.append(title)
    corpus_abstracts.append(abstract)
    corpus_texts.append(f"{title}. {abstract}")

print(f"Corpus size: {len(corpus_texts)} papers")

model_name = "allenai/specter2_base"
print(f"[SPECTER] Loading tokenizer/model on CPU: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

if device == "cuda":
    model.to("cuda")

corpus_embeddings = []
batch_size = 8  # safe for 10GB GPU

for i in range(0, len(corpus_texts), batch_size):
    batch = corpus_texts[i : i + batch_size]
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")

    if device == "cuda":
        inputs = inputs.to("cuda")

    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)

    corpus_embeddings.append(outputs.cpu())

corpus_embeddings = torch.cat(corpus_embeddings, dim=0)

# Move model *back to CPU* after embedding to keep GPU memory low
model.to("cpu")


def specter_agent(state: MessagesState):

    # Parse coordinator message
    msg = state["messages"][-1].content

    try:
        routing = eval(msg) if isinstance(msg, str) else msg
    except:
        return {
            "messages": [AIMessage(content="SPECTER2_ERROR: Could not parse coordinator message.")]
        }

    queries = routing.get("queries", [])
    if not queries:
        return {"messages": [AIMessage(content="SPECTER2 received no queries")]}

    # Encode query on GPU if available
    if device == "cuda":
        model.to("cuda")

    inputs = tokenizer(
        [queries[0]], padding=True, truncation=True, max_length=512, return_tensors="pt"
    )

    if device == "cuda":
        inputs = inputs.to("cuda")

    with torch.no_grad():
        q_emb = model(**inputs).last_hidden_state.mean(dim=1).cpu()

    # Move model back to CPU (free GPU)
    model.to("cpu")

    # Compute similarity
    scores = torch.cosine_similarity(q_emb.expand_as(corpus_embeddings), corpus_embeddings, dim=1)
    top_indices = torch.topk(scores, k=5).indices.tolist()

    results = [{"title": corpus_titles[i], "score": float(scores[i])} for i in top_indices]

    return {"messages": [AIMessage(content=str(results), name="specter2")]}
