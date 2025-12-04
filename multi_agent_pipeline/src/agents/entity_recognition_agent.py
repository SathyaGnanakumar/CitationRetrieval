from langchain_core.messages import AIMessage
import langextract as lx
from langextract import data
from pydantic import BaseModel
from rapidfuzz import process, fuzz 
import re

class ResearchEntities(BaseModel):
    method_names: list[str]
    datasets: list[str]
    evaluation_metrics: list[str]
    tasks: list[str]
    
example = lx.data.ExampleData(
    text="We evaluated ResNet on CIFAR-10 using accuracy.",
    extractions=[
        lx.data.Extraction(
            extraction_class="name",
            extraction_text="ResNet on CIFAR-10 using accuracy",
            attributes={"method_names": ["ResNet"], "datasets": ["CIFAR-10"], "evaluation_metrics": ["accuracy"], "tasks": ["image classification"]}
        )
    ]
)

# find span offsets
# use str.find() to find exact match
def exact_span(text, span):
    if not span:
        return None, None
    start = text.find(span)
    if start == -1:
        return None, None
    return start, start + len(span)

# use rapidfuzz to find approximate match
def approx_span(text, span, threshold=85):
    matches = process.extract(span, [text], scorer=fuzz.partial_ratio)
    score = matches[0][1]
    if score < threshold:
        return None, None
    start = text.lower().find(span.lower().split()[0])
    return start, start + len(span)

def spanning(lx_result):
    for doc_extractions in lx_result:
        text = doc_extractions.text
        for extraction in doc_extractions.extractions:
            span = extraction.extraction_text
            
            start, end = exact_span(text, span)
            if start is None:
                start, end = approx_span(text, span)
            
            extraction.attributes["span_offsets"] = {
                "start": start,
                "end": end,
                "text": span
            }
            
    return lx_result

def entity_recognition_agent(state):    
    messages = state["messages"]

    docs = [
        data.Document(document_id=f"doc_{i}", text=m.content)
        for i, m in enumerate(messages)
    ]

    result = lx.extract(
        text_or_documents=docs,
        prompt_description="Extract method names, datasets, evaluation metrics, and tasks.",
        examples=[example],
        model_id="gemini-2.5-flash"
    )
    
    result = spanning(result)

    return {
        "messages": [
            AIMessage(
                name="analysis",
                content=str(result)
            )
        ]
    }