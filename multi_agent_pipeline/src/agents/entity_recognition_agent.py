from langchain_core.messages import AIMessage
import langextract as lx
from langextract import data
from pydantic import BaseModel
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

    return {
        "messages": [
            AIMessage(
                name="analysis",
                content=str(result)
            )
        ]
    }