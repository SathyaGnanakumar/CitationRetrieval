from langchain_core.messages import AIMessage
import langextract as lx
from langextract import data
from pydantic import BaseModel
import re
import json


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
            attributes={
                "method_names": ["ResNet"],
                "datasets": ["CIFAR-10"],
                "evaluation_metrics": ["accuracy"],
                "tasks": ["image classification"]
            }
        )
    ]
)


def entity_recognition_agent(state):

    messages = state["messages"]

    docs = [
        data.Document(document_id=f"doc_{i}", text=m.content)
        for i, m in enumerate(messages)
    ]

    # Call LangExtract (v1.1.1: returns a MIXED list)
    raw_result = lx.extract(
        text_or_documents=docs,
        prompt_description="Extract method names, datasets, evaluation metrics, and tasks.",
        examples=[example],
        model_id="gemini-2.5-flash"
    )

    # KEEP ONLY Extraction objects
    all_extractions = [
        x for x in raw_result if isinstance(x, data.Extraction)
    ]

    enriched_docs = []
    filtered_docs = []

    for doc in docs:

        # Only extractions tied to this doc
        doc_extractions = [
            e for e in all_extractions if e.document_id == doc.document_id
        ]

        highlighted_text = doc.text
        entity_list = []

        all_methods, all_datasets, all_metrics, all_tasks = [], [], [], []

        for e in doc_extractions:
            attrs = e.attributes or {}
            extracted_text = e.extraction_text

            # Collect fields
            all_methods.extend(attrs.get("method_names", []))
            all_datasets.extend(attrs.get("datasets", []))
            all_metrics.extend(attrs.get("evaluation_metrics", []))
            all_tasks.extend(attrs.get("tasks", []))

            # Grounding (manual: string search)
            match = re.search(re.escape(extracted_text), doc.text)
            if match:
                start, end = match.start(), match.end()
            else:
                start, end = None, None

            if start is not None:
                highlighted_text = (
                    highlighted_text[:start]
                    + "[["
                    + highlighted_text[start:end]
                    + "]]"
                    + highlighted_text[end:]
                )

            entity_list.append({
                "text": extracted_text,
                "method_names": attrs.get("method_names", []),
                "datasets": attrs.get("datasets", []),
                "evaluation_metrics": attrs.get("evaluation_metrics", []),
                "tasks": attrs.get("tasks", []),
                "start_char": start,
                "end_char": end,
            })

        enriched_doc = {
            "document_id": doc.document_id,
            "original_text": doc.text,
            "highlighted_text": highlighted_text,
            "entities": entity_list,
            "all_method_names": list(set(all_methods)),
            "all_datasets": list(set(all_datasets)),
            "all_metrics": list(set(all_metrics)),
            "all_tasks": list(set(all_tasks)),
        }

        enriched_docs.append(enriched_doc)
        if entity_list:
            filtered_docs.append(enriched_doc)

    return {
        "messages": [
            AIMessage(
                name="entity-recognition-output",
                content=json.dumps({
                    "enriched_docs": enriched_docs,
                    "filtered_docs": filtered_docs,
                })
            )
        ]
    }
