"""CLI utility to run entity recognition and verification on ScholarCopilot examples."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

from .agents import EntityRecognitionAgent, VerificationAgent
from .data import ScholarCopilotDataset
from .types import AgentContext, RecognizedEntity

LOGGER = logging.getLogger("multi_agent_pipeline.verify")


def format_entities(entities: list[RecognizedEntity]) -> list[Dict[str, Any]]:
    return [
        {
            "text": entity.text,
            "type": entity.entity_type,
            "confidence": round(entity.confidence, 3),
            "span": entity.span.to_tuple() if entity.span else None,
        }
        for entity in entities
    ]


def run(args: argparse.Namespace) -> None:
    dataset = ScholarCopilotDataset(data_path=args.dataset)
    examples = dataset.examples()

    if not examples:
        raise SystemExit("Dataset returned no examples.")

    if args.example_index >= len(examples) or args.example_index < 0:
        raise SystemExit(
            f"example_index {args.example_index} out of range (0-{len(examples) - 1})."
        )

    example = examples[args.example_index]
    context: AgentContext = example.context

    # Run NER to annotate entities in the citation context
    ner_agent = EntityRecognitionAgent(
        model_name=args.ner_model,
        min_confidence=args.min_confidence,
        device=args.device,
    )
    context = ner_agent(context)

    # Run verification (optionally with Semantic Scholar lookup)
    verifier = VerificationAgent(
        enable_external_lookup=not args.disable_external_lookup,
        external_title_threshold=args.external_title_threshold,
        external_author_overlap_threshold=args.external_author_threshold,
    )
    verification = verifier(context)

    payload = {
        "query_id": example.query_id,
        "citation_context": context.citation_context,
        "recognized_entities": format_entities(context.recognized_entities),
        "selected_candidate": verification.selected_candidate.title if verification.selected_candidate else None,
        "is_verified": verification.is_verified,
        "justification": verification.justification,
        "metadata": verification.metadata,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"Saved verification output to {args.output}")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run entity recognition and verification on a ScholarCopilot example.",
    )
    parser.add_argument(
        "--dataset",
        default="datasets/scholar_copilot_eval_data_1k.json",
        help="Path to ScholarCopilot JSON dataset.",
    )
    parser.add_argument(
        "--example-index",
        type=int,
        default=0,
        help="Index of the example to evaluate.",
    )
    parser.add_argument(
        "--ner-model",
        default="dslim/bert-base-NER",
        help="Hugging Face model for entity recognition.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum confidence threshold for NER entities.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device string for NER inference (e.g. cuda:0).",
    )
    parser.add_argument(
        "--disable-external-lookup",
        action="store_true",
        help="Disable Semantic Scholar verification lookup.",
    )
    parser.add_argument(
        "--external-title-threshold",
        type=float,
        default=0.85,
        help="Minimum title similarity ratio to accept Semantic Scholar match.",
    )
    parser.add_argument(
        "--external-author-threshold",
        type=float,
        default=0.4,
        help="Minimum author overlap ratio to accept Semantic Scholar match.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write JSON output instead of printing.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":  # pragma: no cover
    main()
