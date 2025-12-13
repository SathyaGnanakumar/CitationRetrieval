"""Evaluate GPT prompts that select the correct citation from candidate papers.

This utility loads paired JSONL files produced by ``split_train.py`` where
``correct.jsonl`` holds the ground-truth citation entries and ``wrong.jsonl``
holds the negative candidates. For each example we construct the numbered
candidate list used by ``SimpleCitationRetriever`` and ask an OpenAI model to
return the most likely cited title. Results are written to JSONL so they can
be analysed later on.

Example usage::

    uv run python -m dspy_prompt_generator.run_prompt_classifier \
        --max-examples 25 --output results/prompt_eval.jsonl

Configure the OpenAI key via ``OPENAI_API_KEY`` (already loaded through
``python-dotenv`` in this project).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Dict, Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORRECT = PROJECT_ROOT / "datasets" / "correct.jsonl"
DEFAULT_WRONG = PROJECT_ROOT / "datasets" / "wrong.jsonl"


@dataclass
class Candidate:
    title: str
    abstract: str


@dataclass
class PromptExample:
    example_id: str
    citation_context: str
    candidates: List[Candidate]
    positive_title: str


def load_examples(
    correct_path: Path,
    wrong_path: Path,
    seed: int,
) -> List[PromptExample]:
    """Merge correct/incorrect JSONL files into prompt-ready examples."""

    positives: Dict[str, PromptExample] = {}

    with correct_path.open("r", encoding="utf-8") as correct_file:
        for line in correct_file:
            record = json.loads(line)
            example_id = record.get("example_id")
            guess = record.get("guess") or {}

            title = (guess.get("title") or "").strip()
            if not example_id or not title:
                continue

            positives[example_id] = PromptExample(
                example_id=example_id,
                citation_context=record.get("citation_context", ""),
                candidates=[Candidate(title=title, abstract=guess.get("abstract") or "")],
                positive_title=title,
            )

    missing_negative = 0

    with wrong_path.open("r", encoding="utf-8") as wrong_file:
        for line in wrong_file:
            record = json.loads(line)
            example_id = record.get("example_id")
            guess = record.get("guess") or {}

            if not example_id or example_id not in positives:
                missing_negative += 1
                continue

            title = (guess.get("title") or "").strip()
            if not title:
                continue

            example = positives[example_id]
            known_titles = {c.title for c in example.candidates}
            if title in known_titles:
                continue

            example.candidates.append(
                Candidate(title=title, abstract=guess.get("abstract") or "")
            )

    examples = [ex for ex in positives.values() if len(ex.candidates) > 1]

    if missing_negative:
        print(f"⚠️  Ignored {missing_negative} negatives without matching positives")

    rng = random.Random(seed)
    for ex in examples:
        rng.shuffle(ex.candidates)

    rng.shuffle(examples)

    return examples


PROMPT_TEMPLATE = """
You are an expert citation retrieval system as described in the paper "Multi-Agent System for Reliable Citation Retrieval".
Your goal is to autonomously retrieve, verify, and recommend academic references given a query or document excerpt.

Task:
Given a citation context from a scientific paper (where a citation is missing), identify the correct paper from a list of candidates.
Analyze the context to understand the specific claim, method, or result being cited.
Then, evaluate each candidate paper to see if it matches the context.
Finally, select the best matching paper.

Citation Context:
{citation_context}

Candidate Papers:
{candidate_block}

Think step-by-step about which candidate best matches the context. Summarize your reasoning, then provide the exact title that should fill the citation. Respond in JSON with fields:
{{
  "reasoning": "<your step-by-step analysis>",
  "selected_title": "<exact candidate title>"
}}
""".strip()


def format_candidates(candidates: Iterable[Candidate]) -> str:
    lines: List[str] = []
    for idx, candidate in enumerate(candidates, start=1):
        lines.append(f"{idx}. Title: {candidate.title}")
        abstract = candidate.abstract.strip()
        if abstract:
            lines.append(f"   Abstract: {abstract}")
        lines.append("")
    return "\n".join(lines).strip()


def classify_example(
    client: OpenAI,
    model: str,
    example: PromptExample,
    temperature: float,
    retry: int,
    sleep: float,
) -> Dict[str, Any]:
    """Send the formatted prompt to the OpenAI Chat Completions API."""

    prompt = PROMPT_TEMPLATE.format(
        citation_context=example.citation_context,
        candidate_block=format_candidates(example.candidates),
    )

    for attempt in range(1, retry + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert assistant that strictly follows the provided task instructions.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            message = response.choices[0].message.content
            parsed = json.loads(message)

            selected = (parsed.get("selected_title") or "").strip()
            reasoning = parsed.get("reasoning")
            cleaned_selected = selected.lower().strip("{}\"' ")
            cleaned_positive = example.positive_title.lower().strip("{}\"' ")
            is_correct = cleaned_selected == cleaned_positive

            return {
                "example_id": example.example_id,
                "prompt": prompt,
                "response": parsed,
                "selected_title": selected,
                "positive_title": example.positive_title,
                "is_correct": is_correct,
                "reasoning": reasoning,
            }

        except Exception as exc:  # noqa: BLE001 broad to keep retry simple
            if attempt == retry:
                raise
            wait = sleep * attempt
            print(f"⚠️  API call failed on attempt {attempt}: {exc}. Retrying in {wait:.1f}s...")
            time.sleep(wait)


def iter_examples(
    examples: List[PromptExample],
    max_examples: int | None,
) -> Iterator[PromptExample]:
    count = 0
    for example in examples:
        if max_examples is not None and count >= max_examples:
            break
        count += 1
        yield example


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate citation prompts with GPT-4o-mini")
    parser.add_argument("--correct-path", type=Path, default=DEFAULT_CORRECT)
    parser.add_argument("--wrong-path", type=Path, default=DEFAULT_WRONG)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("results/prompt_eval.jsonl"))
    args = parser.parse_args()

    examples = load_examples(args.correct_path, args.wrong_path, seed=args.seed)

    if not examples:
        raise SystemExit("No examples available after merging positives/negatives.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    total = 0
    correct = 0

    with args.output.open("w", encoding="utf-8") as sink:
        for example in iter_examples(examples, args.max_examples):
            result = classify_example(
                client=client,
                model=args.model,
                example=example,
                temperature=args.temperature,
                retry=args.retry,
                sleep=args.sleep,
            )
            total += 1
            correct += int(result["is_correct"])
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(
                f"[{total:03d}] {example.example_id} → {result['selected_title']!r}"
                f" (truth: {example.positive_title!r})"
                f" | {'✅' if result['is_correct'] else '❌'}"
            )

    accuracy = correct / total if total else 0.0
    print(f"\nFinished {total} examples. Accuracy: {accuracy:.1%}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
