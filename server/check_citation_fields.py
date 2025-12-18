#!/usr/bin/env python3
"""Check what fields are in citation records."""

import json
import os
import sys
import re
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_loaders.scholarcopilot import load_dataset

load_dotenv()

dataset_path = os.getenv("DATASET_DIR")
dataset = load_dataset(dataset_path)

# Find first citation
for paper in dataset[:5]:
    paper_text = paper.get("paper", "")
    bib_info = paper.get("bib_info", {})

    if not bib_info:
        continue

    # Get first citation
    cite_pattern = re.compile(r"<\|cite_\d+\|>")
    for match in cite_pattern.finditer(paper_text):
        cite_token = match.group(0)

        if cite_token in bib_info:
            refs = bib_info[cite_token]
            if refs:
                print("=" * 80)
                print(f"Citation token: {cite_token}")
                print(f"Number of refs: {len(refs)}")
                print(f"\nFirst reference fields:")
                print(json.dumps(refs[0], indent=2))
                print("=" * 80)

                # Check what ID fields exist
                print(f"\nAvailable ID fields:")
                for key in refs[0].keys():
                    if "id" in key.lower() or "key" in key.lower():
                        print(f"  - {key}: {refs[0][key]!r}")

                exit(0)

print("No citations found!")
