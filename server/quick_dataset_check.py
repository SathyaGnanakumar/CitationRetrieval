#!/usr/bin/env python3
"""Quick check of dataset structure."""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_loaders.scholarcopilot import load_dataset

load_dotenv()

dataset_path = os.getenv("DATASET_DIR")
dataset = load_dataset(dataset_path)

# Find first paper with citations
for paper in dataset[:5]:
    bib_info = paper.get("bib_info", {})
    if bib_info:
        print("Sample paper bib_info structure:")
        print(json.dumps(list(bib_info.values())[0][0], indent=2))
        print("\nAvailable fields in citation:")
        print(list(list(bib_info.values())[0][0].keys()))
        break

print(f"\nSample paper keys:")
print(list(dataset[0].keys()))
