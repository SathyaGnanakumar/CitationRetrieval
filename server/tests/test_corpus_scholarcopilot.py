import sys
from pathlib import Path

# Ensure `import src...` works when running pytest from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add parent directory for datasets import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets.scholarcopilot import build_citation_corpus


def test_build_citation_corpus_uses_bib_info_and_dedupes_by_title():
    dataset = [
        {
            "paper_id": "p1",
            "title": "Top Level Paper (should not be indexed here)",
            "bib_info": {
                "refs": [
                    {"citation_key": "c1", "title": "Paper A", "abstract": "A1"},
                    {"citation_key": "c2", "title": "Paper A", "abstract": "A2 (duplicate title)"},
                    {"citation_key": "c3", "title": "Paper B", "abstract": ""},
                ]
            },
        }
    ]

    docs = build_citation_corpus(dataset, clean=False)
    titles = sorted(d["title"] for d in docs)

    assert titles == ["Paper A", "Paper B"]
    assert all("Top Level Paper" not in t for t in titles)
    assert all("id" in d and "text" in d for d in docs)
