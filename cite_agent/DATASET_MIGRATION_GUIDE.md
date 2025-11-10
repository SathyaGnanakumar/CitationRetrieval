# Dataset Migration Guide: CiteME.tsv → Scholar Copilot JSON

## Overview

This guide explains how to adapt the cite_agent code to work with the Scholar Copilot evaluation dataset (`scholar_copilot_eval_data_1k.json`) instead of the CiteME TSV dataset.

---

## Dataset Structure Comparison

### Current Dataset: CiteME.tsv
**Format**: Tab-separated values (TSV)
**Structure**: One citation per row

```
id | excerpt | target_paper_title | target_paper_url | source_paper_title | source_paper_url | year | split
```

**Example**:
```
10 | "...through a series of convolutions [CITATION]." | "Handwritten Digit Recognition..." | url | "Learning to Drive..." | url | 2018 | test
```

### New Dataset: scholar_copilot_eval_data_1k.json
**Format**: JSON
**Structure**: One paper per entry with multiple citations

```json
{
  "paper_id": "2004.11886",
  "title": "Paper Title",
  "abstract": "Abstract with <|reference_start|>...<|reference_end|>",
  "paper": "Full text with <|cite_0|>, <|cite_1|>, etc.",
  "bib_info": {
    "<|cite_0|>": [{
      "citation_key": "...",
      "title": "Cited Paper Title",
      "abstract": "...",
      "citation_corpus_id": "..."
    }]
  }
}
```

**Key Differences**:
- **Multiple citations per entry**: Each paper contains multiple `<|cite_X|>` markers
- **Nested structure**: Citation details are in `bib_info` dictionary
- **Full paper text**: The `paper` field contains the complete text, not just an excerpt
- **Rich metadata**: Includes abstracts for both citing and cited papers

---

## Required Code Changes

### 1. Data Loading (`main.py` lines 47-58)

**Current Code**:
```python
c = pd.read_csv("../CiteME/CiteME.tsv", sep="\t")
c.set_index("id", inplace=True)
```

**New Code**:
```python
import json

# Load the JSON dataset
with open("../dense_retrieval/scholar_copilot_eval_data_1k.json", "r") as f:
    scholar_data = json.load(f)

# Transform to a list of citation instances
citation_instances = []
for paper in scholar_data:
    paper_id = paper["paper_id"]
    paper_text = paper["paper"]
    bib_info = paper.get("bib_info", {})

    # Extract each citation marker
    for cite_marker, cite_list in bib_info.items():
        # cite_marker is like "<|cite_0|>"
        if cite_list:  # Check if citation info exists
            cited_paper = cite_list[0]  # Usually one citation per marker

            # Extract excerpt around citation marker
            excerpt = extract_excerpt_around_marker(paper_text, cite_marker)

            citation_instances.append({
                "id": f"{paper_id}_{cite_marker}",
                "excerpt": excerpt,
                "target_paper_title": cited_paper.get("title", ""),
                "target_paper_abstract": cited_paper.get("abstract", ""),
                "source_paper_title": paper.get("title", ""),
                "source_paper_id": paper_id,
                "year": extract_year_from_text(cited_paper.get("title", "")),
                "split": "test",  # Scholar Copilot data doesn't have train/test splits
                "citation_marker": cite_marker,
            })

# Convert to DataFrame for compatibility with existing code
c = pd.DataFrame(citation_instances)
c.set_index("id", inplace=True)
```

### 2. Helper Function: Extract Excerpt

Add this function before the data loading section:

```python
def extract_excerpt_around_marker(text: str, marker: str, context_chars=300):
    """
    Extract text around a citation marker.

    Args:
        text: Full paper text
        marker: Citation marker like "<|cite_0|>"
        context_chars: Number of characters to include before/after marker

    Returns:
        Excerpt with [CITATION] replacing the marker
    """
    marker_pos = text.find(marker)

    if marker_pos == -1:
        return ""

    # Get context before and after
    start = max(0, marker_pos - context_chars)
    end = min(len(text), marker_pos + len(marker) + context_chars)

    excerpt = text[start:end]

    # Replace the specific marker with [CITATION] to match CiteME format
    excerpt = excerpt.replace(marker, "[CITATION]")

    # Clean up other citation markers in the excerpt
    import re
    excerpt = re.sub(r'<\|cite_\d+\|>', '[CITATION]', excerpt)

    return excerpt.strip()
```

### 3. Helper Function: Extract Year

Add this function to extract year information:

```python
def extract_year_from_text(title: str, default_year=2024):
    """
    Extract year from citation title or use default.
    The Scholar Copilot dataset doesn't always have explicit year info.

    Args:
        title: Citation title
        default_year: Default year if not found

    Returns:
        Year as integer
    """
    import re

    # Try to find a 4-digit year in the title
    year_match = re.search(r'\b(19|20)\d{2}\b', title)

    if year_match:
        return int(year_match.group())

    return default_year
```

### 4. Update Main Loop (lines 100-168)

The existing loop should work with minimal changes. The main compatibility issue is that `target_paper_title` might not have the `TITLE_SEPARATOR` anymore:

**Update line 107**:
```python
# Old:
target_titles = citation["target_paper_title"].split(TITLE_SEPERATOR)

# New (handle both formats):
if TITLE_SEPERATOR in citation["target_paper_title"]:
    target_titles = citation["target_paper_title"].split(TITLE_SEPERATOR)
else:
    target_titles = [citation["target_paper_title"]]
```

---

## Complete Modified main.py Template

Here's a template for the modified `main.py`:

```python
import json
import os
import pandas as pd
import re
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
from retriever.agent import (
    LLMSelfAskAgentPydantic,
    LLMNoSearch,
    OutputSearchOnly,
    Output,
)
from utils.str_matcher import find_match_psr, find_multi_match_psr
from utils.tokens import num_tokens_from_string
from datetime import datetime
from time import time
from rich.progress import track
from retriever.llm_base import DEFAULT_TEMPERATURE

# -- Helper Functions --

def extract_excerpt_around_marker(text: str, marker: str, context_chars=300):
    """Extract text around a citation marker."""
    marker_pos = text.find(marker)
    if marker_pos == -1:
        return ""

    start = max(0, marker_pos - context_chars)
    end = min(len(text), marker_pos + len(marker) + context_chars)
    excerpt = text[start:end]
    excerpt = excerpt.replace(marker, "[CITATION]")
    excerpt = re.sub(r'<\|cite_\d+\|>', '[CITATION]', excerpt)

    return excerpt.strip()

def extract_year_from_text(title: str, default_year=2024):
    """Extract year from citation title."""
    year_match = re.search(r'\b(19|20)\d{2}\b', title)
    if year_match:
        return int(year_match.group())
    return default_year

# -- Modify the following variables as needed --
TITLE_SEPERATOR = "[TITLE_SEPARATOR]"
RESULT_FILE_NAME = f"scholar_copilot_results.json"
INCREMENTAL_SAVE = True

metadata = {
    "model": "gpt-4o",
    "temperature": DEFAULT_TEMPERATURE,
    "executor": "LLMSelfAskAgentPydantic",
    "search_provider": "SemanticScholarSearchProvider",
    "prompt_name": "few_shot_search",
    "actions": "search_relevance,search_citation_count,read,select",
    "search_limit": 10,
    "threshold": 0.8,
    "execution_date": datetime.now().isoformat(),
    "only_open_access": False,
    "dataset_split": "all",
    "use_web_search": False,
    "max_actions": 15,
    "dataset": "scholar_copilot_1k",
}

console = Console()

## Load the Scholar Copilot dataset
console.log("Loading Scholar Copilot dataset...")
with open("../../dense_retrieval/scholar_copilot_eval_data_1k.json", "r") as f:
    scholar_data = json.load(f)

# Transform to citation instances
citation_instances = []
for paper in scholar_data:
    paper_id = paper["paper_id"]
    paper_text = paper["paper"]
    bib_info = paper.get("bib_info", {})

    for cite_marker, cite_list in bib_info.items():
        if cite_list:
            cited_paper = cite_list[0]
            excerpt = extract_excerpt_around_marker(paper_text, cite_marker)

            if excerpt:  # Only add if excerpt was successfully extracted
                citation_instances.append({
                    "id": f"{paper_id}_{cite_marker.replace('<|cite_', '').replace('|>', '')}",
                    "excerpt": excerpt,
                    "target_paper_title": cited_paper.get("title", ""),
                    "source_paper_title": paper.get("title", ""),
                    "year": extract_year_from_text(cited_paper.get("title", "")),
                    "split": "test",
                })

c = pd.DataFrame(citation_instances)
c.set_index("id", inplace=True)

console.log(f"Loaded {len(c)} citation instances from {len(scholar_data)} papers")

# Filter by split
if metadata["dataset_split"] == "all":
    pass
elif metadata["dataset_split"] == "test":
    c = c[c["split"] == "test"]
elif metadata["dataset_split"] == "train":
    c = c[c["split"] == "train"]
else:
    raise ValueError("Invalid dataset split")

## Select executor
# ... (rest of the code remains the same as original)
```

---

## Key Considerations & Challenges

### 1. **Context Window Size**
The Scholar Copilot dataset has full paper texts. The `context_chars=300` parameter in `extract_excerpt_around_marker()` controls how much context to include. You may want to:
- Experiment with different values (150, 300, 500)
- Extract sentence-level context instead of character-level
- Include the full paragraph containing the citation

### 2. **Multiple Citations to Same Paper**
Some `<|cite_X|>` markers may reference the same paper. The current code treats each as a separate instance, which is correct for evaluation but may affect metrics.

### 3. **Year Information**
Scholar Copilot doesn't always provide explicit year data. The `extract_year_from_text()` helper tries to extract it from titles, but defaults to 2024. This may affect search quality if the agent filters by year.

**Options**:
- Remove year filtering from agent searches
- Use paper_id to lookup year from external sources
- Parse bib_text for year information

### 4. **Train/Test Split**
Scholar Copilot data doesn't have train/test splits. All instances are marked as "test". If you need splits, consider:
- Random splitting
- Splitting by paper_id
- Using a different evaluation protocol

### 5. **Abstract Information**
Scholar Copilot provides abstracts for cited papers, which could be used to enhance the agent's search/selection. Consider:
- Adding abstracts to the context
- Using them for verification
- Comparing with retrieved abstracts

### 6. **Excerpt Quality**
The quality of extracted excerpts depends on:
- How much context you include
- Whether citations appear mid-sentence or at sentence boundaries
- Presence of multiple nearby citations

**Recommendation**: Manually inspect ~10-20 extracted excerpts to validate quality.

---

## Testing the Migration

### Step 1: Validate Data Loading
```python
# Add after data loading:
print(f"Total papers: {len(scholar_data)}")
print(f"Total citations: {len(citation_instances)}")
print("\nSample citation instance:")
print(citation_instances[0])
```

### Step 2: Inspect Excerpts
```python
# Check excerpt quality
for i in range(min(5, len(c))):
    print(f"\n--- Citation {i} ---")
    print(f"Excerpt: {c.iloc[i]['excerpt'][:200]}...")
    print(f"Target: {c.iloc[i]['target_paper_title']}")
```

### Step 3: Run on Small Subset
Before processing all 1k entries:
```python
# Add before the main loop:
c = c.head(10)  # Test with 10 citations first
```

### Step 4: Compare Results
Run the same test set on both datasets and compare:
- Success rate
- Types of errors
- Agent behavior patterns

---

## Alternative Approach: Sentence-Level Extraction

For potentially better excerpts, consider extracting complete sentences:

```python
def extract_sentence_around_marker(text: str, marker: str, num_sentences=2):
    """Extract complete sentences around citation marker."""
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize

    # Split into sentences
    sentences = sent_tokenize(text)

    # Find sentence containing marker
    marker_sentence_idx = None
    for i, sent in enumerate(sentences):
        if marker in sent:
            marker_sentence_idx = i
            break

    if marker_sentence_idx is None:
        return ""

    # Get surrounding sentences
    start_idx = max(0, marker_sentence_idx - num_sentences)
    end_idx = min(len(sentences), marker_sentence_idx + num_sentences + 1)

    excerpt = " ".join(sentences[start_idx:end_idx])
    excerpt = excerpt.replace(marker, "[CITATION]")
    excerpt = re.sub(r'<\|cite_\d+\|>', '[CITATION]', excerpt)

    return excerpt.strip()
```

Add `nltk` to your dependencies if using this approach.

---

## Rate Limiting: How to AVOID Hitting Limits

### The Problem

**CRITICAL**: The Scholar Copilot dataset is **much larger** than CiteME, and **without rate limiting you WILL hit API limits and your script will crash!**

- **CiteME**: ~100 citation instances
- **Scholar Copilot**: ~1000 papers × ~5-10 citations each = **5,000-10,000 instances**

Each citation triggers:
- **Agent actions**: Up to 15 actions (searches, reads, selects)
- **Semantic Scholar API calls**: 2-5 calls per action
- **LLM API calls**: 1-3 GPT-4o calls per action
- **Total per citation**: ~50-100 API calls

**Total for full dataset**: 250,000 - 1,000,000 API calls!

### API Rate Limits (What Happens Without Protection)

#### Semantic Scholar API
- **Without API key**: 100 requests per 5 minutes → **You'll hit this in ~2 minutes**
- **With API key**: ~1 request per second (unofficial) → **You'll hit this in ~5 minutes**
- **Result**: 429 errors, script crashes

#### OpenAI API (GPT-4o)
- **Tier 1**: 500 requests/minute → **You'll max out in 1-2 citations**
- **Tier 2**: 5,000 requests/minute → **You'll max out in 10-20 citations**
- **Result**: Rate limit errors, wasted API calls, script crashes

### How to PREVENT Rate Limit Errors

**The current code has NO rate limiting** - you MUST add it before running on Scholar Copilot!

#### 1. Add Rate Limiting to Semantic Scholar API

Edit `cite_agent/src/utils/semantic_scholar.py` and add delays:

```python
import time
from functools import wraps

def rate_limit(min_interval=1.0):
    """Decorator to enforce minimum time between calls."""
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator
```

Then modify the API methods:

```python
class SemanticScholarAPI:
    # ... existing code ...

    @rate_limit(min_interval=1.0)  # 1 second between calls
    def relevance_search(self, query, fields, ...):
        # ... existing code ...

    @rate_limit(min_interval=1.0)
    def get_details(self, paper_id, fields):
        # ... existing code ...
```

#### 2. Add Retry Logic with Exponential Backoff

Handle rate limit errors gracefully:

```python
import time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=2):
    """Retry with exponential backoff on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limit
                        delay = base_delay * (2 ** attempt)
                        print(f"Rate limited. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        raise
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(base_delay)
            raise Exception(f"Failed after {max_retries} retries")
        return wrapper
    return decorator
```

Apply to API calls:

```python
@retry_with_backoff(max_retries=5, base_delay=2)
@rate_limit(min_interval=1.0)
def relevance_search(self, query, fields, ...):
    response = requests.get(...)
    response.raise_for_status()  # Raise exception on HTTP errors
    return response.json()
```

#### 3. Add Progress Saving & Resume Capability

Modify `main.py` to support resuming from failures:

```python
import os

# Load existing results if available
results = []
processed_ids = set()

if os.path.exists(RESULT_FILE_NAME):
    console.log(f"[yellow]Found existing results file. Loading...")
    with open(RESULT_FILE_NAME, "r") as f:
        existing_data = json.load(f)
        results = existing_data.get("results", [])
        processed_ids = {r["id"] for r in results}
    console.log(f"[green]Resuming from {len(results)} completed citations")

# Main loop
for cid, citation in track(...):
    # Skip if already processed
    if cid in processed_ids:
        console.log(f"[yellow]{cid}: Skipping (already processed)")
        continue

    # ... rest of processing ...
```

#### 4. Batch Processing Strategy

Process in smaller batches to reduce risk:

```python
# In main.py, after loading data:

BATCH_SIZE = 100  # Process 100 citations at a time
START_INDEX = 0   # Change this to resume from specific point

c = c.iloc[START_INDEX:START_INDEX + BATCH_SIZE]
console.log(f"Processing citations {START_INDEX} to {START_INDEX + BATCH_SIZE}")
```

#### 5. Add Delays Between Citations

Add a small delay between processing each citation:

```python
import time

for cid, citation in track(...):
    # ... processing code ...

    # Add delay between citations to avoid rate limits
    time.sleep(0.5)  # 0.5 second delay
```

### Recommended Configuration for Scholar Copilot

Add this to your `main.py` after imports:

```python
# Rate limiting configuration
ENABLE_RATE_LIMITING = True
SEMANTIC_SCHOLAR_DELAY = 1.0  # seconds between S2 API calls
CITATION_DELAY = 0.5           # seconds between processing citations
BATCH_SIZE = 100               # process in batches
ENABLE_RESUME = True           # resume from existing results

# API configuration
OPENAI_RPM = 500  # requests per minute (adjust to your tier)
```

### Cost Estimation

Before running on full dataset, estimate costs:

```python
# Cost calculator
num_citations = len(c)
avg_actions_per_citation = 10  # conservative estimate
avg_tokens_per_action = 2000  # input + output

total_tokens = num_citations * avg_actions_per_citation * avg_tokens_per_action

# GPT-4o pricing (as of 2024)
input_price_per_1m = 2.50  # $2.50 per 1M input tokens
output_price_per_1m = 10.00  # $10.00 per 1M output tokens

estimated_cost = (total_tokens / 1_000_000) * ((input_price_per_1m + output_price_per_1m) / 2)

print(f"Estimated citations: {num_citations}")
print(f"Estimated tokens: {total_tokens:,}")
print(f"Estimated cost: ${estimated_cost:.2f}")
```

**For Scholar Copilot (5000 citations)**:
- Estimated tokens: 100M tokens
- Estimated cost: $500-$1000

### Recommended Workflow

1. **Start small**: Test with 10 citations
   ```python
   c = c.head(10)
   ```

2. **Test one batch**: Process 100 citations
   ```python
   c = c.head(100)
   ```

3. **Monitor costs**: Check OpenAI usage dashboard

4. **Process in batches**: Run overnight in batches of 100-500
   ```python
   for batch_start in range(0, len(c), BATCH_SIZE):
       batch_end = min(batch_start + BATCH_SIZE, len(c))
       process_batch(c.iloc[batch_start:batch_end])
   ```

5. **Enable resume**: Use INCREMENTAL_SAVE and skip processed IDs

### Complete Rate-Limited Implementation

Here's a complete implementation you can add to `main.py`:

```python
import time
from datetime import datetime, timedelta

class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_minute=60):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0

    def wait(self):
        """Wait if necessary to respect rate limit."""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_call = time.time()

# Initialize rate limiters
s2_limiter = RateLimiter(calls_per_minute=60)  # 1 call per second
openai_limiter = RateLimiter(calls_per_minute=500)  # adjust to your tier

# Patch the API calls in agent
original_search = agent.search_provider.s2api.relevance_search

def rate_limited_search(*args, **kwargs):
    s2_limiter.wait()
    return original_search(*args, **kwargs)

agent.search_provider.s2api.relevance_search = rate_limited_search
```

---

## Summary Checklist

### Initial Setup
- [ ] Backup original `main.py`
- [ ] Set up Semantic Scholar API key in `.env`
- [ ] Check OpenAI API tier and rate limits
- [ ] Estimate costs for full dataset

### Code Changes
- [ ] Add helper functions (`extract_excerpt_around_marker`, `extract_year_from_text`)
- [ ] Replace CSV loading with JSON loading
- [ ] Transform JSON data to DataFrame format
- [ ] Update `target_titles` splitting logic
- [ ] Add rate limiting to Semantic Scholar API calls
- [ ] Add retry logic with exponential backoff
- [ ] Add resume capability for interrupted runs

### Testing
- [ ] Test on 10 citations first
- [ ] Validate excerpt quality manually
- [ ] Check API usage and costs
- [ ] Test resume functionality
- [ ] Run on 100 citation batch

### Production Run
- [ ] Configure batch processing
- [ ] Enable incremental saving
- [ ] Monitor API usage and costs
- [ ] Run full evaluation in batches
- [ ] Compare results with CiteME baseline

---

## Questions & Troubleshooting

**Q: Excerpts are too short/long?**
A: Adjust `context_chars` parameter (try 150, 300, 500) or use sentence-level extraction.

**Q: Year extraction failing?**
A: Check the `ori_bib_text` field in `bib_info` - it contains BibTeX with year info.

**Q: Multiple citations in one excerpt?**
A: This is expected. The agent should handle `[CITATION]` appearing multiple times.

**Q: Performance is worse than CiteME?**
A: Scholar Copilot may be harder - full papers vs curated excerpts. Consider:
- Increasing context size
- Adjusting agent parameters
- Using different prompts

---

## Future Enhancements

1. **Use citation context**: Extract semantic context (e.g., "building on [CITATION]" vs "unlike [CITATION]")
2. **Leverage abstracts**: Use cited paper abstracts for better matching
3. **Multi-citation handling**: Special handling for papers cited multiple times
4. **BibTeX parsing**: Extract structured data from `ori_bib_text`
5. **Year filtering**: Improve year extraction from BibTeX or external sources

---

Good luck with the migration! Start small, validate carefully, and iterate.
