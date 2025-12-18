# One-Time Optimization Run (uv run python scripts/optimize_once.py)

## Run summary

- Baseline score: 0.1298
- Optimized score: 0.1490
- Improvement: +0.0192 (+14.8%)
- Queries evaluated per phase: 100
- Metric compared: weighted retrieval score = 0.4*R@5 + 0.3*R@10 + 0.3\*MRR

## Terminal excerpt

```
Progress: 100/100, Avg Score: 0.149
Batch evaluation complete. Average score: 0.149
✓ Optimized score: 0.1490
✓ Improvement: +0.0192 (+14.8%)
SUMMARY
Baseline score:   0.1298
Optimized score:  0.1490
Improvement:      +0.0192
✅ One-time optimization complete!
```

## What changed to yield the improvement

- Main lift: GEPA optimization of the DSPy picker prompt (teacher: GPT-5 Mini) produced a better paper-selection prompt, raising the weighted retrieval score by +0.0192.
- Throughput/observability tweaks (helped run reliability, not the metric directly):
  - Reranker model cached in `resources["reranker_model"]` to avoid reloading every query.
  - Query-level logging + tqdm progress bar in `evaluate_batch`.
  - Lightweight JSON checkpointing every N queries for long jobs.

## Notes

- Warnings in logs (`self.cobj cannot be converted to a Python object for pickling`) come from best-effort persistence of evaluation batches and do not affect scoring.
- Command used: `uv run python scripts/optimize_once.py`
