# Pipeline Default Configuration

## 🎯 Default Behavior (All Features Enabled)

By default, the retrieval pipeline runs with **ALL features enabled** for maximum performance:

### ✅ Enabled by Default

1. **All Retrievers**

   - BM25 (sparse keyword matching)
   - E5 (dense embeddings)
   - SPECTER (scientific paper embeddings)

2. **Caching**

   - Resources are cached after first build
   - Subsequent runs load from cache (30s vs 10+ mins)

3. **Cross-Encoder Reranker**

   - BAAI/bge-reranker-v2-m3 (fast, accurate)
   - No external dependencies

4. **Evaluation Mode**

   - Shows ground truth comparison when available
   - Displays which gold citations were found

5. **Logging**
   - INFO level with timestamps and emojis
   - Shows progress for each pipeline stage

## 🔧 How to Run (Defaults)

```bash
# All features enabled automatically
uv run python main.py \
  --dataset ../datasets/scholar_copilot_eval_data_1k.json \
  --query "transformer architecture for sequence modeling" \
  --k 5
```

**First run:** ~10 minutes (builds indexes and embeddings)  
**Subsequent runs:** ~30 seconds (loads from cache)

## ⚙️ Customization (Opt-Out)

### Disable Specific Features

```bash
# Disable dense embeddings (BM25 only)
uv run python main.py --dataset ... --query "..." --bm25-only

# Disable only E5 (keep BM25 + SPECTER)
uv run python main.py --dataset ... --query "..." --no-e5

# Disable only SPECTER (keep BM25 + E5)
uv run python main.py --dataset ... --query "..." --no-specter

# Disable caching (rebuild from scratch)
uv run python main.py --dataset ... --query "..." --no-cache

# Disable evaluation mode
uv run python main.py --dataset ... --query "..." --no-eval

# Enable debug logging
uv run python main.py --dataset ... --query "..." --debug
```

### Enable Optional Features

```bash
# Use LLM reranker (requires Ollama)
uv run python main.py --dataset ... --query "..." --llm-reranker

# Force rebuild cache
uv run python main.py --dataset ... --query "..." --force-rebuild

# Clear cache and exit
uv run python main.py --dataset ... --clear-cache
```

## 📊 Evaluation Script

The evaluation script also uses all features by default:

```bash
# Evaluate with all retrievers + caching
uv run python evaluate.py \
  --dataset ../datasets/scholar_copilot_eval_data_1k.json \
  --num-queries 10

# BM25 only (faster)
uv run python evaluate.py --dataset ... --num-queries 10 --bm25-only

# With LLM reranker
uv run python evaluate.py --dataset ... --num-queries 10 --llm-reranker
```

## 🚀 Quick Start Examples

```bash
# 1. First run (builds everything)
uv run python main.py --dataset ../datasets/scholar_copilot_eval_data_1k.json \
  --query "attention mechanism in neural networks" --k 5

# 2. Second run (uses cache - FAST!)
uv run python main.py --dataset ../datasets/scholar_copilot_eval_data_1k.json \
  --query "efficient transformers for long sequences" --k 5

# 3. BM25 only (fastest, no embeddings)
uv run python main.py --dataset ../datasets/scholar_copilot_eval_data_1k.json \
  --query "BERT language model" --k 5 --bm25-only

# 4. Full evaluation on 20 queries
uv run python evaluate.py --dataset ../datasets/scholar_copilot_eval_data_1k.json \
  --num-queries 20 --k 10
```

## 📝 Summary

| Feature                | Default     | Flag to Change                  |
| ---------------------- | ----------- | ------------------------------- |
| BM25                   | ✅ Enabled  | `--bm25-only` (disables others) |
| E5                     | ✅ Enabled  | `--no-e5` or `--bm25-only`      |
| SPECTER                | ✅ Enabled  | `--no-specter` or `--bm25-only` |
| Caching                | ✅ Enabled  | `--no-cache`                    |
| Cross-Encoder Reranker | ✅ Enabled  | Can't disable (always runs)     |
| LLM Reranker           | ❌ Disabled | `--llm-reranker`                |
| Evaluation Mode        | ✅ Enabled  | `--no-eval`                     |
| Debug Logging          | ❌ Disabled | `--debug`                       |

## 💡 Tips

1. **First time setup:** Just run with defaults, let it build the cache
2. **Subsequent runs:** Cache makes everything fast
3. **Quick experiments:** Use `--bm25-only` (no embedding computation)
4. **Best results:** Use all retrievers (default) + cross-encoder reranker
5. **Production:** Pre-build cache once, then serve from cache
