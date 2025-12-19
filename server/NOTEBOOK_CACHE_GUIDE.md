# Notebook Resource Caching Guide

## What Changed

The `test_single_example.ipynb` notebook now supports **resource caching**, making it much faster to run multiple times!

## Performance Improvement

| Run | Time |
|-----|------|
| **First run** (builds and caches) | ~20-30 minutes |
| **Subsequent runs** (loads from cache) | ~30 seconds |

**Speedup: 40-60x faster!** 🚀

---

## How It Works

### First Run: Build & Cache

```python
# Cell configuration
USE_CACHE = True
ENABLE_LLM_RERANKER = True

# What happens:
# 1. Builds BM25 index (~1 min)
# 2. Builds E5 embeddings (~10 min)
# 3. Builds SPECTER embeddings (~10 min)
# 4. Loads LLM model (~2-5 min)
# 5. Saves everything to .cache/ folder
```

**Output:**
```
🔨 Building resources from scratch...
   This may take a few minutes on first run (models will be downloaded)
============================================================
🔨 Building BM25 index for 9740 documents...
✅ BM25 index built successfully
============================================================
🔨 Building E5 embeddings for 9740 documents...
✅ E5 embeddings built successfully
============================================================
🔨 Building SPECTER embeddings for 9740 documents...
✅ SPECTER embeddings built successfully
============================================================
🔨 Building LLM Reranker resources...
   Inference Engine: huggingface
   Model: google/gemma-2-9b-it
✅ Hugging Face model loaded and cached!
============================================================
✅ All retrieval resources built successfully!

💾 Saving resources to cache for future use...
✅ Cache saved! Next run will be faster.
```

---

### Subsequent Runs: Load from Cache

**Output:**
```
📦 Attempting to load from cache...
📦 Loading resources from cache: .cache/a1b2c3d4/
✅ Resources loaded from cache!

✅ Resources ready:
   - BM25: 9740 documents indexed
   - E5: 9740 embeddings
   - SPECTER: 9740 embeddings
   - LLM Reranker: google/gemma-2-9b-it (huggingface)
```

**Total time: ~30 seconds instead of 20-30 minutes!**

---

## Configuration Options

### Option 1: Use Cache with LLM (Default - Fastest)

```python
USE_CACHE = True
ENABLE_LLM_RERANKER = True
```

- ✅ Loads everything from cache
- ✅ LLM model ready for reranking
- ✅ Fastest after first run

---

### Option 2: Use Cache without LLM

```python
USE_CACHE = True
ENABLE_LLM_RERANKER = False
```

- ✅ Loads BM25, E5, SPECTER from cache
- ⚠️ LLM reranking step will be slower (loads on-the-fly)

---

### Option 3: Rebuild from Scratch

```python
USE_CACHE = False
ENABLE_LLM_RERANKER = True
```

- ⚠️ Rebuilds everything (slow)
- ✅ Useful if you changed the dataset
- ✅ Will save new cache after building

---

### Option 4: No Cache, No LLM Preload

```python
USE_CACHE = False
ENABLE_LLM_RERANKER = False
```

- ⚠️ Slowest option
- ⚠️ Only use for debugging

---

## Cache Location

Cache is stored in:
```
corpus_loaders/scholarcopilot/.cache/<dataset_hash>/
├── metadata.json          # Cache info
├── corpus.pkl            # Document corpus
├── bm25.pkl              # BM25 metadata
├── bm25_index/           # BM25 index files
├── e5.pkl                # E5 metadata
├── e5_embeddings.pt      # E5 embeddings (large!)
├── specter.pkl           # SPECTER metadata
└── specter_embeddings.pt # SPECTER embeddings (large!)
```

**Note:** LLM models are cached by Hugging Face/transformers separately in `~/.cache/huggingface/`

---

## Cache Management

### Clear Cache

If you want to rebuild everything:

```python
from src.resources.cache import clear_cache

# Clear cache for current dataset
clear_cache(dataset_path)
```

Or manually delete the cache folder:
```bash
rm -rf corpus_loaders/scholarcopilot/.cache/
```

### Check Cache Status

```python
from src.resources.cache import get_cache_path
from pathlib import Path

cache_path = get_cache_path(dataset_path)
if (cache_path / "metadata.json").exists():
    print(f"✅ Cache exists at: {cache_path}")
else:
    print(f"❌ No cache found")
```

---

## Testing Different Queries

With caching enabled, you can quickly test different queries:

```python
# Change this to test different queries
QUERY_INDEX = 0  # Try 0, 1, 2, 3, etc.

# Run the notebook
# Resources load from cache in ~30 seconds
# Test different queries without rebuilding!
```

**Example workflow:**
1. First run: Build and cache everything (20-30 min)
2. Test query #0 (30 sec load + inference)
3. Change to `QUERY_INDEX = 1`
4. Test query #1 (30 sec load + inference)
5. Repeat for as many queries as you want!

---

## LLM Model Caching

### Hugging Face Models

When `ENABLE_LLM_RERANKER = True` and `INFERENCE_ENGINE = "huggingface"`:

**First time:**
```
🔄 Loading Hugging Face model: google/gemma-2-9b-it
   This will take a few minutes on first run...
✅ Hugging Face model loaded and cached!
```

**Subsequent runs:**
```
🔄 Loading Hugging Face model: google/gemma-2-9b-it
   This will take a few minutes on first run...
✅ Hugging Face model loaded and cached!
```

The model loads from HuggingFace's cache (`~/.cache/huggingface/`) which is fast (~2 min vs 5-10 min downloading).

**Then during inference:**
```
🚀 Using cached LLM model: google/gemma-2-9b-it
```

No reloading between examples!

---

### OpenAI Models

When `ENABLE_LLM_RERANKER = True` and `INFERENCE_ENGINE = "openai"`:

```
🔄 Initializing OpenAI with model: gpt-4o-mini
✅ OpenAI ready!
```

OpenAI client initializes instantly (cloud-based, no local loading).

---

### Ollama Models

When `ENABLE_LLM_RERANKER = True` and `INFERENCE_ENGINE = "ollama"`:

```
🔄 Initializing Ollama with model: gemma3:4b
✅ Ollama ready!
```

Ollama client initializes quickly (connects to local server).

---

## Troubleshooting

### Cache Not Loading

**Symptom:** Always rebuilds even with `USE_CACHE = True`

**Causes:**
1. Dataset changed - cache is invalidated automatically
2. Cache folder deleted
3. Different Python environment

**Solution:**
- Let it rebuild once, cache will be saved
- Or check if `.cache/` folder exists

---

### Out of Memory

**Symptom:** Crash when loading E5 or SPECTER embeddings

**Cause:** Large embeddings don't fit in RAM/VRAM

**Solution:**
```python
# Load only what you need
USE_CACHE = True
ENABLE_LLM_RERANKER = True  # Or False if tight on memory

# In the load_resources call, disable heavy components:
resources = load_resources(
    dataset_path,
    enable_bm25=True,
    enable_e5=False,      # Disable if needed
    enable_specter=False  # Disable if needed
)
```

---

### LLM Not Using Cache

**Symptom:** LLM loads on every query

**Cause:** `ENABLE_LLM_RERANKER = False`

**Solution:**
```python
# Enable LLM caching
ENABLE_LLM_RERANKER = True
```

Then you'll see:
```
🚀 Using cached LLM model: google/gemma-2-9b-it
```

---

## Summary

✅ **Always use caching** for faster iterations
✅ **Enable LLM reranker** for best performance
✅ **First run is slow** but saves time on all future runs
✅ **Cache is automatic** - builds and saves on first run
✅ **Test multiple queries quickly** after initial cache build

**Time saved: 20-30 minutes → 30 seconds per run!**
