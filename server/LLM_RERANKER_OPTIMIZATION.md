# LLM Reranker Performance Optimization

## Problem

The Hugging Face LLM model was being loaded **inside the evaluation loop** for every single example, causing:
- ❌ Massive performance degradation (loading time multiplied by number of examples)
- ❌ Unnecessary GPU memory allocation/deallocation
- ❌ Wasted time downloading/caching model weights repeatedly

## Solution

Load the model **once** when building resources, then **reuse** it across all examples.

## Changes Made

### 1. Added `build_llm_reranker_resources()` in `src/resources/builders.py`

This function loads the LLM model once and caches it:

```python
def build_llm_reranker_resources(
    inference_engine: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build LLM reranker resources (load model once, reuse across examples).

    Args:
        inference_engine: "ollama" or "huggingface" (defaults to INFERENCE_ENGINE env var)
        model_name: Model identifier

    Returns:
        Dict with llm_model, inference_engine, model_name
    """
```

**Supports both:**
- Ollama: Initializes ChatOllama once
- Hugging Face: Loads model with `device_map="auto"` once

### 2. Updated `build_inmemory_resources()`

Added `enable_llm_reranker` parameter:

```python
def build_inmemory_resources(
    docs: List[Dict[str, Any]],
    *,
    enable_bm25: bool = True,
    enable_e5: bool = True,
    enable_specter: bool = True,
    enable_llm_reranker: bool = False,  # NEW!
    ...
) -> Dict[str, Any]:
```

When enabled, adds `resources["llm_reranker"]` with cached model.

### 3. Modified `llm_reranker()` in `src/agents/formulators/llm_agent.py`

Now checks for cached model first:

```python
# Check if LLM model is cached in resources (PERFORMANCE OPTIMIZATION)
llm_reranker_res = resources.get("llm_reranker")

if llm_reranker_res and "llm_model" in llm_reranker_res:
    # Use cached model (FAST PATH - no loading overhead)
    llm = llm_reranker_res["llm_model"]
    model_id = llm_reranker_res.get("model_name", "cached")
    print(f"🚀 Using cached LLM model: {model_id}")
else:
    # Fallback: load model on-the-fly (SLOW PATH)
    print(f"⚠️  LLM model not found in resources - loading on-the-fly (SLOW!)")
    ...
```

### 4. Updated `compare_baselines_vs_system.py`

Enables LLM reranker caching when `--llm-reranker` flag is used:

```python
resources = build_inmemory_resources(
    corpus,
    enable_bm25=True,
    enable_e5=True,
    enable_specter=True,
    enable_llm_reranker=use_llm_reranker,  # NEW!
)
```

Also handles cache loading case:

```python
# If using LLM reranker and model not in cache, load it now
if use_llm_reranker and "llm_reranker" not in resources:
    logger.info(f"\n🔧 LLM reranker enabled but not in cache - loading now...")
    resources["llm_reranker"] = build_llm_reranker_resources()
```

## Usage

### For 500-example evaluation:

```bash
# Make sure you have Hugging Face inference enabled
# In .env:
#   INFERENCE_ENGINE="huggingface"
#   LOCAL_LLM="google/gemma-2-9b-it"

python compare_baselines_vs_system.py \
  --num-examples 500 \
  --use-dspy \
  --llm-reranker \
  --output-dir final \
  --k 20
```

### What you'll see:

**First time (building resources):**
```
🔨 Building LLM Reranker resources...
   Inference Engine: huggingface
   Model: google/gemma-2-9b-it
🔄 Loading Hugging Face model: google/gemma-2-9b-it
   This will take a few minutes on first run...
✅ Hugging Face model loaded and cached!
```

**Then for each example:**
```
🚀 Using cached LLM model: google/gemma-2-9b-it  # FAST!
```

**Instead of (old behavior):**
```
🔄 Loading Hugging Face model: google/gemma-2-9b-it...  # SLOW - repeated 500 times!
```

## Performance Impact

**Before:**
- Loading model: ~2-5 minutes per example
- 500 examples: **1,000 - 2,500 minutes** (16-41 hours!)

**After:**
- Loading model: ~2-5 minutes **once**
- Inference per example: ~5-10 seconds
- 500 examples: **~5 minutes + 500×10s = ~90 minutes** (1.5 hours)

**Speedup: ~10-25x faster!**

## Backwards Compatibility

The code includes a fallback path that loads the model on-the-fly if not found in resources, maintaining compatibility with existing code that doesn't use the new caching.

You'll see a warning:
```
⚠️  LLM model not found in resources - loading on-the-fly (SLOW!)
💡 Tip: Enable llm_reranker in build_inmemory_resources for better performance
```

## Configuration

### .env file settings:

```bash
# Use Hugging Face inference (recommended for GPU clusters)
INFERENCE_ENGINE="huggingface"

# Choose your model
LOCAL_LLM="google/gemma-2-9b-it"
# Or:
# LOCAL_LLM="meta-llama/Llama-3.1-8B-Instruct"
# LOCAL_LLM="mistralai/Mistral-7B-Instruct-v0.3"

# Your HF token (for gated models)
hf_key="your_hugging_face_token"
```

### For Ollama (if preferred):

```bash
INFERENCE_ENGINE="ollama"
LOCAL_LLM="gemma3:4b"
```

## Testing

Test with the single example notebook/script first:

```bash
jupyter notebook test_single_example.ipynb
```

You should see:
- Model loads once at the beginning
- Fast inference on the single example
- No repeated model loading

## Notes

- The cached model stays in GPU memory for the entire run
- Make sure your GPU has enough memory for the model
- If you run out of memory, use a smaller model or reduce batch size
- The model is automatically placed on GPU with `device_map="auto"`
