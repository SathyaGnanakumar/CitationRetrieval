# LLM Reranker Fixes and Configuration Options

## Issues Fixed

### 1. ✅ Hugging Face Response Format Error
**Problem:** `'str' object has no attribute 'content'`

**Root Cause:** Different LLM backends return responses in different formats:
- `ChatOllama` / `ChatOpenAI`: Returns object with `.content` attribute
- `HuggingFacePipeline`: Returns string directly or dict/list with text

**Fix:** Added robust response handling in `src/agents/formulators/llm_agent.py:103-119`:
```python
# Handle different response formats
if isinstance(response, str):
    response_text = response
elif hasattr(response, 'content'):
    response_text = response.content
elif isinstance(response, dict) and 'text' in response:
    response_text = response['text']
elif isinstance(response, list) and len(response) > 0:
    if isinstance(response[0], dict) and 'generated_text' in response[0]:
        response_text = response[0]['generated_text']
    else:
        response_text = str(response)
else:
    response_text = str(response)
```

### 2. ✅ Added OpenAI Support
**Added:** Full support for OpenAI models as a third inference option alongside Ollama and Hugging Face.

**Benefits:**
- No local GPU needed
- Faster inference (cloud-based)
- Access to GPT-4o, GPT-4o-mini, GPT-4-turbo
- Better quality for complex reranking tasks

---

## Configuration Options

You now have **3 ways** to run LLM reranking:

### Option 1: Ollama (Local, Free, Medium Quality)

**Best for:** Local development, privacy-sensitive work, offline usage

**Setup:**
```bash
# In .env
INFERENCE_ENGINE="ollama"
LOCAL_LLM="gemma3:4b"
```

**Requirements:**
- Ollama server running locally
- Download model: `ollama pull gemma3:4b`

**Pros:**
- ✅ Free
- ✅ Fast (after initial load)
- ✅ Works offline
- ✅ Privacy (runs locally)

**Cons:**
- ⚠️ Requires Ollama server running
- ⚠️ Medium quality compared to GPT-4
- ⚠️ Needs some RAM (4-8GB for 4B models)

---

### Option 2: Hugging Face (Local/Cloud, Free, Variable Quality)

**Best for:** GPU clusters, customization, research

**Setup:**
```bash
# In .env
INFERENCE_ENGINE="huggingface"
LOCAL_LLM="google/gemma-2-9b-it"
hf_key="your_hugging_face_token"
```

**Recommended Models:**
- `google/gemma-2-9b-it` - Good balance
- `meta-llama/Llama-3.1-8B-Instruct` - Better reasoning
- `mistralai/Mistral-7B-Instruct-v0.3` - Fast inference

**Requirements:**
- GPU with enough VRAM (8-16GB for 7-9B models)
- Hugging Face account (free)
- Model downloads automatically on first run

**Pros:**
- ✅ Free
- ✅ Full control over model
- ✅ Works great on GPU clusters
- ✅ Supports gated models (with token)
- ✅ Cached after first load (FAST!)

**Cons:**
- ⚠️ Requires GPU for good performance
- ⚠️ First run downloads model (slow)
- ⚠️ VRAM requirements

---

### Option 3: OpenAI (Cloud, Paid, Best Quality)

**Best for:** Production, best quality results, no GPU available

**Setup:**
```bash
# In .env
INFERENCE_ENGINE="openai"
LOCAL_LLM="gpt-4o-mini"  # or "gpt-4o", "gpt-4-turbo"
OPENAI_API_KEY="sk-proj-..."
```

**Model Options:**
- `gpt-4o-mini` - Fast, cheap, good quality ($0.15/1M input tokens)
- `gpt-4o` - Better quality, more expensive ($5/1M input tokens)
- `gpt-4-turbo` - Highest quality ($10/1M input tokens)

**Requirements:**
- OpenAI API key (paid)
- Internet connection

**Pros:**
- ✅ Best quality reranking
- ✅ No GPU needed
- ✅ Fast inference (cloud-based)
- ✅ No model downloads
- ✅ Scales easily

**Cons:**
- ⚠️ Costs money (though gpt-4o-mini is cheap)
- ⚠️ Requires internet
- ⚠️ Sends data to OpenAI

---

## Usage Examples

### Running with Hugging Face on GPU Cluster:

```bash
# In .env
INFERENCE_ENGINE="huggingface"
LOCAL_LLM="google/gemma-2-9b-it"

# Run evaluation
python compare_baselines_vs_system.py \
  --num-examples 500 \
  --use-dspy \
  --llm-reranker \
  --output-dir final \
  --k 20
```

**Expected output:**
```
🔄 Loading Hugging Face model: google/gemma-2-9b-it
   This will take a few minutes on first run...
✅ Hugging Face model loaded and cached!

Example 1: 🚀 Using cached LLM model: google/gemma-2-9b-it
Example 2: 🚀 Using cached LLM model: google/gemma-2-9b-it
...
```

### Running with OpenAI:

```bash
# In .env
INFERENCE_ENGINE="openai"
LOCAL_LLM="gpt-4o-mini"
OPENAI_API_KEY="sk-proj-..."

# Run evaluation
python compare_baselines_vs_system.py \
  --num-examples 500 \
  --use-dspy \
  --llm-reranker \
  --output-dir final \
  --k 20
```

**Expected output:**
```
🔄 Initializing OpenAI with model: gpt-4o-mini
✅ OpenAI ready!

Example 1: 🚀 Using cached LLM model: gpt-4o-mini
Example 2: 🚀 Using cached LLM model: gpt-4o-mini
...
```

### Running with Ollama:

```bash
# In .env
INFERENCE_ENGINE="ollama"
LOCAL_LLM="gemma3:4b"

# Make sure Ollama is running first
ollama serve  # In separate terminal

# Run evaluation
python compare_baselines_vs_system.py \
  --num-examples 500 \
  --use-dspy \
  --llm-reranker \
  --output-dir final \
  --k 20
```

---

## Cost Comparison (for 500 examples)

Assuming ~1000 tokens per reranking request:

| Method | Setup Cost | Per-Example Cost | 500 Examples Total |
|--------|-----------|------------------|-------------------|
| **Ollama** | Free (local hardware) | $0 | **$0** |
| **Hugging Face** | Free (GPU time) | $0 | **$0** |
| **OpenAI (gpt-4o-mini)** | Free | ~$0.00015 | **~$0.08** |
| **OpenAI (gpt-4o)** | Free | ~$0.005 | **~$2.50** |

---

## Recommendations

### For Development/Testing:
Use **Ollama** or **Hugging Face** (free, fast enough)

### For GPU Cluster (Your Case):
Use **Hugging Face** with cached model:
- First run: 2-5 min model load
- Subsequent examples: Fast (model stays in GPU memory)
- Total 500 examples: ~90 minutes

### For Production/Best Quality:
Use **OpenAI gpt-4o-mini**:
- No GPU needed
- Best quality/cost ratio
- Easy to scale

### For Research/Experimentation:
Use **Hugging Face** with different models:
- Full control
- Can fine-tune if needed
- Free

---

## Troubleshooting

### Hugging Face: "Out of Memory"
- Use smaller model (e.g., `google/gemma-2-2b-it`)
- Reduce `max_new_tokens` in builders.py:222
- Enable quantization (4-bit/8-bit loading)

### OpenAI: "Rate limit exceeded"
- Add delays between requests
- Use `gpt-4o-mini` (higher rate limits)
- Upgrade API tier

### Ollama: "Connection refused"
- Make sure Ollama server is running: `ollama serve`
- Check if model is downloaded: `ollama list`
- Pull model if needed: `ollama pull gemma3:4b`

---

## Performance Summary

| Metric | Ollama | Hugging Face | OpenAI |
|--------|--------|--------------|--------|
| **Quality** | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| **Speed (cached)** | ★★★★☆ | ★★★★★ | ★★★★☆ |
| **Cost** | ★★★★★ (Free) | ★★★★★ (Free) | ★★★☆☆ (Paid) |
| **GPU Required** | ❌ | ✅ | ❌ |
| **Setup Complexity** | Medium | Medium | Easy |
| **Best For** | Local dev | GPU clusters | Production |
