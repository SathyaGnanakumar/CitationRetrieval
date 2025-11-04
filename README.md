# CitationRetrieval

Sathya Gnanakumar, Ishaan Kalra, Dhruv Suri, Kushal Kapoor, Vishnu Sreekanth, Vibhu Singh

# Timeline of Progess

Week 1:
- Explored CiteMe paper and determined that it will serve as our final benchmark for our model. Began looking into TF/IDF and BM25 baseline model development on retrieved data.

Weeks 2-4:
- Pivoted our approach for our baselines as we gained further clarity from talking with Rifaa and Tom. We will be using BM25 and Dense Retrieval as our baselines. We will then run the baselines on the ScholarCopilot eval data from the training dataset.
- We determined that ScholarCopilot will serve as our official dataset for training and evaluating our pipeline. Currently training on a smaller 1K dataset due to compute constraints but will need to determine how to split the larger training dataset of 600K examples into batches for bigger training runs. 
- Clarified how we want to implement our multi-agent pipeline. The agents will be trained on a subset of the ScholarCopilot data and then the pipeline will narrow down possible options to produce the most accurate citation.

Week 5:
- Identified database of papers and training dataset from ScholarCopilot containing sentences with in-text citations and the corresponding cited papers
- Worked on and obtained results from BM25 and Dense Retrieval Baselines (SPECTER2 and E5-Large) 


## ScholarCopilot Database of Papers:
https://huggingface.co/datasets/TIGER-Lab/ScholarCopilot-Data-v1
## ScholarCopilot Training Dataset:
https://huggingface.co/datasets/ubowang/ScholarCopilot-TrainingData
## CiteME Dataset:
https://huggingface.co/datasets/bethgelab/CiteME

## Our Dense Retrieval Baseline Results:
https://drive.google.com/drive/folders/1L1Eo1dE77bOelBOvWEy466Hhir8OSYPE?usp=sharing

---

## Evaluation Framework

Week 6+:
- Built comprehensive evaluation framework for citation retrieval models
- Unified evaluation harness supporting BM25, SPECTER2, E5-Large baselines
- Comprehensive metrics: Recall@k, Precision@k, MRR, Exact Match
- Automatic error analysis and failure logging

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run example (quick test on 50 examples)
python example_evaluation.py

# Run full baseline evaluation
python run_baseline_evaluation.py --data_path BM25/scholar_copilot_eval_data_1k.json

# Evaluate specific models only
python run_baseline_evaluation.py --models bm25 specter2
```

### Framework Structure

```
evaluation/
├── evaluator.py        # Main evaluation harness
├── metrics.py          # Metrics calculation (Recall@k, MRR, etc.)
├── data_loader.py      # Dataset loading and preprocessing
├── models/
│   ├── base_model.py   # Abstract base class
│   ├── bm25_model.py   # BM25 baseline
│   └── dense_model.py  # Dense retrieval (SPECTER2, E5)
└── README.md           # Detailed documentation
```

**See [evaluation/README.md](evaluation/README.md) for detailed documentation.**

### Usage Example

```python
from evaluation import CitationEvaluator, CitationDataLoader
from evaluation.models import BM25Model, DenseRetrievalModel

# Load data
loader = CitationDataLoader("BM25/scholar_copilot_eval_data_1k.json")
examples = loader.extract_examples()

# Initialize models
models = {
    'BM25': BM25Model(),
    'SPECTER2': DenseRetrievalModel('allenai/specter2'),
    'E5-Large': DenseRetrievalModel('intfloat/e5-large-v2')
}

# Run comparison
evaluator = CitationEvaluator()
comparison = evaluator.compare_models(models, examples)
```

### Next Steps
1. Run baseline evaluation on full 1K dataset
2. Implement CiteAgent multi-agent pipeline
3. Compare CiteAgent vs baselines
4. Scale to 600K dataset
5. Benchmark on CiteME test set