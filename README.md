# CitationRetrieval

Sathya Gnanakumar, Ishaan Kalra, Dhruv Suri, Kushal Kapoor, Vishnu Sreekanth, Vibhu Singh

# Timeline of Progess

Week 1:
- Accumulated training data from ArXiv based on CiteMe Benchmark. Began looking into TF/IDF and BM25 baseline model development on retrieved data.

Weeks 2-4:
- Pivoted our approach for our baselines as we gained further clarity from talking with Rifaa and Tom. We will be using BM25 and Dense Retrieval as our baselines. We will then run the baselines on the ScholarCopilot eval data from the training dataset.
- We determined that ScholarCopilot will serve as our official dataset for training and evaluating our pipeline. 
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