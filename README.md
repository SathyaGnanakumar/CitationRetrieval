# CitationRetrieval

Week 1:
- Accumulate training data from ArXiv based on CiteMe Benchmark. Begin TF/IDF and BM25 baseline model development on retrieved data.

tasks:
- each person, find 1 excerpt from 10 different papers. remove the source from each excerpt and add to dataset. overall, we have 60 new data points to add to the dataset of 130 we already found
- test 2 LLMs: gemini, claude (potentially)

Dataset : https://huggingface.co/datasets/bethgelab/CiteME

Background

To ensure that CiteMe is a challenging and robust dataset, we remove all dataset instances that GPT-4o can correctly answer. Filtering datasets by removing the samples that a strong model can correctly answer was previously done in Bamboogle [71] and the Graduate-Level Google-Proof Q&A Benchmark [73]. In our filtering process, GPT-4o was used with no Internet access or any other external tools. Therefore, it could answer only correctly specified papers that it memorized from its training process. We ran each sample through GPT-4o five times to cover its different outcomes. In the end, we filtered out 124 samples, leaving 130 samples in total.

CiteME measures how well models can identify which scientific paper a given statement (excerpt) should cite.
Each excerpt in the dataset is curated to satisfy four strict criteria:
- Attributable – The citation directly supports the claim.
- Unambiguous – Only one paper could reasonably be cited.
- Non-Trivial – The excerpt does not include author names or title keywords.
- Reasonable – The excerpt is coherent and provides sufficient context.

Additionally, easy examples that a strong LLM can recall from memory are filtered out through closed-book GPT-4o evaluation—mirroring the filtering procedure used in the CiteME paper itself.

Data Cleaning:
- We create a script called `data_cleaning.py` that evaluates the new examples we are adding to the 
CiteMe benchmark with the same criteria posed in the paper. We run GPT-4o offline so it only can evaluate
citations based on its internal knowledge and cannot refer to the internet. 

data.csv
   │
   ├─► evaluate_criteria() → Uses GPT-4o to assess each excerpt on the four CiteMe criteria
   │
   ├─► filtering_stage()   → Runs closed-book filtering to check if the model can recall the cited paper
   │
   └─► verified_examples.json / .csv -> output verified citations

31/130 examples survived closed-book filtering on GPT-4o based on the current script paremeters

Do we just keep the 31 verified examples or stick with the 130 examples from CiteMe? Ask Dr. Goldstein about this and whether we need to relax the criteria + find more examples to augment these new verified examples. 
