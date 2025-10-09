# CitationRetrieval

tasks:
- each person, find 1 excerpt from 10 different papers. remove the source from each excerpt and add to dataset. overall, we have 60 new data points to add to the dataset of 130 we already found
- test 2 LLMs: gemini, claude (potentially)


Dataset : https://huggingface.co/datasets/bethgelab/CiteME

To ensure that CiteMe is a challenging and robust dataset, we
remove all dataset instances that GPT-4o can correctly answer. Filtering datasets by removing the
samples that a strong model can correctly answer was previously done in Bamboogle [71] and the
Graduate-Level Google-Proof Q&A Benchmark [73]. In our filtering process, GPT-4o was used with
no Internet access or any other external tools. Therefore, it could answer only correctly specified
papers that it memorized from its training process. We ran each sample through GPT-4o five times to
cover its different outcomes. In the end, we filtered out 124 samples, leaving 130 samples in total.