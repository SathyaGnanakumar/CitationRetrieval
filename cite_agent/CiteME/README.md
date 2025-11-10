---
license: cc-by-sa-4.0
language:
- en
tags:
- citation
- scientific paper
- grounding
---

# **CiteME is a benchmark designed to test the abilities of language models in finding papers that are cited in scientific texts.** #

## Dataset Structure

The dataset is provided in CSV format and includes the following columns:

| Column Name          | Description                                          |
|----------------------|------------------------------------------------------|
| `id`                 |            A unique identifier for each paper, used consistently across all experiments.         |
| `excerpt`            |            The text excerpt from the source paper that describes the target paper.                |
| `target_paper_title` |            The title of the paper that is being cited in the excerpt.                            |
| `target_paper_url`   |            The URL linking to the target paper.                                                   |
| `source_paper_title` |            The title of the paper from which the excerpt is taken.                               |
| `source_paper_url`   |            The URL linking to the source paper.                                                   |
| `year`               |            The publication year of the source paper.                                              |
| `split`              |            Indicates the dataset split: `train` or `test`.                                       |

### Example


| id  | excerpt                                                                                           | target_paper_title                   | target_paper_url                       | source_paper_title                 | source_paper_url                     | year | split |
|-----|---------------------------------------------------------------------------------------------------|--------------------------------------|----------------------------------------|------------------------------------|--------------------------------------|------|-------|
| 1   | "As demonstrated in [Smith et al., 2020], the proposed method improves accuracy significantly."    | "Improving Accuracy in ML Models"    | https://example.com/target1            | "Advancements in Machine Learning" | https://example.com/source1           | 2020 | train |
| 2   | "Building upon the framework introduced by [Doe, 2019], we extend the applicability to NLP tasks." | "Framework for NLP Applications"     | https://example.com/target2            | "Foundations of NLP"               | https://example.com/source2           | 2019 | test  |


### Load the Dataset

You can load the dataset using popular data processing libraries such as `pandas`.

```python
import pandas as pd

dataset = pd.read_csv('DATASET.csv')
print(dataset.head())
```

## Dataset Structure

### If you find our work helpful, please use the following citation:
```
@misc{press2024citeme,
    title={CiteME: Can Language Models Accurately Cite Scientific Claims?},
    author={Ori Press and Andreas Hochlehnert and Ameya Prabhu and Vishaal Udandarao and Ofir Press and Matthias Bethge},
    year={2024},
    eprint={2407.12861},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2407.12861}
}
```