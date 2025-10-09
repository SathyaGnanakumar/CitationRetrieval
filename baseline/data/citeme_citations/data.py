from datasets import load_dataset, load_from_disk
dataset = load_from_disk(".")
dataset.save_to_disk("data/citeme_citations")
dataset["train"].to_csv("dataset/data.csv")
