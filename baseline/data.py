from datasets import load_dataset

dataset = load_dataset("bethgelab/CiteME")
dataset.save_to_disk("data/citeme_citations")