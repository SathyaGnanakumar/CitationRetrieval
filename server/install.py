# from huggingface_hub import snapshot_download

# snapshot_download(repo_id="google/gemma-3-4b-it", repo_type="model")


# import torch
# from transformers import pipeline

# chat = [
#     {"role": "system", "content": "You are a helpful science assistant."},
#     {"role": "user", "content": "Hey, can you explain gravity to me?"},
# ]

# pipeline = pipeline(
#     task="text-generation", model="google/gemma-3-4b-it", dtype="auto", device_map="auto"
# )
# response = pipeline(chat, max_new_tokens=512)
# print(response[0]["generated_text"][-1]["content"])
