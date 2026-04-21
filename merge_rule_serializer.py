import json

with open("gemma-4-E4B-it/tokenizer.json", "r") as file:
    contents = json.load(file)
print(contents.keys())
