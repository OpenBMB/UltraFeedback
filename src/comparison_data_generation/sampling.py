from datasets import Dataset
import pandas as pd
import random
import json


model_pool = [
    "gpt-4", "gpt-3.5-turbo", "bard", 
    "ultralm-65b", "wizardlm-30b", "vicuna-33b", "llama-2-70b-chat", 
    "ultralm-13b", "wizardlm-13b", "llama-2-13b-chat", 
    "wizardlm-7b", "alpaca-7b", "llama-2-7b-chat", 
    "falcon-40b-instruct", "starchat", "mpt-30b-chat", "pythia-12b"
]


if __name__ == "__main__":

    for subset in ["truthful_qa"]:
        # dataset = json.load(open(f"./completion_data/{subset}.json"))
        dataset = pd.read_json(f"./completion_data/{subset}.json", lines=True)
        dataset = Dataset.from_pandas(pd.DataFrame(dataset))
        dataset = dataset.map(lambda x: {"models": random.sample(model_pool, 1), "completions": []}, desc=subset)

        with open(f"./completion_data/{subset}.json", "w") as f:
                json.dump([{k: v for k, v in data.items()} for data in dataset], f, indent=4)