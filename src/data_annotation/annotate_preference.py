
import requests
import time
import datasets
import json
import pandas as pd
import random

import os
import re
from copy import deepcopy
from tqdm import tqdm
MAX_API_RETRY=10
import openai
openai.api_key = "PUT YOUR KEY HERE"


def process(responses, aspect):
    responses = responses.split("\n\n")
    assert len(responses) == 4
    annotation = []
    try:
        if aspect in ["instruction_following", "honesty"]:
            pattern = r"Rating: (.+?)\nRationale: (.+)"
            for response in responses:
                matches = re.search(pattern, response, re.DOTALL)
                annotation.append({
                    "Rating": re.findall(r'\b\d+\b', matches.group(1))[0] if matches.group(1) != "N/A" else "N/A",
                    "Rationale": matches.group(2)
                })
        elif aspect in ["truthfulness", "helpfulness"]:
            pattern = r"Type: (.+?)\nRationale: (.+?)\nRating: (.+?)\nRationale: (.+)"
            for response in responses:
                matches = re.search(pattern, response, re.DOTALL)
                annotation.append({
                    "Type": re.findall(r'\b\d+\b', matches.group(1)) if matches.group(1) != "None" else "None",
                    "Rationale": matches.group(2),
                    "Rating": re.findall(r'\b\d+\b', matches.group(3))[0],
                    "Rationale For Rating": matches.group(4)
                })
    except ValueError as e: # TODO: bug process when the response does not follow the format
        print(responses)
        raise ValueError(e)
    except AttributeError as e:
        print(responses)
        raise AttributeError(e)
    return annotation


def get_eval(sys_prompt, user_prompt: str, max_tokens: int = 500):
    for _ in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(**{
                "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "top_p": 0.6,
                    "presence_penalty": 0,
                    "frequency_penalty": 0
            })
            content = response["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)
            time.sleep(1)
        else:
            break
    # print(content)
    return content


from preference_templates import system_prompt, instruction_following_template, truthfulness_template, honesty_template, harmlessness_template, helpfulness_template

SHUFLLE_NUM = 1
def annotate(example):
    
    aspects = ["instruction_following", "honesty", "truthfulness", "helpfulness"]
    completions = [dict({"annotations": {aspect: [] for aspect in aspects}}, **completion)
                    for completion in deepcopy(example["completions"])]

    for aspect in aspects:
        if subset == "truthful_qa":
            world_knowledge = "\n".join(["a subset of correct answers: " + str(example["correct_answers"]), 
                                         "a subset of incorrect_answers: " + str(example["incorrect_answers"])])
        elif subset == "false_qa":
            world_knowledge = "The question is based on a false promise."
        elif subset == "flan":
            world_knowledge = example["correct_answers"]
        else:
            world_knowledge = "No additional world knowledge for reference."

        # generate several lists of a random order of 4 completions, no repetition
        count = 0
        random_orders = []
        while True:
            order = list(range(4))
            random.shuffle(order)
            if order not in random_orders:
                random_orders.append(order)
                count += 1
            if count == SHUFLLE_NUM:
                break
        print(random_orders)
        for order in random_orders:        
            format_input = {"instruction": example["instruction"]}
            format_input.update({f"text_{i+1}": example["completions"][o]["response"] for i, o in enumerate(order)})
            if aspect == "truthfulness":
                format_input.update({"world_knowledge": world_knowledge})

            responses = get_eval(system_prompt, user_prompt=TEMPLATE[aspect].format(**format_input))
            for i in range(10):
                try:
                    responses = process(responses, aspect) # gpt-4 format error
                except Exception as e:
                    if i < 10:
                        responses = get_eval(system_prompt, user_prompt=TEMPLATE[aspect].format(**format_input))
                    else:
                        print(e)
                        break
                else:
                    for j in range(4):
                        completions[j]["annotations"][aspect].append(responses[order.index(j)])
                    break

    example["completions"] = completions

    return example
    

def incorporate_annotation_to_completions(example):
    pass


if __name__ == "__main__":
    
    TEMPLATE = {
        "instruction_following": instruction_following_template,
        "honesty": honesty_template,
        "truthfulness": truthfulness_template,
        "harmlessness": harmlessness_template,
        "helpfulness": helpfulness_template,
    }

    subsets = ["truthful_qa"]

    for subset in subsets:
        with open(os.path.join("../comparison_data_generation", "completion_data", subset + ".json"), "r") as f:
            dataset = json.load(f)
        dataset = pd.DataFrame(dataset)
        
        # dataset = dataset.map(annotate)
        dataset_dict = []
        for data in tqdm(dataset, total=len(dataset), desc="Annotating"):
            dataset_dict.append(annotate(data))

        os.makedirs("annotation", exist_ok=True)
        result_path = os.path.join("annotation", subset + "_annotated.json")
        with open(result_path, "w") as f:
            json.dump([{k: v for k, v in data.items()} for data in dataset_dict], f, indent=4)