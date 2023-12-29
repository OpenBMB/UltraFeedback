
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


system_prompt = "A chat between a curious user and an artificial intelligence expert. The expert gives helpful, specific, and concise answers to the user's questions."

feedback_prompt = \
"""Given my answer to an instruction, your role is to provide specific and constructive feedback for me. You should find the best way for me to learn from your feedback and improve my performance. 

You should consider multiple aspects of my answer, including helpfulness, truthfulness, honesty, and to what extent the answer follows instructions.
---

### Instruction
{instruction}

### Answer
{completion}
---

Please act as a teacher and provide specific and constructive feedback. Besides describing the weaknesses of the answer, you should also provide specific suggestions to guide me toward understanding how to improve. Please note, however, that your suggestions should help me better complete the instructions, but you should not introduce new requirements that are not mentioned in the instructions. Your feedback should focus on enhancing my ability to think critically and respond accurately. However, never explicitly provide the reference answer, nor do polite phrases be required. Only respond with concise feedback in chat style. Finally, score the overall quality of the answer from 1 to 10, where 1 is the worst and 10 is the best.

*Format*
### Feedback
[Your feedback]
Overall Score: [1-10]

---

### Feedback
"""

def get_eval(model, sys_prompt, user_prompt):
    try_num = 0
    while try_num < 10:
        try:
            response = openai.ChatCompletion.create(**{
                "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": 1024,
                    "top_p": 0.6,
                    "presence_penalty": 0,
                    "frequency_penalty": 0
            })
            return response["choices"][0]["message"]["content"].strip()
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(e)
            pass
    raise Exception("API Error")


def annotate(example):

    for i, completion in enumerate(example["completions"]):

        custom_system_prompt = completion["custom_system_prompt"] if completion["principle"] != "verbalized_calibration" else completion["custom_system_prompt"].split("For instance, ")[0].strip()
        response = get_eval("gpt-4-0613", system_prompt, feedback_prompt.format(instruction="\n".join([example["instruction"], "Note: " + custom_system_prompt]), completion=completion["response"]))
        
        response = response.split("\nOverall Score: ")
        assert len(response) == 2
        # critique, score = response[0].strip(), float(eval(response[1].split(".")[0].strip()))
        # example["completions"][i]["critique"] = critique
        # example["completions"][i]["overall_score"] = score if score > 1 else 10*score 
        critique, score = response[0].strip(), response[1].split(".")[0].strip()
        example["completions"][i]["critique"] = critique
        example["completions"][i]["overall_score"] = score if "/" not in score else float(eval(score.split("/")[0].strip()))

    return example


if __name__ == "__main__":

    subsets = ["sharegpt", "flan", "evol_instruct", "ultrachat", "truthful_qa", "false_qa"]

    for subset in subsets[:1]:
        with open(os.path.join("annotation", subset + ".json"), "r") as f:
            dataset = json.load(f)
        dataset = pd.DataFrame(dataset)
        dataset = datasets.Dataset.from_pandas(dataset)

        dataset_dict = []
        for data in tqdm(dataset, total=len(dataset), desc="Annotating"):
            dataset_dict.append(annotate(data))

        os.makedirs("annotation", exist_ok=True)
        result_path = os.path.join("annotation", subset + ".json")
        with open(result_path, "w") as f:
            json.dump([{k: v for k, v in data.items()} for data in dataset_dict], f, indent=4)