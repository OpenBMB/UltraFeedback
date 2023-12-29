from typing import List, Dict, Optional, Any
from datasets import load_dataset


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
{critique}
Overall Score: 
"""

def get_eval(model, sys_prompt, user_prompt):
    try_num = 0
    while try_num < 10:
        try:
            response = openai.ChatCompletion.create(**{
                "model": model,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": 1,
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



def calculate_average_rating(annotations):
    ratings = [int(aspect['Rating']) for aspect in annotations.values() if 'Rating' in aspect and aspect['Rating'] != "N/A"]
    return sum(ratings) / len(ratings) if ratings else None


def check_score(completion):
    if completion["fine-grained_score"] <= 2:
        return 2 # should flip
    elif completion["fine-grained_score"] <= 4:
        return 1 # re-annotate
    else: 
        return 0 # remain

def process_completions(example):
    global count_global, count_10
    count = {0:0,1:0,2:0}
    num_1 = sum(completion["overall_score"]==1 for completion in example["completions"])
    for completion in example["completions"]:
        completion.update({"fine-grained_score": calculate_average_rating(completion["annotations"])})
        if completion["overall_score"] == 10:
            flag = check_score(completion)
            count[flag] += 1
            if flag > 0:
                if flag == 2:
                    completion["overall_score"] = 1
                elif flag == 1:
                    # re-annotate
                    custom_system_prompt = completion["custom_system_prompt"] if completion["principle"] != "verbalized_calibration" else completion["custom_system_prompt"].split("For instance, ")[0].strip()
                    response = get_eval("gpt-4-0613", system_prompt, feedback_prompt.format(instruction="\n".join([example["instruction"], "Note: " + custom_system_prompt]), completion=completion["response"], critique=completion["critique"]))

                    if "/" in response:
                        response = response.split("/")[0].strip()
                    score = float(eval(response.strip()))
                    completion["overall_score"] = score

    num_2 = sum(completion["overall_score"]==1 for completion in example["completions"])
    assert num_2 - num_1 >= count[2]
    
    for k in count.keys():
        count_global[k] += count[k]
    
    return example

if __name__ == "__main__":

    # Load the dataset
    dataset = load_dataset("openbmb/UltraFeedback")["train"]
    count_global = {0:0,1:0,2:0}
    dataset = dataset.map(process_completions, load_from_cache_file=False)
    print(count_global) 
    print("{} completions with an overall_score of 10, {} of them remained, {} re-annotated, and {} flipped.".format(sum(count_global.values()), count_global[0], count_global[1], count_global[2]))
    dataset.save_to_disk("UltraFeedback")