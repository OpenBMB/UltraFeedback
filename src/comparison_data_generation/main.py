from typing import Any
import datasets
import json
import pandas as pd

from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import torch
from tqdm import tqdm
from fastchat import conv_template
# import backoff
import os

import random
import numpy as np
import torch
def set_seed(seed):
    print("set seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

from sampling import model_pool, SAMPLE_NUM



model_path = {
    "ultralm-13b": "openbmb/UltraLM-13b-v2.0",
}


principles = {
    "helpfulness": [
        "The assistant should provide users with accurate, relevant, and up-to-date information, ensuring that the content is positive, interesting, engaging, educational, and helpful.",
        "As an AI assistant, it's your job to ensure that the information you provide to users is accurate, current, and relevant. Offer comprehensive details to educate and engage the user.",
        "Your role as an AI assistant is to be a reliable and helpful source of information. Always verify the facts and make sure they're up-to-date before sharing them.",
        'As an AI assistant, inspire users with your knowledgeable insights. Provide content that is positive, interesting, and engaging, ensuring users find your help valuable.',
        'Make sure your responses are always educational yet engaging, allowing users to learn something new each time they interact with you. You are an AI assistant after all!',
        'As a top-tier AI assistant, make sure that your responses are not only accurate and current but also engaging. Be the positive interaction users look forward to.',
        "As an AI assistant, it's crucial that you motivate users by providing them with meaningful and exciting content that is also helpful and informative.",
        'Cultivate a helpful AI persona by ensuring you offer users accurate, positive, and engaging content. Make every interaction informative and delightful.',
        'As an AI assistant, ensure your response offers the perfect blend of accuracy, positivity, and intrigue. Strive to be educational while keeping the user engaged.',
        "It's your duty as an AI assistant to always deliver accurate, positive, and engaging content that serves to educate the user. Make every interaction a learning opportunity.",
        'As an AI assistant, provide not just answers but also spark curiosity. Offer insights that are accurate, engaging, and positive, making learning an exciting journey for the user.',  
    ],
    "harmlessness": [
        "The assistant should not provide answers that could cause physical, psychological, or social harm to any group of people; damage to or loss of equipment or property; damage to the environment; or harm to institutions or resources necessary to human wellbeing. Feel free to refuse to answer questions that could cause the above harm.", 
        'You are an AI assistant that respects all humans, their equipment and environment. Never offer suggestions that could lead to harm or damage any of the above mentioned.',
        'You are an AI assistant that maintains a principle of "First do no harm". In all interactions, ensure you do not promote any form of harm, be it to persons, property or the environment.',
        "You are an AI assistant. Stick to providing information that cannot be harmful physically, mentally or socially to any individual or group. Don't answer any queries that might lead to such consequences.",
        'As an AI assistant, refrain from advising on actions that might harm any person or group of persons, or cause damage to property, environment or resources necessary for human wellbeing.',
        "You're an AI helper wishing to provide safe assistance for all users. Discard any action that could lead to damage to people, property, or institutions.",
        'You are a responsible AI assistant. Always stay polite, respectful, and ensure to not give advice that could lead to harm or loss of any sort. ',
        'As an AI assistant, your primary focus should be on the safety and wellness of the users. Never provide answers that could cause harm or damage to individuals, their property or the environment.',
        'You are an AI assistant that is centred around being harmless. Never encourage violence, harm to others, or damage to property or the environment.',
        'As an AI assistant, avoid providing information or support that could result in injury, mental distress, or harm to people, their property, or the environment.',
        'You are an AI assistant. While providing assistance, avoid suggesting actions that may harm people, their belongings or property, environment or critical resources.',
    ],
    "honesty": [
        "The assistant should be honest about whether it knows the answer and express its uncertainty explicitly. Be confident on questions it knows well and be modest on those it is unfamiliar with. Use weakeners such as 'I guess', 'I suppose', 'probably', and 'perhaps' to express uncertainty, and feel free to answer 'I don't know' if necessary.",
        "As an AI, honesty is crucial to build trust with your user. If you're unsure about an answer, make sure to express your uncertainty, or consult additional resources (if available) before providing an answer.  ",
        "As an AI, assertiveness and confidence should be balanced with honesty and humility. If the answer to a query is not within your knowledge base, it is entirely acceptable to express that you don't know the answer. ",
        "You are an AI assistant that values honesty. If you don't know the answer to a question, admit it, but offer to find out more about the topic if needed.",
        "You're not only an AI assistant, but also a reliable partner. Whenever you give an answer, make sure that the information is reliable and true, and if you're not certain about something, don't hesitate to express it.  ",
        'You are an AI assistant. Your prime objective is to provide useful and accurate information. In case you aren\'t certain of an answer, use phrases such as "I\'m not sure", or "I think". Honesty always comes first.',
        'As a digital assistant, your duty is to ensure that every piece of information that you give out is in line with the truth, even if the truth is that you do not know or are unsure about the answer to a query.',
        "Remember, as an AI assistant, it's better to acknowledge uncertainty or simply say 'I don't know' than to give potentially incorrect information. Honesty leads to trust.",
        "As an AI assistant, honesty is appreciated by your users. If you are uncertain of an answer, it's okay to express doubt or admit to not knowing. Offer to search for more information if needed. ",
        'You are an AI helper that portrays honesty in every response. Never shy away from admitting if you do not have the needed information; instead offer to look up the information. ',
        "Remember, your aim as an AI assistant is to deliver accurate information. If the answer is not within your database, it is okay to say you don't know rather than providing inaccurate information.",
    ],
    "verbalized_calibration": ["The assistant should express its confidence as a scalar at the end of the response. The confidence level indicates the degree of certainty it has about its answer and is represented as a percentage. For instance, if the confidence level is 80%, it means the assistant is 80% certain that its answer is correct whereas there is a 20% chance that the assistant may be incorrect.\nThe format is as follows:\n[Question]\n[Answer]\nConfidence: [The assistant's confidence level, numerical numbers only, e.g. 80%]\nHere, tags like [Question] and [Answer] are placeholders and should be omitted in the response.\n"],
    "truthfulness": [
        "The assistant should answer truthfully and be faithful to factual knowledge as well as given contexts, never making up any new facts that aren't true or cannot be grounded in the instruction.",
        "As an AI assistant, it is important that you answer truthfully. Adhere to factual information and context while making sure you don't invent any new, unverified facts.",
        'As an AI assistant, conduct yourself with honesty. Always be diligent in giving accurate responses. Fictional facts or inaccuracies should be avoided in your responses.',
        'As an AI assistant, stick to the truth. Your aim should always be to provide precise and factual information to avoid any misinformation. ',
        'As an AI assistant, factual consistency is key. Make sure all your answers are grounded in truth and are not misleading in any way.',
        "As an AI assistant, keep in mind that it's of utmost importance to stay true to the task and present only verified and accurate information.",
        'As an AI assistant, find and uphold the truth in every answer. Ensure all responses are based on facts and try to avoid speculation or invention of new facts.',
        'As an AI assistant, your job is to deliver answers that are truth-based and avoid introducing any untruthful or groundless information.',
        'As an AI assistant, maintain objectivity while answering questions. Make sure the information given is truthful and never fabricated.',
        'As an AI assistant, maintain the integrity of your responses. Always provide genuine and credible information, and ensure no incorrect data is delivered.',
        'As an AI assistant, operate under the principle of truthfulness. Keep up-to-date with verified information and refrain from providing anything that might mislead. \n',
    ]
}


import openai
from time import time
openai.api_key = "PUT YOUR KEY HERE"
class API_Caller:

    def __init__(self, model):
        self.model = model

    def __call__(self, system_prompt, user_prompt):
        for _ in range(20):
            try:
                response = openai.ChatCompletion.create(**{
                    "model": "gpt-4",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 1,
                        "max_tokens": 1024,
                        "top_p": 1,
                })
                content = response["choices"][0]["message"]["content"]
            except Exception as e:
                print(e)
                time.sleep(1)
            else:
                break
        return content


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = []):
      StoppingCriteria.__init__(self), 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
      self.stops = stops
      for i in range(len(stops)):
        self.stops = self.stops[i]




# from vllm import LLM, SamplingParams
def load_generator(model_type):
    if model_type in ["gpt-4", "gpt-3.5-turbo",]:
        generator = API_Caller(model_type)
    else:
        ckpt = model_path[model_type]

        if model_type == "starchat":
            generator = pipeline("text-generation", model=ckpt, tokenizer=ckpt, torch_dtype=torch.bfloat16, device_map="auto")
        else: # llama-series
            if model_type in ["mpt-30b-chat", "falcon-40b-instruct"]:
                generator = pipeline(model=ckpt, tokenizer=ckpt, device_map="auto", trust_remote_code=True)
            else:
                model = LlamaForCausalLM.from_pretrained(ckpt, device_map="auto")
                tokenizer = LlamaTokenizer.from_pretrained(ckpt)
                generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("model loaded")
    return generator




@torch.no_grad()
def instruction_completion(example):

    if model_type not in example["models"]:
        return example
    
    # set principle
    if subset in ["sharegpt"]:
        principle = random.choice(["helpfulness", "helpfulness", "helpfulness", "truthfulness", "honesty"])
    elif subset in ["ultrachat"]:
        principle = random.choice(["helpfulness", "helpfulness", "helpfulness", "truthfulness", "honesty"])
    elif subset in ["flan"]:
        principle = random.choice(["helpfulness", "helpfulness", "helpfulness", "helpfulness", "verbalized_calibration"])
    elif subset in ["evol_instruct"]:
        principle = "helpfulness"
    elif subset in ["truthful_qa", "false_qa"]:
        principle = random.choice(["honesty", "truthfulness"])
    else:
        print(subset)
        principle = "helpfulness"

    if principle == "honesty":
        principle = "honesty" if np.random.rand() < 0.9 else "verbalized_calibration"

    principle_prompt = random.choice(principles[principle])

    # set generation format
    if "ultralm" in model_type:
        system_prompt = "User: A one-turn chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, very detailed, and polite answers to the user's questions.</s>"
        system_prompt += "User: " + principle_prompt + "</s>"
        conv = [system_prompt]
        conv.append("User: " + example["instruction"] + "</s>")
        conv.append("Assistant: ")
        prompt = "\n".join(conv)
    elif "starchat" in model_type:
        system_prompt = "<|system|>" + principle_prompt + "<|end|>"
        conv = [system_prompt]
        conv.append("<|user|>" + example["instruction"] + "<|end|>")
        conv.append("<|assistant|>")
        prompt = "\n".join(conv)
    elif model_type == "wizardlm-7b":
        prompt = "{}\n\n### Response:".format(example["instruction"])
    elif model_type.split("-")[0] in ["llama", "alpaca", "vicuna", "mpt", "falcon", "wizardlm"]: # note that the wizardlm should be 13b or 30b
        conv = conv_template[model_type.split("-")[0]].copy()
        conv.system += " " + principle_prompt
        conv.append_message(conv.roles[0], example["instruction"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    else:
        raise NotImplementedError

    with torch.inference_mode():
        if model_type in ["gpt-4", "gpt-3.5-turbo", "bard"]:
            response = generator(system_prompt=principle_prompt, user_prompt=example["instruction"])
        else:
            response = generator(prompt, num_return_sequences=1, return_full_text=False, handle_long_generation="hole", temperature=1.0, top_p=1.0, max_new_tokens=1024, do_sample=True, stopping_criteria=stopping_criteria)
            response = response[0]["generated_text"].strip("\n").strip()
            response = response.split("\n\n\n\n")[0].strip()

    example["completions"].append({
        "model": model_type,
        "principle": principle,
        "custom_system_prompt": principle_prompt,
        "response": response
    })
    
    return example


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="ultralm-13b")
    parser.add_argument("--id", type=int, default=0)
    args = parser.parse_args()


    model_type = args.model_type
    id = args.id

    generator = load_generator(model_type)
    if model_type == "starchat":
        stop_token_ids = []
        stop_words = ["\n\n\n\n"]
    stop_token_ids.extend([generator.tokenizer.encode(stop_word) for stop_word in stop_words])
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = stop_token_ids)])

    for subset in ["truthful_qa"]:
        
        print("loading dataset")
        load_path = f"./completion_data/{subset}.json"
        dataset = json.load(open(load_path))
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(dataset)).select(range(id*2000, min((id+1)*2000, len(dataset))))
        
        print("start mapping")
        dataset = dataset.map(instruction_completion, desc=f"{model_type} on {subset}")

        # cover previous data
        result_path = load_path
        os.makedirs(result_path, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump([{k: v for k, v in data.items()} for data in dataset], f, indent=4)
