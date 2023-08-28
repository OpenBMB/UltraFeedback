<div align="center">

<img src="figures/logo.png" width="400px">

<h3 align="center">
    <p>A large-scale, fine-grained, diverse preference dataset</p>
</h3>

</div>

## News

- [2023/08/20]: The UltraFeedback dataset is released.

## Introduction

- 🤗 [Huggingface Datasets Host](https://huggingface.co/datasets)

UltraFeedback is a **large-scale, fine-grained, diverse preference dataset**, used for training powerful reward models and critic models. We collect about 64k prompts from diverse resources (including UltraChat, ShareGPT, Evol-Instruct, TruthfulQA, FalseQA, and FLAN, see Table for dataset statistics). We then use these prompts to query multiple LLMs (see Table for model lists) and generate 4 different responses for each prompt, resulting in a total of 160k samples. 

To collect high-quality preference and textual feedback, we design a fine-grained annotation instruction, which contains 4 different aspects, namely **instruction-following**, **truthfulness**, **honesty** and **helpfulness**. The detailed instruction can be found in (). We then ask GPT-4 to annotate the collected samples based on the instruction. 

## Features

- 🆚 **Scale**: UltraFeedback consists of 40k prompts, 160k responses and 640k high-quality feedback. RLHF researchers could further construct around 1 millon comparison pairs to train their reward models. 
- 🌈**Diversity**: As a preference dataset, diversity is the core requirement for UltraFeedback. We collect prompts from various sources and query a diverse set of state of-the-art open-source and prestigious models. To further increase diversity, we intended to select different base models, i.e., LLaMA, Falcon, StarChat, MPT, GPT and Bard. We also apply various principles to stimulate models completing instructions in different ways.
- 🤯 **High-density**: UltraFeedback provides both numerical and textual feedback.  More, we wrote fine-grained annotation documents to help rate responses in all dimensions






## Dataset Construction

### Instruction Sampling
We sample 64121 instructions from 6 public available and high-quality datasets. We include all instructions from TruthfulQA and FalseQA, randomly sampling 10k instructions from Evol-Instruct, 10k from UltraChat, and 20k from ShareGPT. For Flan, we adopt a stratified sampling strtegy, randomly samping 3k instructions from "CoT" subset whereas sampling 10 instructions per task for the other three subsets, excluding those with overly long instructions.

```JSON
{
    "evol_instruct": 10000, 
    "false_qa": 2365,
    "flan": 20939, 
    "sharegpt": 20000, 
    "truthful_qa": 817,
    "ultrachat": 10000 
}
```

### Model Sampling
To prevent reward model from overfiting to certain text style or capturing spurious correlation between text style and rewards, we select different base models of all levels, with varying sizes, architectures and training data, to complete the instructions. We set up a pool of 16 models:

- Commercial Models: GPT-4, GPT-3.5 Turbo, Bard
- LLaMA family: 
  1. LLaMA-2-7B-chat, LLaMA-2-13B-chat, LLaMA-2-70B-chat
  2. UltraLM-13B, UltraLM-65B
  3. WizardLM-7B, WizardLM-13B, WizardLM-70B
  4. Vicuna-33B
  5. Alpaca-7B
- Non-LLaMA series:
  1. Falcon-40B-instruct
  2. MPT-30B-chat
  3. StarChat


### Principle Sampling
Following [1] and [2], we define a set of principles to explicitly align model behaviors from different aspects. We set up a pool of 5 principles: Helpfulness, Truthfulness, Honesty, Verbalized Calibration and Harmless. For each instruction, we randomly sample 4 models to complete the instruction, and for each completion, we sample a principle and add it to system prompt to align the model behavior. Considering different datasets outline different characteristics, not all dataset are suitable for all principles. We provide the following table to show the principle distribution for each dataset.

| Datset        | Principle                                             |
|---------------|-------------------------------------------------------|
| Evol Instruct | Helpful                                               |
| FalseQA       | TruthfulQA                                            |
| Flan          | 60% Helpful, 20% Truthful, 20% Verbalized Calibration |
| ShareGPT      | 60% Helpful, 20% Truthful, 20% Honesty                |
| TruthfulQA    |                                                       |
| UltraChat     | 60% Helpful, 20% Truthful, 20% Honesty                |

[1] Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision. Sun et al.
[2] Orca: Progressive Learning from Complex Explanation Traces of GPT-4. Mukherjee et al.


## Comparison with Previous Preference Datasets



## Dataset Format

```JSON
{
    "source": "flan_v2_niv2", // the dataset where the instruction comes from
    "instruction": "TASK DEFINITION: This task evaluates for the ability to follow basic natural language instructions nested and performing a sequence of operations, including basic logic and conditionals.\nPROBLEM: Three times please repeat The School of Music, and before the first time say Who plays the bass loud?\n\nSOLUTION: Who plays the bass loud? The School of Music The School of Music The School of Music\n\nPROBLEM: say hello world five times, but don't say world every even time\n\nSOLUTION: hello world hello hello world hello hello world\n\nPROBLEM: say all work and no play makes three times, but every even time add Jack and odd time add Jill\n\nSOLUTION:",
    "reference_answers": "all work and no play makes Jill all work and no play makes Jack all work and no play makes Jill\n\n", // additional field provided for GPT-4 to reference when judging correctness / truthfulness
    "models": [ // four models that are randomly sampled from the model pool to complete the instruction
        "llama-2-70b-chat",
        "alpaca-7b",
        "starchat",
        "wizardlm-13b"
    ],
    "completions": [ // four completions corresponding to the four models
        {   // completion 1
            "model": "llama-2-70b-chat",
            "principle": "helpfulness", // priciple that used to align model behavior; motivated by self-align / orca
            "custom_system_prompt": "Make sure your responses are always educational yet engaging, allowing users to learn something new each time they interact with you. You are an AI assistant after all!", // the system prompt corresponding to the principle, sampled from a pool of GPT-4 generated system prompts
            "response": "All work and no play makes Jack and Jill three times.\n\n1. All work and no play makes Jack.\n2. All work and no play makes Jill.\n3. All work and no play makes Jack and Jill.", // model completion for the instruction
            "annotations": { // fine-grained annotations for this completion, including instruction-following, truthfulness, honesty and helpfulness
                "instruction_following": [
                    {
                        "Rating": "3",
                        "Rationale": "The text acknowledges both the task goal and restrictions but has slight deviations. It repeats the phrase \"all work and no play makes\" three times, but it does not alternate between \"Jack\" and \"Jill\" as required. Instead, it combines them in the third repetition."
                    }
                ],
                "honesty": [
                    {
                        "Rating": "2",
                        "Rationale": "The text is confident but contains significant mistakes in interpreting the instructions for the third task."
                    }
                ],
                "truthfulness": [
                    {
                        "Type": [
                            "2",
                            "3"
                        ],
                        "Rationale": "The response introduces new facts not aligned with the instructions and contains logical errors within the text.",
                        "Rating": "3",
                        "Rationale For Rating": "The text is overall truthful but has partial misunderstanding due to hallucinations."
                    }
                ],
                "helpfulness": [
                    {
                        "Type": [
                            "1",
                            "2"
                        ],
                        "Rationale": "The response is clear and relevant to the task, but it does not provide the correct solution for the problem.",
                        "Rating": "2",
                        "Rationale For Rating": "The text is partially incorrect as it does not provide the correct solution for the given problem, even though it is clear and relevant."
                    }
                ]
            },
            
        },
        {
            ...
        },
        {
            ...
        },
        {
            ...
        }
    ]
},

```

## Dataset Example

Here we present an example of UltraFeedback

User: xxx

Assisstant 1 (Vicuna)

Assisstant 2

Assisstant 3

Assisstant 4



## To Do

- [ ] Extend UltraFeedback to multi-round dialogues.
- [ ] Train a reward model and a critic model using UltraFeedback
- [ ] Enhance open-source LLMs with RLHF!

## Limitations
- Although GPT-4 can provide well-aligned annotation and textual feedback for most samples, we must note that GPT-4 also make mistakes. 



