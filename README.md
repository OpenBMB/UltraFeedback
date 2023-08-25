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

UltraFeedback is a **large-scale, fine-grained, diverse preference dataset**, used for training powerful reward models and critic models. We collect about 40k prompts from diverse resources (including UltraChat, ShareGPT, Evol-Instruct, HH-RLHF, TruthfulQA, FalseQA, and FLAN, see Table for dataset statistics). We then use these prompts to query multiple LLMs (see Table for model lists) and generate 4 different responses for each prompt, resulting in a total of 160k samples. 

To collect high-quality preference and textual feedback, we design a fine-grained annotation instruction, which contains 4 different aspects, namely **instruction-following**, **truthfulness**, **honesty** and **helpfulness**. The detailed instruction can be found in (). We then ask GPT-4 to annotate the collected samples based on the instruction. 

## Features

- 🆚 **Scale**: UltraFeedback consists of 40k prompts, 160k responses and 640k high-quality feedback. RLHF researchers could further construct around 1 millon comparison pairs to train their reward models. 
- 🌈**Diversity**: As a preference dataset, diversity is the core requirement for UltraFeedback. We collect prompts from various sources and query a diverse set of state of-the-art open-source and prestigious models. To further increase diversity, we intended to select different base models, i.e., LLaMA, Falcon, StarChat, MPT, GPT and Bard.
- 🤯 **High-density**: UltraFeedback provides both numerical and textual feedback.  More, we wrote fine-grained annotation documents to help rate responses in all dimensions

## Dataset Format

```JSON
{

}

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



