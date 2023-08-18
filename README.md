<div align="center">

<img src="figures/logo.png" width="400px">

<h3 align="center">
    <p>A large-scale, fine-grained, diverse preference dataset</p>
</h3>

</div>

## News

- August xx, 2023: The UltraFeedback dataset is released.

## Introduction

- 🤗 [Huggingface Datasets Host](https://huggingface.co/datasets)

UltraFeedback is a **large-scale, fine-grained, diverse preference dataset**, used for training powerful reward models and critic models. We collect about 40k prompts from diverse resources (including UltraChat, ShareGPT, Evol-Instruct, HH-RLHF, TruthfulQA, FalseQA, and FLAN, see Table for dataset statistics). We then use these prompts to query multiple LLMs (see Table for model lists) and generate 4 different responses for each prompt, resulting in a total of 160k samples. 

To collect high-quality preference and textual feedback, we design a fine-grained annotation instruction, which contains 4 different aspects, namely **instruction-following**, **truthfulness**, **honesty** and **helpfulness**. The detailed instruction can be found in (). We then ask GPT-4 to annotate the collected samples based on the instruction. 

## Dataset Format

```JSON
{

}

```

## To Do
- [ ] Extend UltraFeedback to multi-round dialogues.
- [ ] Use UltraFeedback to train a reward model and a critic model.
- [ ] Perform RLHF to enhance existing open-source LLMs!

## Limitations
- Although UltraFeedback.



