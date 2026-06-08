# AI Citation

## Sanjay Manivasagam's contribution

I led and coordinated the project as lead author. I ran the generalization test, the error analysis, and the statistics, and pulled the final report together.

## AI use

I used Claude (Anthropic) to help me understand the project after reading our approved proposal, and to help with some of my analysis code. I shared the proposal and asked questions about table question answering, the generalization test from WikiTableQuestions to TabFact, how to categorize the model's errors, and how to read the statistics (significance testing and Cohen's kappa). Claude also helped me draft and debug some of the analysis and statistics code.

*Shared conversation:* [Claude chat](https://claude.ai/share/c4d80292-a61f-4615-ac49-996bd1691080)

## Nikhil Karthikeyan's contribution

Nikhil built the seq2seq training harness, ran the answers-only fine-tuning across both seeds, and co-designed and applied the chain-quality rating rubrics.

## AI use

I used Claude (Anthropic) to help me understand the project from the ground up after reading our approved proposal. I shared the proposal and asked questions about what seq2seq and fine-tuning mean mechanically, what a "training harness" consists of in practice, the optimizer and hyperparameter settings (AdamW, learning rate, gradient accumulation), why running with two random seeds matters for reporting results, what chain-quality ratings are measuring and why exact match alone is insufficient, and why chain-of-thought prompting can hurt smaller models. I used this to clarify concepts and plan my work - not to generate project code or write report text.

*Shared conversation:* [Claude chat] (https://claude.ai/share/4b12fc2b-47ae-4c39-8ecc-d86c6d914170)


## Anant Madhok's Contribution 
I was responsible for the rule-based reasoning-trace template generator and Condition C of the project. I generated reasoning traces for training data, ran the fine-tuning with reasoning traces experiments across two seeds, reused the shared trainer, and contributed approximately five Chain-of-Thought exemplars.
*shared conversation* https://chatgpt.com/share/6a1cb605-22f0-83e8-a3fd-7a2233a9d037

## AI Use

I used ChatGPT to help me understand and refine the project after reviewing our approved proposal. I discussed model selection (including the Flan-T5 family), fine-tuning approaches, reasoning-trace generation, chain-quality evaluation, Cohen's kappa, McNemar's test, and methods for comparing trained versus untrained models. I also used ChatGPT to explore potential evaluation metrics, dataset considerations, and experimental-design decisions. I used these conversations to clarify concepts, evaluate methodological choices, and plan my work.
