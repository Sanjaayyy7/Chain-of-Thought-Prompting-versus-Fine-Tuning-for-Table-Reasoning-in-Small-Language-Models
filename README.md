# Chain-of-Thought Prompting versus Fine-Tuning for Table Reasoning in Small Language Models

ECS 111 Final Project, University of California, Davis

This project asks a simple question. When you can only use small, free language models, does chain-of-thought prompting or supervised fine-tuning help a model answer questions about tables? We tested two small models, FLAN-T5-base (250M) and FLAN-T5-large (780M), on the WikiTableQuestions dataset. The result is a negative one. Neither chain-of-thought prompting nor fine-tuning beat a plain zero-shot baseline. The best system was FLAN-T5-large prompted with no examples, at 0.241 exact match. The fine-tuned models also failed to transfer to a second dataset, TabFact, where they scored below the 0.551 majority-class floor.

## Where to start

- The paper is `report/FINAL_PAPER.pdf`.
- The approved proposal is `proposal/Project_Proposal.pdf`.
- The runnable report notebook is `report/FINAL_REPORT.ipynb`.
- The raw numbers are in `results/`.
- The code is in `src/`, with tests in `tests/`.
- The Colab reproducibility notebooks are in `notebooks/`.
- The AI-use citations are in `AI_Citations/`, one file per team member.

## Folder guide

- `report/`: the final paper (PDF and LaTeX source), the report notebook, and figures.
- `AI_Citations/`: each team member's AI-use disclosure, one file per member.
- `proposal/`: the approved project proposal.
- `src/`: the project code, including data loading, prompts, training, evaluation, and analysis.
- `scripts/`: helper scripts to run the experiments and build the figures and report.
- `tests/`: the automated tests that check the code.
- `notebooks/`: the Colab notebooks, one per experiment condition plus a full run.
- `results/`: the raw output numbers from every run.

## Reproducing the results

First set up a virtual environment and install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To run the experiments, use the per-condition Colab notebooks in `notebooks/`, or run everything locally with:

```bash
python scripts/run_all_local.py
```

To rebuild the paper:

```bash
cd report && latexmk -pdf FINAL_PAPER.tex
```

## Authors

Adisesh Venkatesh, Amar Thota, Nikhil Karthikeyan, Sanjay Manivasagam, Anant Madhok
