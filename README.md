# Fine-Tuning Large Language Models for Character Consistency Using Synthetic Datasets

This repository contains the code, prompts, and resources accompanying the paper **"Fine-Tuning Large Language Models for Character Consistency Using Synthetic Datasets"**.

📄 The full paper is available in this repository: [`Finetuning_Large_Language_Models_for_Character_Consistency.pdf`](./Finetuning_Large_Language_Models_for_Character_Consistency.pdf)

The project investigates whether a small language model (LLaMA 3.2 3B/7B Instruct), adapted via parameter-efficient fine-tuning (LoRA), can reproduce the emotionally coherent and personality-consistent behavior of a fictional character — Harry Potter — with fidelity comparable to a large foundation model such as GPT-4o.

## Overview

The pipeline consists of three stages:

1. **Synthetic data generation** — *Harry Potter and the Philosopher's Stone* is segmented into chapters. GPT-4o extracts narrative events, involved characters, and their emotional/interpersonal dynamics from each chapter. These structured event contexts are then used to generate multi-turn dialogues between Harry Potter and a user (total generation cost: ~10€ via the OpenAI API).
2. **Fine-tuning** — The synthetic dialogue corpus (569 formatted dialogues) is used to fine-tune LLaMA 3.2 Instruct with LoRA adapters using the Hugging Face `transformers` and `peft` libraries.
3. **Evaluation** — The fine-tuned model and GPT-4o receive identical system prompts and event contexts across 12 unseen test scenarios. Outputs are compared qualitatively along coherence, faithfulness to character, and naturalness of behavior.

## Repository Structure

```
hs2025/
├── character_llm/                  # Shared prompt and database utilities
│   ├── PromptsCharacterLLM.py      # Prompt classes + Pydantic output validation
│   └── database/                   # SQLite persistence layer for chapters, events, dialogues
│       ├── character_llm.db
│       ├── character_llm_db.py
│       ├── models.py
│       └── read_write_db.py
├── synthetic_dialogue/             # Stage 1: synthetic dataset generation
│   ├── code/
│   │   └── dialog_generation.py    # Event extraction & dialogue generation via GPT-4o (LangChain)
│   └── resources/
│       ├── harry_potter_books.csv  # Source text, segmented by book/chapter
│       ├── hpd/                    # HPD dataset resources (per-chapter character data)
│       │   ├── char_attributes/    # Character attributes per chapter (Book 1)
│       │   └── char_relations/     # Character relations per chapter (Book 1)
│       └── prompts/
│           ├── extract_events              # Prompt: extract events, tone & dynamics per chapter
│           └── generate_dialogue_for_event # Prompt: generate Harry–user dialogue for one event
├── finetuning/                     # Stage 2: LoRA fine-tuning
│   ├── finetuning.ipynb            # Full pipeline: data loading, tokenization, training, export
│   ├── finetuning_reqs.txt         # Dependencies for the fine-tuning environment
│   └── images/
├── requirements.txt                # Dependencies for dataset generation
└── README.md
```

## Method Summary

### Synthetic Data Generation
- **Model:** GPT-4o (via `langchain-openai`), temperature 0.7
- **Two-step prompting:** (1) extract emotionally grounded events per chapter, (2) generate persona-consistent multi-turn dialogues per event
- Structured outputs are validated against Pydantic models and persisted in a SQLite database

### Dataset Preparation
- Dialogues are formatted using the LLaMA 3.2 chat template (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`, …) with explicit roles for system, user, and *Harry Potter*
- **Conversation windowing:** overlapping windows of 20–30 messages (30–50% overlap) plus a mix of full-length conversations and short exchanges for robust generalization
- **Context length:** token-length analysis (histogram + CDF) led to a maximum sequence length of **1250 tokens**, covering ~80% of all samples

### Fine-Tuning (LoRA)
| Parameter | Value |
|---|---|
| Base model | LLaMA 3.2 Instruct (3B/7B) |
| Task type | Causal LM |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj` |
| Learning rate | 2e-4 (cosine decay) |
| Effective batch size | 32 (4 × 8 gradient accumulation) |
| Epochs | up to 100, with custom early stopping |
| Optimizer | `adamw_8bit` |

Training was performed on an NVIDIA H100 SXM; with reduced context lengths, training is also feasible on a single consumer GPU (e.g., RTX 4090, 24 GB VRAM). After training, LoRA weights are merged into the base model for inference.

### Evaluation
12 unseen scenarios across four categories:
- **A — Emotional Inner States** (introspection, self-doubt)
- **B — Social Interaction** (everyday conversational tone)
- **C — Conflict and Stress** (behavior under pressure)
- **D — Moral Decisions and Reflection** (value-driven choices)

Each response was rated (weak / acceptable / strong) on **coherence**, **faithfulness to character**, and **naturalness of behavior**. Full side-by-side comparisons are provided in the paper's appendix.

## Getting Started

### Requirements
- Python 3.10+
- An OpenAI API key (for synthetic data generation)
- A CUDA-capable GPU (for fine-tuning)

### Installation

```bash
git clone https://github.com/Dominik-CH/hs2025.git
cd hs2025
pip install -r requirements.txt
```

For the fine-tuning environment:

```bash
pip install -r finetuning/finetuning_reqs.txt
```

### 1. Generate the Synthetic Dataset

Set your OpenAI API key (e.g., in a `.env` file), then run the generation script:

```bash
cd synthetic_dialogue/code
python dialog_generation.py
```

Extracted events and generated dialogues are stored in the SQLite database under `character_llm/database/`.

### 2. Fine-Tune the Model

Open and run the notebook:

```bash
jupyter notebook finetuning/finetuning.ipynb
```

The notebook covers dataset loading, token-length analysis, tokenization and padding, LoRA configuration, training with early stopping, and merging/exporting the final model.

## Key Results

- The LoRA fine-tuned model preserves stylistic and emotional continuity broadly comparable to GPT-4o, despite using a fraction of the parameters and compute.
- The complete synthetic dataset cost roughly **10€** to generate.
- Validation loss is a weak proxy for persona alignment: many stylistically valid responses exist per prompt, so qualitative evaluation of coherence, character faithfulness, and naturalness is used instead.

## References

Key resources used in this project:

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
- Chen et al., *Large Language Models Meet Harry Potter: A Dataset for Aligning Dialogue Agents with Characters* (HPD dataset, EMNLP Findings 2023)
- Vaswani et al., *Attention Is All You Need* (2017)
- Raschka, *Practical Tips for Finetuning LLMs*
- Meta AI, *Llama 3.2*
- Hugging Face `transformers` and `peft`

See the paper for the full reference list.

## Disclaimer

This project is a non-commercial academic study. *Harry Potter* and all related characters are the intellectual property of J.K. Rowling and their respective rights holders. The synthetic dialogues are used solely for research on persona-consistent fine-tuning.

## License

No license has been specified yet. Until one is added, all rights are reserved by the author.
