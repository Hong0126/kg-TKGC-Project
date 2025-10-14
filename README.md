# kg-TKGC-Project

This repository contains a two-stage pipeline for temporal knowledge graph completion (TKGC):

1. **Temporal Graph Neural Network (GNN) training** – an inductive graph attention network (IGAT) that predicts time intervals for missing facts in a temporal knowledge graph **and serves as the structural encoder that later guides LLM evidence selection**.
2. **Large Language Model (LLM) alignment** – a supervised fine-tuning (SFT) stack that distills graph-derived supervision into an instruction-following model for natural-language reasoning over temporal facts.

The project pairs structured graph learning with instruction tuning so that downstream systems can answer temporal questions with calibrated time spans.

## Repository layout

```
.
├── data/              # Benchmark temporal KGs and inductive splits
├── gnn/               # IGAT encoder used for entity retrieval & temporal supervision
├── model/             # LLM fine-tuning, prompt tuning, and LoRA utilities
├── prompt/            # Prompt builders shared by scripts and trainers
├── scripts/           # Experiment helpers for data prep and evaluation
└── requirements.txt   # Core Python dependencies
```

### `gnn/`
* `main.py` runs end-to-end IGAT training with hybrid ranking + gIoU optimization, including temporal normalization and evaluation helpers. The resulting checkpoints expose attention weights that Stage B can consume as structural priors.
* `igat_batcher.py` and `igat_predictor.py` define the inductive temporal batching logic and model architecture consumed by the trainer. Their graph attention heads double as the entity encoder for subgraph selection.

### `model/`
* `finetune_trainer.py` provides a configurable QLoRA + TRL SFT pipeline for causal LLMs, streaming JSONL training data produced by the scripts below.
* `merge_lora_and_save.py` merges trained LoRA adapters back into the base checkpoint for standalone inference.
* `mipro_prompt_tuner.py` leverages DSPy MiPROv2 to optimize few-shot prompts with a gIoU-based objective, enabling prompt-only baselines or hybrid systems.

### `scripts/`
* `build_data.py` converts temporal KG triples into structured SFT JSONL splits using the shared prompt builder, with optional multi-hop context sampling.
* `score_saved_preds.py` provides offline evaluation utilities for stored predictions.
* `zero_shot_eval.py` evaluates zero-shot models against TKGC test sets using the same metrics as training.

## Environment setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The project relies on PyTorch, Hugging Face Transformers, PEFT/QLoRA tooling, TRL, and utilities such as NetworkX and datasets.

GPU acceleration is recommended for both IGAT training and LLM fine-tuning.

## Data preparation

Temporal benchmark datasets (e.g., WIKIDATA12k, YAGO11k) are stored under `data/` with inductive splits already organized. Use `scripts/build_data.py` to transform KG triples into SFT-ready prompts and targets:

```bash
python scripts/build_data.py data/WIKIDATA12k \
  --output_dir data/WIKIDATA12k/sft \
  --add_multihop
```
The script samples prompts, formats targets as year spans, and writes `sft_train.jsonl` / `sft_valid.jsonl` splits for LLM training.

## Stage A & Stage B: Graph evidence preparation

Before the LLM sees any text, the pipeline performs two graph-centric stages that turn a raw temporal KG into a compact, query-specific evidence list.

### Stage A — Multi-hop subgraph extraction with temporal filtering

Implemented across `prompt/selector.py` and `prompt/prompt_builder.py`, Stage A grows a leakage-safe neighborhood around the subject and object anchors (or their text-matched proxies for inductive queries). It:

1. seeds anchors from the query (or nearest neighbors retrieved via the description encoder when anchors are unseen in training data),
2. infers a coarse year window by scraping descriptions and neighbor facts,
3. runs a BFS-style expansion up to two hops with per-hop degree caps, and
4. keeps only edges whose intervals overlap the window or sit within a small gap threshold.

Each retained fact caches overlap and gap metrics so later stages can reason about temporal proximity, and entity descriptions are prepared for downstream serialization.

### Stage B — Dynamic context pruning

With the filtered subgraph in hand, Stage B (`prompt/selector.py`) ranks edges by a convex combination of temporal affinity, structural salience (including IGAT attention priors), and textual relevance. It enforces hop caps, maintains coverage of both query endpoints, and respects either edge-count or token budgets when assembling the evidence list. The final ranked subset feeds the prompt builder which converts it into natural-language evidence for the LLM.

Running these stages manually is optional—the utilities are invoked automatically when building prompts (see below)—but they can be exercised directly, for example:

```bash
python -m prompt.selector --kg_root data/WIKIDATA12k --subject 123 --object 456 --budget 12
```

Consult the module docstrings for CLI flags that toggle hop counts, budgets, or custom gain functions.

## Training the IGAT temporal GNN

Run the trainer directly to learn temporal representations on the inductive split and to produce the attention priors consumed during Stage B:

```bash
python -m gnn.main
```

The script loads train/validation/test sets, constructs inductive history batches, and optimizes a hybrid ranking + gIoU loss with AMP and gradient scaling. Best checkpoints are stored under `saved_igat_models_final/` by default and can be loaded by the selector for structural scoring.

## Fine-tuning the LLM with QLoRA

After generating SFT data, launch supervised fine-tuning:

```bash
python -m model.finetune_trainer \
  --model_name_or_path Qwen/Qwen-7B \
  --train_file data/WIKIDATA12k/sft/sft_train.jsonl \
  --valid_file data/WIKIDATA12k/sft/sft_valid.jsonl \
  --output_dir ckpts/qwen7b-lora-tkgc \
  --epochs 3 --per_device_batch 4 --lora_r 16
```

* To auto-tune few-shot prompts with MiPROv2 and a gIoU metric, run `model/mipro_prompt_tuner.py` pointing to a dataset directory and target LLM endpoint.
* Use `scripts/zero_shot_eval.py` to score baseline or fine-tuned models against held-out sets with the same generalized IoU metric used during GNN training.
* `scripts/score_saved_preds.py` offers offline scoring for stored predictions, enabling reproducible comparisons across experiments.

## Workflow summary

1. **Prepare data** → Format KG triples into prompts with `scripts/build_data.py`.
2. **Train IGAT encoder** → `python -m gnn.main` for temporal embedding, interval prediction, and attention priors used by Stage B.
3. **Run Stage A/B selection** → Use the prompt builder (auto-triggered during data prep) to extract and prune graph evidence per query.
4. **Fine-tune LLM** → `python -m model.finetune_trainer` using generated SFT splits.
5. **Merge or deploy** → Merge LoRA adapters if needed and evaluate with the provided scripts.