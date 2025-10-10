# -*- coding: utf-8 -*-
"""finetune_trainer.py
```bash
python -m model.finetune_trainer \
  --model_name_or_path Qwen/Qwen-7B \
  --train_file data/wikidata12k/sft_train.jsonl \
  --valid_file data/wikidata12k/sft_valid.jsonl \
  --output_dir ckpts/qwen7b-lora-tkgc \
  --epochs 3 --per_device_batch 4 --lora_r 16
```
```
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    model_name_or_path: str
    train_file: str
    valid_file: str | None = None
    output_dir: str = "./lora_out"
    epochs: int = 3
    per_device_batch: int = 4
    lr: float = 5e-5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_len: int = 512
    gradient_accumulation_steps: int = 1

    def to_args(self):
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_batch,
            per_device_eval_batch_size=self.per_device_batch,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.epochs,
            learning_rate=self.lr,
            logging_steps=20,
            eval_strategy="steps",      # ← 改名
            save_strategy="steps",      # ← 改名
            eval_steps=200,
            save_steps=1000,
            fp16=True,
            report_to="none",
        )

def jsonl_stream(path: str) -> Iterator[Dict[str, str]]:
    """Yield dicts with single 'text' field for SFTTrainer."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                # 拼接 prompt + target，或按需拆分
                yield {"text": obj["prompt"] + obj["target"]}
# ---------------------------------------------------------------------------
class JSONLIterable:
    """Streaming IterableDataset from jsonl with {prompt,target}."""

    def __init__(self, path: str | Path):
        self.path = str(path)

    def __iter__(self) -> Iterator[Dict]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


# ---------------------------------------------------------------------------
class FinetuneTrainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self._prepare_model_tokenizer()
        self._prepare_data()
        self._build_trainer()

    # -----------------------
    def _prepare_model_tokenizer(self):
        logger.info("Loading base model %s (4‑bit QLoRA) …", self.cfg.model_name_or_path)
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name_or_path,
            quantization_config=bnb_cfg,
            trust_remote_code=True,
            device_map="auto",
        )
        # LoRA
        logger.info("Adding LoRA adapters r=%d …", self.cfg.lora_r)
        lora_cfg = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

    # -----------------------

                    
    def _prepare_data(self):
        logger.info("Loading dataset …")
        train_ds = Dataset.from_generator(jsonl_stream, gen_kwargs={"path": self.cfg.train_file})
        self.train_ds = train_ds
        self.eval_ds = None
        if self.cfg.valid_file:
            self.eval_ds = Dataset.from_generator(jsonl_stream, gen_kwargs={"path": self.cfg.valid_file})

    # -----------------------
    def _build_trainer(self):
        logger.info("Building SFTTrainer …")
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            max_seq_length=self.cfg.max_seq_len,
            packing=True,
            args=self.cfg.to_args(),
            dataset_text_field="text",
        )

    # -----------------------
    def train(self):
        logger.info("Start training …")
        self.trainer.train()
        logger.info("Saving adapter to %s", self.cfg.output_dir)
        self.trainer.model.save_pretrained(self.cfg.output_dir)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("LoRA fine‑tune TKGC")
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--valid_file")
    ap.add_argument("--output_dir", default="./lora_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--per_device_batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1,
                help="gradient accumulation steps")

    args = ap.parse_args()

    cfg = TrainConfig(**vars(args))
    trainer = FinetuneTrainer(cfg)
    trainer.train()
