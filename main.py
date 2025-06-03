import random, torch
from config import ExperimentConfig
from data.kg_loader import load_kg
from graph.subgraph_utils import (
    extract_hop_subgraph, temporal_filter, attention_prune
)
from data.prompt import build_prompt
from models.llm_loader import load_llm
from training.trainer import TKGCTrainer
from torch.utils.data import Dataset

class TKGCTextDataset(Dataset):
    def __init__(self, triples, kg, cfg, tokenizer):
        self.data = []
        for s,r,o,ts,te in triples:              # replace with real loader
            sg = extract_hop_subgraph(kg, s, o, cfg.hop_radius)
            if cfg.use_temporal_filter:
                sg = temporal_filter(sg, (ts,te))
            if cfg.use_attention_pruning:
                sg = attention_prune(sg, cfg.max_subgraph_edges)
            prompt = build_prompt(s,r,o,(ts,te), sg, cfg.prompt_style)
            label = f" [{ts}, {te}]"
            self.data.append(prompt + label)
        self.tok = tokenizer
        self.max_len = 2048
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        enc = self.tok(
            self.data[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        enc = {k:v.squeeze(0) for k,v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()
        return enc

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    kg = load_kg(cfg.data_dir / "wikidata12k.tsv")
    sample_triples = [("Amsterdam","capital_of","Netherlands",1815,1983)]

    model, tok = load_llm(cfg)
    train_ds = TKGCTextDataset(sample_triples, kg, cfg, tok)
    trainer  = TKGCTrainer(model, tok, cfg, train_ds)
    trainer.train()
    trainer.evaluate(train_ds)
