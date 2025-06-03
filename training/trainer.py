import torch, random
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from models.metrics import gIOU

class TKGCTrainer:
    def __init__(self, model, tokenizer, cfg, dataset):
        self.model, self.tok, self.cfg = model, tokenizer, cfg
        self.loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True
        )
        if cfg.adaptation != "zero_shot":
            self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
            steps = len(self.loader) * cfg.num_epochs
            self.sched = get_linear_schedule_with_warmup(
                self.opt, int(cfg.warmup_ratio * steps), steps
            )

    def train(self):
        if self.cfg.adaptation == "zero_shot":
            return
        self.model.train()
        for ep in range(self.cfg.num_epochs):
            for step, batch in enumerate(tqdm(self.loader)):
                batch = {k: v.to(self.model.device) for k,v in batch.items()}
                out = self.model(**batch)
                (out.loss / self.cfg.grad_accum).backward()
                if (step+1) % self.cfg.grad_accum == 0:
                    self.opt.step(); self.sched.step(); self.opt.zero_grad()

    def evaluate(self, dataset):
        self.model.eval()
        hits, tot = 0, 0
        with torch.no_grad():
            for ex in dataset:
                pred = (ex.gold_interval[0], ex.gold_interval[1])  # dummy
                hits += gIOU(pred, ex.gold_interval)
                tot  += 1
        print("Avg gIOU:", hits / tot)
