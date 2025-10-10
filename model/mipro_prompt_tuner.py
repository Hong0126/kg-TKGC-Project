
from __future__ import annotations
import argparse, random, re
from pathlib import Path
import dspy
import pickle

def build_lm(model_path: str, bs: int):
    #lm = dspy.LM(model = "openai/qwen7b-ft",api_base="http://0.0.0.0:7501/v1", api_key="123", model_type="chat", batch_size=4,max_tokens = 10000, temperature = 0.0,)
    lm = dspy.LM(f"azure/gpt-4.1-mini",
        api_base="https://smartsearch-models-sweden.openai.azure.com/",
        api_key="",
        api_version="2025-04-01-preview",
        temperature=0.0,
    )
    return lm


# ---------- 2. PromptBuilder ----------
from prompt.prompt_builder import PromptBuilder, Triple

# ---------- 3. gIOU helpers ----------
import re
_DIG4 = re.compile(r"\d{4}")
_SPAN = re.compile(r"(\d{4})[^0-9]{0,10}(\d{4})")

def safe_text(obj):
    """取纯文本."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return obj.get("content") or obj.get("text") or str(obj)
    return str(obj)

def _years(text: str):
    txt = safe_text(text)
    if m := _SPAN.search(txt): return int(m[1]), int(m[2])
    if m := _DIG4.search(txt): y = int(m[0]); return y, y
    return None, None

def _norm(ps,pe,gs,ge):
    if ps>pe: ps,pe=pe,ps
    if gs>ge: gs,ge=ge,gs
    inter = max(0, min(pe,ge)-max(ps,gs)+1)
    len_p = pe-ps+1
    len_g = ge-gs+1
    union = len_p + len_g - inter
    hull  = max(pe,ge) - min(ps,gs) + 1
    return inter, union, hull
    
def giou(pred, gold):
    (ps,pe),(gs,ge) = pred, gold
    if None in (ps,pe,gs,ge): return 0.0
    inter, union, hull = _norm(ps,pe,gs,ge)
    if hull<=0: return 0.0
    iou = inter/union if union>0 else 0.0
    return iou - (hull - union) / hull
    

# ---------- 4. 构造训练集 ----------
def load_examples(root: Path, k: int, seed=42):
    pb   = PromptBuilder(root, context_mode="both")
    lines = (root/"valid.txt").read_text().splitlines()
    random.Random(seed).shuffle(lines)
    exs  = []
    for l in lines[:k]:
        s,r,o,t1,t2 = l.split("\t")
        tri: Triple = (int(s),int(r),int(o),t1,t2)
        prompt = pb.build_prompt([], tri)
        gold   = f"{t1[:4]}-{t2[:4]}"
        #关键：必须 with_inputs("question")
        exs.append(
            dspy.Example(question=prompt, answer=gold)
                .with_inputs("question")
        )
    return exs

# ---------- 5. 主流程 ----------
def run(cfg):
    dspy.configure(lm=build_lm(cfg.base_model, cfg.bs))

    # trainset = load_examples(Path(cfg.dataset), cfg.k)

    from dspy.teleprompt import MIPROv2
    tele = MIPROv2(
        metric=lambda y_pred, ex, *_: giou(_years(y_pred), _years(ex.answer)),
        auto="heavy",
        max_bootstrapped_demos = 4,
        max_labeled_demos = 4,
        verbose=True,
    )

    print(">>> Optimizing prompt with MIPROv2 ...")
    examples = pickle.load(open("data/yago11k/examples_train_0hop.pkl", "rb"))
    best = tele.compile(
        dspy.ChainOfThought("question -> answer"),
        trainset=examples,
        requires_permission_to_run=False,
    )

    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best.save(out, save_program=True)
    print("✓ Saved best prompt →", out)

# ---------- 6. CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser("MiProV2 prompt tuner (DSPy 2.6.27)")
    p.add_argument("--dataset",    required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument("--api_base",   default=None)
    p.add_argument("--k",      type=int, default=500)
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--bs",     type=int, default=4)
    p.add_argument("--out",    default="best_prompt.txt")
    cfg = p.parse_args()
    run(cfg)

