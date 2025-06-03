from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ExperimentConfig:
    # --- E1 ---
    hop_radius: int = 3                 # 1 / 2 / 3
    # --- E2 ---
    use_temporal_filter: bool = True
    use_attention_pruning: bool = True
    max_subgraph_edges: int = 128
    # --- E3 ---
    prompt_style: str = "nl"            # "triple" | "nl"
    # --- E4 ---
    adaptation: str = "lora"            # "zero_shot" | "lora" | "full_ft"
    model_name: str = "Qwen/Qwen1.5-7B-Chat"
    # LoRA hyper-params
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # training
    lr: float = 2e-5
    batch_size: int = 2
    num_epochs: int = 3
    grad_accum: int = 4
    warmup_ratio: float = 0.05
    seed: int = 42
    # paths
    data_dir: Path = Path("data/raw")
    cache_dir: Path = Path("data/cache")
    out_dir: Path   = Path("outputs")
