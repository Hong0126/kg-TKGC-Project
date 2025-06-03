from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from config import ExperimentConfig

def load_llm(cfg: ExperimentConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if cfg.adaptation == "zero_shot":
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, device_map="auto")
        model.eval()
        return model, tok

    # 4-bit quant for memory
    bnb = BitsAndBytesConfig(load_in_4bit=True, llm_int8_threshold=6.0)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, device_map="auto", quantization_config=bnb
    )

    if cfg.adaptation == "lora":
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_cfg)

    model.train()
    return model, tok
