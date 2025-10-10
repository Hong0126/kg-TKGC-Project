from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct" 
LORA_DIR   = "/root/auto-tmp/ckpts/qwen7b-lora-tkgc" 
OUT_DIR    = "/root/autodl-tmp/qwen7b-ft-merged"

def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading base …")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype="auto")

    print("Merging LoRA …")
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model = model.merge_and_unload()

    print("Saving merged model …")
    model.save_pretrained(out)

    print("Copying tokenizer …")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tok.save_pretrained(out)

    print("Done! merged model dir:", out)

if __name__ == "__main__":
    main()
