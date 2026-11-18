import json
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid json on line {line_number} of {path}") from exc
    if not rows:
        raise ValueError(f"no examples found in {path}")
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tokenizer(model_id: str):
    adapter_config = Path(model_id) / "adapter_config.json"
    if adapter_config.exists():
        with adapter_config.open("r", encoding="utf-8") as handle:
            model_id = json.load(handle)["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def lora_config(task_type: TaskType = TaskType.CAUSAL_LM) -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def precision_kwargs() -> dict:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return {"bf16": True, "fp16": False}
        return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False}


def load_causal_model(model_id: str, trainable: bool = False, base_model: str | None = None):
    adapter_config = Path(model_id) / "adapter_config.json"
    if adapter_config.exists():
        with adapter_config.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        base_id = base_model or config["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(base_id)
        model = PeftModel.from_pretrained(model, model_id, is_trainable=trainable)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model
