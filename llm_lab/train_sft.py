import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from trl import SFTConfig, SFTTrainer

from .common import DEFAULT_MODEL, load_tokenizer, lora_config, precision_kwargs, seed_everything


def main():
    parser = argparse.ArgumentParser(description="instruction tuning with trl and peft")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default="data/sft.jsonl")
    parser.add_argument("--output", default="runs/sft")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    dataset = load_dataset("json", data_files=args.data, split="train")
    if not {"prompt", "completion"}.issubset(dataset.column_names):
        raise ValueError("sft data must contain prompt and completion fields")

    tokenizer = load_tokenizer(args.model)
    model_kwargs = {}
    if args.qlora:
        if not torch.cuda.is_available():
            raise RuntimeError("qlora requires a supported cuda gpu")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs.update({"quantization_config": quantization_config, "device_map": "auto"})
    else:
        model_kwargs.update({"torch_dtype": torch.float32 if not torch.cuda.is_available() else "auto"})

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        completion_only_loss=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        **precision_kwargs(),
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(),
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved adapter to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
