import argparse

from datasets import load_dataset

from trl import DPOConfig, DPOTrainer

from .common import DEFAULT_MODEL, load_tokenizer, lora_config, precision_kwargs, seed_everything


def main():
    parser = argparse.ArgumentParser(description="direct preference optimization with trl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default="data/preferences.jsonl")
    parser.add_argument("--output", default="runs/dpo")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    dataset = load_dataset("json", data_files=args.data, split="train")
    if not {"prompt", "chosen", "rejected"}.issubset(dataset.column_names):
        raise ValueError("preference data must contain prompt, chosen, and rejected fields")
    tokenizer = load_tokenizer(args.model)
    config = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=min(256, args.max_length // 2),
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        **precision_kwargs(),
    )
    trainer = DPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(),
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
