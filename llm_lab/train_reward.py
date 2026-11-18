import argparse

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification

from trl import RewardConfig, RewardTrainer

from .common import DEFAULT_MODEL, load_tokenizer, precision_kwargs, seed_everything


def main():
    parser = argparse.ArgumentParser(description="train a scalar reward model from preference pairs")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default="data/preferences.jsonl")
    parser.add_argument("--output", default="runs/reward")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    dataset = load_dataset("json", data_files=args.data, split="train")
    if not {"prompt", "chosen", "rejected"}.issubset(dataset.column_names):
        raise ValueError("preference data must contain prompt, chosen, and rejected fields")
    tokenizer = load_tokenizer(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id

    config = RewardConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        **precision_kwargs(),
    )
    trainer = RewardTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
