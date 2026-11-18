import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from trl.experimental.ppo import PPOConfig, PPOTrainer

from .common import DEFAULT_MODEL, precision_kwargs, read_jsonl, seed_everything


def main():
    parser = argparse.ArgumentParser(description="ppo-based rl with a learned reward model")
    parser.add_argument("--policy", default=DEFAULT_MODEL)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--data", default="data/sft.jsonl")
    parser.add_argument("--output", default="runs/ppo")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--response-length", type=int, default=64)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [row["prompt"] for row in read_jsonl(args.data)]
    encoded = [
        tokenizer(prompt, truncation=True, max_length=args.max_prompt_length)["input_ids"]
        for prompt in prompts
    ]
    dataset = Dataset.from_dict({"input_ids": encoded})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    policy = AutoModelForCausalLM.from_pretrained(args.base_model)
    if Path(args.policy).is_dir() and (Path(args.policy) / "adapter_config.json").exists():
        policy = PeftModel.from_pretrained(policy, args.policy, is_trainable=True)
    elif args.policy != args.base_model:
        policy = AutoModelForCausalLM.from_pretrained(args.policy)
    reference = AutoModelForCausalLM.from_pretrained(args.base_model)
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, num_labels=1)
    value_model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)
    reward_model.config.pad_token_id = tokenizer.pad_token_id
    value_model.config.pad_token_id = tokenizer.pad_token_id

    config = PPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        total_episodes=args.episodes,
        num_ppo_epochs=args.ppo_epochs,
        response_length=args.response_length,
        local_rollout_forward_batch_size=args.batch_size,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        **precision_kwargs(),
    )
    trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=policy,
        ref_model=reference,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
