import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer

from .common import DEFAULT_MODEL, lora_config, precision_kwargs, read_jsonl, seed_everything, write_jsonl


def sequence_distill(args):
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher.to(device).eval()
    generated = []
    for row in read_jsonl(args.data):
        prompt = row["prompt"]
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        with torch.inference_mode():
            output = teacher.generate(
                **encoded,
                max_new_tokens=args.response_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        generated.append({"prompt": prompt, "completion": completion})
    del teacher
    Path(args.output).mkdir(parents=True, exist_ok=True)
    generated_path = str(Path(args.output) / "teacher_sequences.jsonl")
    write_jsonl(generated_path, generated)

    student = AutoModelForCausalLM.from_pretrained(args.student)
    student.config.use_cache = False
    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        completion_only_loss=True,
        gradient_checkpointing=True,
        report_to="none",
        **precision_kwargs(),
    )
    trainer = SFTTrainer(
        model=student,
        args=config,
        train_dataset=Dataset.from_list(generated),
        processing_class=tokenizer,
        peft_config=lora_config(),
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


def logits_distill(args):
    rows = read_jsonl(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.get_vocab() != student_tokenizer.get_vocab():
        raise ValueError("logit distillation requires identical teacher and student token-id mappings")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher).to(device).eval()
    student = AutoModelForCausalLM.from_pretrained(args.student).to(device)
    student = get_peft_model(student, lora_config())
    teacher.requires_grad_(False)
    student.config.use_cache = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    temperature = args.temperature
    student.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        used = 0
        for row in rows:
            prompt = row["prompt"]
            text = prompt + row["completion"]
            full_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)["input_ids"].to(device)
            prompt_length = len(tokenizer(prompt, truncation=True, max_length=args.max_length)["input_ids"])
            if full_ids.shape[1] < 3 or prompt_length >= full_ids.shape[1]:
                continue
            attention = torch.ones_like(full_ids)
            with torch.no_grad():
                teacher_logits = teacher(input_ids=full_ids, attention_mask=attention).logits[:, :-1]
            student_logits = student(input_ids=full_ids, attention_mask=attention).logits[:, :-1]
            mask = (torch.arange(student_logits.shape[1], device=device) >= max(prompt_length - 1, 0)).float()
            token_kl = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="none",
            ).sum(dim=-1).squeeze(0)
            loss = (token_kl * mask).sum() / mask.sum().clamp_min(1.0)
            loss = loss * temperature * temperature
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            used += 1
        if used == 0:
            raise ValueError("no valid prompt/completion pairs remain after tokenization")
        print(f"epoch {epoch + 1}: mean_kl={total_loss / used:.6f}")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    student.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


def main():
    parser = argparse.ArgumentParser(description="logit-level and sequence-level knowledge distillation")
    parser.add_argument("--mode", choices=["logits", "sequence"], required=True)
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--student", default=DEFAULT_MODEL)
    parser.add_argument("--data", default="data/sft.jsonl")
    parser.add_argument("--output", default="runs/distillation")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--response-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.mode == "logits":
        logits_distill(args)
    else:
        sequence_distill(args)


if __name__ == "__main__":
    main()
