import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import DEFAULT_MODEL, read_jsonl, seed_everything, write_jsonl


def generate(model, tokenizer, prompt: str, temperature: float, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="generate preference pairs using a local model judge")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default="data/sft.jsonl")
    parser.add_argument("--output", default="data/rlaif_preferences.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    preferences = []
    for row in read_jsonl(args.data):
        prompt = row["prompt"]
        candidate_a = generate(model, tokenizer, prompt, 0.4, args.max_new_tokens)
        candidate_b = generate(model, tokenizer, prompt, 0.9, args.max_new_tokens)
        judge_prompt = (
            "Compare two assistant responses to the same user request. "
            "Choose the response that is more correct, relevant, complete, and direct. "
            "Do not reward length alone. Return exactly A or B.\n\n"
            f"User request: {prompt}\n\nCandidate A:\n{candidate_a}\n\nCandidate B:\n{candidate_b}\n\nChoice:"
        )
        judgment = generate(model, tokenizer, judge_prompt, 0.0, 8)
        match = re.search(r"\b([AB])\b", judgment.upper())
        if match:
            chosen, rejected = (candidate_a, candidate_b) if match.group(1) == "A" else (candidate_b, candidate_a)
        else:
            chosen, rejected = (candidate_a, candidate_b) if len(candidate_a) <= len(candidate_b) else (candidate_b, candidate_a)
        if chosen and rejected and chosen != rejected:
            preferences.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    if not preferences:
        raise RuntimeError("the judge produced no distinct response pairs")
    write_jsonl(args.output, preferences)
    print(f"wrote {len(preferences)} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
