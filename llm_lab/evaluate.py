import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .common import DEFAULT_MODEL, load_causal_model, load_tokenizer, read_jsonl, seed_everything


def normalize(text: str) -> str:
    return re.sub(r"\\s+", " ", text.strip().lower())


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize(prediction).split()
    expected = normalize(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = sum((Counter(predicted) & Counter(expected)).values())
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def valid_json_object(text: str) -> bool:
    cleaned = text.strip()
    if cleaned.startswith(chr(96) * 3):
        lines = cleaned.splitlines()
        cleaned = "\\n".join(lines[1:-1]).strip()
    try:
        return isinstance(json.loads(cleaned), dict)
    except json.JSONDecodeError:
        return False


def run_model(model_id: str, rows: list[dict], max_new_tokens: int, seed: int):
    seed_everything(seed)
    tokenizer = load_tokenizer(model_id)
    model = load_causal_model(model_id)
    model.eval()
    device = next(model.parameters()).device
    results = []
    for row in rows:
        inputs = tokenizer(row["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        output_tokens = max(0, output.shape[1] - inputs["input_ids"].shape[1])
        results.append({
            "task_type": row.get("task_type", "general"),
            "prediction": text,
            "exact_match": float(normalize(text) == normalize(row.get("completion", ""))),
            "token_f1": token_f1(text, row.get("completion", "")),
            "json_valid": valid_json_object(text) if row.get("task_type") == "tool_use" else None,
            "latency_seconds": elapsed,
            "tokens_per_second": output_tokens / max(elapsed, 1e-9),
        })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    grouped = defaultdict(list)
    for result in results:
        grouped[result["task_type"]].append(result)
    summary = {}
    for task, values in grouped.items():
        summary[task] = {
            "examples": len(values),
            "exact_match": sum(x["exact_match"] for x in values) / len(values),
            "token_f1": sum(x["token_f1"] for x in values) / len(values),
            "mean_latency_seconds": sum(x["latency_seconds"] for x in values) / len(values),
            "mean_tokens_per_second": sum(x["tokens_per_second"] for x in values) / len(values),
        }
        tool_values = [x["json_valid"] for x in values if x["json_valid"] is not None]
        if tool_values:
            summary[task]["valid_tool_json_rate"] = sum(tool_values) / len(tool_values)
    return {"model": model_id, "summary": summary, "examples": results}


def main():
    parser = argparse.ArgumentParser(description="compare an adapted model with a fixed baseline")
    parser.add_argument("--baseline", default=DEFAULT_MODEL)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--data", default="data/eval.jsonl")
    parser.add_argument("--output", default="runs/eval.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_jsonl(args.data)
    baseline = run_model(args.baseline, rows, args.max_new_tokens, args.seed)
    candidate = run_model(args.candidate, rows, args.max_new_tokens, args.seed)
    report = {"baseline": baseline, "candidate": candidate}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "baseline": baseline["summary"],
        "candidate": candidate["summary"],
        "saved_to": str(target),
    }, indent=2))


if __name__ == "__main__":
    main()
