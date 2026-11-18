import argparse

from vllm import LLM, SamplingParams

from .common import DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description="local inference with vllm")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()

    engine = LLM(model=args.model, max_model_len=args.max_model_len, trust_remote_code=False)
    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )
    result = engine.generate([args.prompt], params)
    print(result[0].outputs[0].text.strip())


if __name__ == "__main__":
    main()
