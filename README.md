# LLM Adaptation, Distillation & RL Alignment Lab

A small, local-first implementation of instruction tuning, knowledge distillation, preference learning, PPO-based RLHF, RLAIF, and inference benchmarking. It also includes PPO baselines and fixed-seed ablations for Gymnasium using Stable-Baselines3 and Ray RLlib.

All examples use open-weight models and local files. No API keys, hosted inference services, experiment trackers, or paid endpoints are required. The first run downloads the selected model from Hugging Face.

## Requirements

- Python 3.10–3.12
- PyTorch 2.6 or newer
- A GPU is strongly recommended for language-model training. The default model is `Qwen/Qwen2.5-0.5B-Instruct`.
- vLLM requires a compatible Linux/CUDA environment. Install it separately from the training environment if its PyTorch/CUDA requirements conflict.

Install the core stack:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

For optional components:

```bash
pip install -r requirements-qlora.txt
pip install -r requirements-rl.txt
```

Install vLLM in a compatible Linux/CUDA environment:

```bash
pip install -r requirements-vllm.txt
```

PyTorch wheels differ by operating system and CUDA version. When needed, install the matching PyTorch build from the official PyTorch selector before installing the remaining requirements.

## Data formats

The repository includes tiny JSONL examples to make the commands runnable. They are smoke-test data, not meaningful training corpora.

Instruction tuning and distillation use:

```json
{"prompt":"Explain gradient descent in one sentence.","completion":"Gradient descent repeatedly adjusts parameters in the direction that most reduces the loss."}
```

Preference training uses:

```json
{"prompt":"What is 2 + 2?","chosen":"4.","rejected":"5."}
```

Use your own local JSONL files with the same fields for real experiments.

## Workflows

### 1. SFT with LoRA

```bash
python -m llm_lab.train_sft --data data/sft.jsonl --output runs/sft
```

The trainer uses TRL's `SFTTrainer` and PEFT LoRA. Change the model with `--model`, or use `--qlora` for 4-bit QLoRA:

```bash
python -m llm_lab.train_sft --data data/sft.jsonl --output runs/qlora --qlora
```

QLoRA requires a supported NVIDIA CUDA setup and `bitsandbytes`.

### 2. Logit- and sequence-level distillation

Logit-level distillation minimizes the temperature-scaled KL divergence between teacher and student token distributions:

```bash
python -m llm_lab.distill --mode logits --teacher Qwen/Qwen2.5-1.5B-Instruct --student Qwen/Qwen2.5-0.5B-Instruct --data data/sft.jsonl --output runs/distill-logits
```

Sequence-level distillation first samples teacher completions, then trains the student on those completions with TRL SFT:

```bash
python -m llm_lab.distill --mode sequence --teacher Qwen/Qwen2.5-1.5B-Instruct --student Qwen/Qwen2.5-0.5B-Instruct --data data/sft.jsonl --output runs/distill-sequence
```

Logit-level distillation requires identical teacher and student token-id mappings. Sequence-level distillation supports different tokenizers because the teacher outputs are converted to text and tokenized again by the student tokenizer.

### 3. Reward modelling

```bash
python -m llm_lab.train_reward --data data/preferences.jsonl --output runs/reward
```

This trains a scalar reward model with TRL's `RewardTrainer`.

### 4. DPO

```bash
python -m llm_lab.train_dpo --data data/preferences.jsonl --output runs/dpo
```

DPO directly trains on prompt/chosen/rejected triples. It does not require an API or a separately trained reward model.

### 5. PPO-based RLHF

Train a reward model first, then use it with a policy and a value model:

```bash
python -m llm_lab.train_ppo --data data/sft.jsonl --policy runs/sft --base-model Qwen/Qwen2.5-0.5B-Instruct --reward-model runs/reward --output runs/ppo
```

The implementation uses TRL's experimental PPO trainer. PPO is computationally heavier than SFT or DPO; start with a small number of episodes and a compact model. The value model is initialized from the base model and trained during PPO.

### 6. RLAIF-style preference generation

Generate two local model responses per prompt, ask a local language model to judge them with a fixed rubric, and save the selected pair:

```bash
python -m llm_lab.rlaif --model Qwen/Qwen2.5-0.5B-Instruct --data data/sft.jsonl --output data/rlaif_preferences.jsonl
python -m llm_lab.train_dpo --data data/rlaif_preferences.jsonl --output runs/rlaif-dpo
```

This is a small self-judging RLAIF demonstration, not a substitute for independently validated human preference labels. The same model is used to generate and judge candidates, so judge bias is possible.

### 7. Controlled evaluation

Compare a base model against an adapted model using the same prompts, greedy decoding, and fixed seed:

```bash
python -m llm_lab.evaluate --baseline Qwen/Qwen2.5-0.5B-Instruct --candidate runs/sft --data data/eval.jsonl --output runs/eval.json
```

The report includes exact-match and token F1 for reference completions, JSON validity for tool-call examples, generated-token throughput, and mean generation latency. These are lightweight diagnostics, not a comprehensive reasoning or tool-use benchmark. Use a held-out, task-specific dataset for credible comparisons.

### 8. vLLM inference

For a PEFT adapter created by the SFT command:

```bash
python -m llm_lab.infer_vllm --model Qwen/Qwen2.5-0.5B-Instruct --lora runs/sft --prompt "Explain LoRA in two sentences."
```

For a regular full model, omit `--lora`.

### 9. PPO on Gymnasium

Stable-Baselines3:

```bash
python -m llm_lab.rl_sb3 --env CartPole-v1 --timesteps 20000 --output runs/cartpole_sb3
```

Ray RLlib:

```bash
python -m llm_lab.rl_rllib --env CartPole-v1 --iterations 10 --output runs/cartpole_rllib
```

Fixed-seed PPO hyperparameter ablations with Stable-Baselines3:

```bash
python -m llm_lab.rl_ablation --env CartPole-v1 --seeds 42,43,44 --learning-rates 0.0003,0.0001 --entropy-coefs 0.0,0.01 --output runs/ppo_ablation.json
```

The ablation runner records each seed/configuration and mean/std evaluation reward in JSON. These scripts provide repeatable PPO baselines and an ablation harness; they do not claim exact reproduction of a particular paper's reported scores. Reproducing a paper requires matching its environment version, preprocessing, network, hyperparameters, training budget, and evaluation protocol.

## CLI options

Every training script exposes `--help`. Common controls include `--model`, `--data`, `--output`, `--epochs`, `--batch-size`, `--learning-rate`, and `--max-length` where applicable. Training outputs are written to `runs/` by default.

## Reproducibility and limitations

- Training scripts set explicit random seeds and use greedy decoding for controlled language-model evaluation.
- Keep the baseline and candidate evaluation prompts, decoding parameters, batch sizes, and hardware identical.
- The included datasets are deliberately tiny examples; replace them before making quality claims.
- Reward models and self-judging RLAIF can encode their own biases. Inspect generated preferences before training.
- QLoRA and vLLM depend on compatible CUDA, driver, and PyTorch builds. Their requirements are kept separate from the core environment.
- Ray and Stable-Baselines3 are optional so LLM training does not pull in the RL environment stack unnecessarily.
