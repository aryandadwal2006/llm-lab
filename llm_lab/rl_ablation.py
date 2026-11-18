import argparse
import json
from itertools import product
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    parser = argparse.ArgumentParser(description="fixed-seed ppo hyperparameter ablations with stable-baselines3")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--learning-rates", default="0.0003,0.0001")
    parser.add_argument("--entropy-coefs", default="0.0,0.01")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output", default="runs/ppo_ablation.json")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",")]
    learning_rates = [float(value) for value in args.learning_rates.split(",")]
    entropy_coefs = [float(value) for value in args.entropy_coefs.split(",")]
    results = []
    for seed, learning_rate, entropy_coef in product(seeds, learning_rates, entropy_coefs):
        train_env = gym.make(args.env)
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            ent_coef=entropy_coef,
            seed=seed,
            verbose=0,
        )
        model.learn(total_timesteps=args.timesteps)
        eval_env = gym.make(args.env)
        eval_env.reset(seed=seed)
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )
        results.append({
            "seed": seed,
            "learning_rate": learning_rate,
            "entropy_coef": entropy_coef,
            "timesteps": args.timesteps,
            "eval_episodes": args.eval_episodes,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
        })
        train_env.close()
        eval_env.close()
        print(f"seed={seed} lr={learning_rate} entropy={entropy_coef} reward={mean_reward:.2f} +/- {std_reward:.2f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
