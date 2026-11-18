import argparse
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig


def main():
    parser = argparse.ArgumentParser(description="train a gymnasium agent with ray rllib")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/cartpole_rllib")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    config = (
        PPOConfig()
        .environment(args.env)
        .framework("torch")
        .env_runners(num_env_runners=0)
        .training(
            train_batch_size_per_learner=args.batch_size,
            lr=args.learning_rate,
            entropy_coeff=args.entropy_coef,
        )
        .debugging(seed=args.seed)
    )
    algorithm = config.build()
    try:
        for iteration in range(args.iterations):
            result = algorithm.train()
            reward = result.get("env_runners", {}).get("episode_return_mean")
            if reward is None:
                reward = result.get("episode_reward_mean", float("nan"))
            print(f"iteration={iteration + 1} mean_reward={reward}")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = algorithm.save(args.output)
        print(f"saved checkpoint: {checkpoint}")
    finally:
        algorithm.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
