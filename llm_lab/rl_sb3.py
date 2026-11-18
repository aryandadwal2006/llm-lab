import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    parser = argparse.ArgumentParser(description="train and evaluate a gymnasium agent with stable-baselines3")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/cartpole_sb3")
    args = parser.parse_args()

    env = gym.make(args.env)
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} std_reward={std_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
