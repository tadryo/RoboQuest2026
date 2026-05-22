"""
逃げロボットの強化学習訓練スクリプト（SB3 PPO）

使用例:
  python scripts/train_flee.py
  python scripts/train_flee.py --timesteps 500000 --save-dir models/my_flee
  python scripts/train_flee.py --distance 2.0 --control 0.01
"""
import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from roboquest.envs.go2_tag_env import Go2TagEnv
from roboquest.utils.reward_utils import RewardConfig


def make_env(reward_cfg: RewardConfig, rank: int = 0):
    def _init():
        env = Go2TagEnv(reward_config=reward_cfg)
        env = Monitor(env)
        return env
    return _init


def train(args):
    cfg = RewardConfig(
        distance_weight=args.distance,
        survival_weight=args.survival,
        control_weight=args.control,
        forward_weight=args.forward,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    n_envs = args.num_envs
    env = SubprocVecEnv([make_env(cfg, i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    eval_env = SubprocVecEnv([make_env(cfg)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 256, 128]},
        verbose=1,
        tensorboard_log=os.path.join(args.save_dir, "tensorboard"),
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path=os.path.join(args.save_dir, "checkpoints"),
            name_prefix="flee",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.save_dir, "best"),
            log_path=os.path.join(args.save_dir, "eval_logs"),
            eval_freq=max(50_000 // n_envs, 1),
            n_eval_episodes=5,
            verbose=0,
        ),
    ]

    print(f"\n=== 逃げロボットの学習開始 ===")
    print(f"パラメータ: distance={cfg.distance_weight}, survival={cfg.survival_weight}, "
          f"control={cfg.control_weight}, forward={cfg.forward_weight}")
    print(f"学習ステップ数: {args.timesteps:,}")
    print(f"並列環境数: {n_envs}")
    print(f"保存先: {args.save_dir}\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n学習を中断しました。現在のモデルを保存します...")

    # 最終モデルを保存
    model_path = os.path.join(args.save_dir, "flee_final")
    model.save(model_path)
    env.save(os.path.join(args.save_dir, "vec_normalize.pkl"))
    print(f"\n✅ 学習完了: {model_path}.zip")
    return model_path + ".zip"


def main():
    parser = argparse.ArgumentParser(description="逃げロボット学習スクリプト")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", default="models/flee_robot")
    # 報酬パラメータ
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--survival", type=float, default=0.1)
    parser.add_argument("--control", type=float, default=0.05)
    parser.add_argument("--forward", type=float, default=0.5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
