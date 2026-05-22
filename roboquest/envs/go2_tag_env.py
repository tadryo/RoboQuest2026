"""
鬼ごっこ環境: 逃げる Go2 vs スクリプト鬼

チームの逃げロボットを単独で強化学習で訓練する環境。
「鬼」は簡単なルールベースエージェント（常に逃げロボットへ向かって移動）。

観測空間: 49 + 3 = 52次元（基本観測 + 鬼への相対位置 xy + 距離）
"""
import os
# ヘッドレス環境（Google Colab など）向け: mujoco import 前に EGL を指定
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"
from typing import Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from roboquest.envs.go2_walk_env import Go2WalkEnv, _MODEL_DIR
from roboquest.utils.reward_utils import RewardConfig

# アリーナサイズ（±メートル）
ARENA_SIZE = 5.0

# タグ判定距離
TAG_DISTANCE = 0.5

# 鬼の移動速度 (m/step)
ONI_SPEED = 0.02

# 場面XML（フラット地面）
SCENE_XML = os.path.join(_MODEL_DIR, "scene.xml")


class Go2TagEnv(Go2WalkEnv):
    """鬼ごっこ環境。

    逃げロボット（学習対象）はアリーナ内で鬼から逃げる。
    鬼はシンプルなルールベース（逃げロボットへ直進）。

    追加観測 (3次元):
      - 鬼への相対位置 dx, dy (2)
      - 鬼までの距離 d (1)
    合計観測: 49 + 3 = 52次元
    """

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        oni_speed: float = ONI_SPEED,
    ):
        super().__init__(
            reward_config=reward_config,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
        )
        self.oni_speed = oni_speed

        # 鬼の位置（2D）
        self._oni_pos = np.zeros(2, dtype=np.float64)

        # 観測空間を拡張 (49 → 52)
        obs_dim = self.observation_space.shape[0] + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)

        # 鬼をアリーナの反対側に配置
        rng = self.np_random
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(ARENA_SIZE * 0.5, ARENA_SIZE * 0.8)
        self._oni_pos = np.array([
            dist * np.cos(angle),
            dist * np.sin(angle),
        ], dtype=np.float64)

        full_obs = self._get_full_obs(obs)
        return full_obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = super().step(action)

        # 鬼を逃げロボットへ向かって移動
        self._move_oni()

        # 鬼ごっこ固有の報酬を加算
        flee_reward = self._compute_flee_reward()
        reward += flee_reward

        # タグ判定
        dist = np.linalg.norm(self.robot_xy - self._oni_pos)
        if dist < TAG_DISTANCE:
            terminated = True
            reward -= self.reward_config.tag_penalty

        # 境界ペナルティ
        xy = self.robot_xy
        if np.any(np.abs(xy) > ARENA_SIZE * 0.9):
            reward -= self.reward_config.boundary_penalty

        info["oni_distance"] = float(dist)
        info["is_tagged"] = float(dist < TAG_DISTANCE)

        full_obs = self._get_full_obs(obs)
        return full_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    def _move_oni(self) -> None:
        direction = self.robot_xy - self._oni_pos
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            self._oni_pos += self.oni_speed * direction / norm

    def _get_full_obs(self, base_obs: np.ndarray) -> np.ndarray:
        rel = self._oni_pos - self.robot_xy
        dist = np.linalg.norm(rel).reshape(1)
        return np.concatenate([base_obs, rel, dist]).astype(np.float32)

    def _compute_flee_reward(self) -> float:
        cfg = self.reward_config
        dist = float(np.linalg.norm(self.robot_xy - self._oni_pos))
        return cfg.distance_weight * dist * 0.01  # スケール調整

    @property
    def oni_position(self) -> np.ndarray:
        return self._oni_pos.copy()
