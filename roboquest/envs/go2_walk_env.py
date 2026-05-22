"""
Go2 四足歩行ロボットの強化学習環境（Gymnasium ラッパー）

unitree_mujoco の go2.xml モデルを使用し、PD 制御で関節を動かす。
観測空間: 43次元
行動空間: 12次元（正規化された目標関節角度 [-1, 1]）
"""
import os
from typing import Optional, Tuple

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from roboquest.utils.reward_utils import RewardConfig

# go2.xml のデフォルト立ち姿勢（keyframe "home" より）
# 順序: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
#       RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
STANDING_POS = np.array([
    0.0,  0.9, -1.8,
    0.0,  0.9, -1.8,
    0.0,  0.9, -1.8,
    0.0,  0.9, -1.8,
], dtype=np.float64)

# 関節可動域（最大偏差 rad）
ACTION_SCALE = 0.3   # action * ACTION_SCALE + STANDING_POS

# PDゲイン（unitree_mujoco 準拠）
KP = 20.0
KD = 0.5

# モデルパス
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "go2")
MODEL_XML = os.path.join(_MODEL_DIR, "go2.xml")


class Go2WalkEnv(gym.Env):
    """Go2 四足歩行の基本環境。

    観測 (43次元):
      [0:3]   胴体 xyz 位置
      [3:7]   胴体四元数 (w, x, y, z)
      [7:10]  胴体線速度 xyz
      [10:13] 胴体角速度 xyz
      [13:25] 関節角度 (立ち姿勢からの相対値)
      [25:37] 関節角速度
      [37:43] 前回行動 (6次元→実際は12次元、下記参照)

    Note: obs[37:49] が前回行動 12次元（合計49次元）。
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        xml_path: Optional[str] = None,
    ):
        super().__init__()
        self.reward_config = reward_config or RewardConfig()
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._step_count = 0

        xml = xml_path or MODEL_XML
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)

        # 観測・行動空間の定義
        obs_dim = 3 + 4 + 3 + 3 + 12 + 12 + 12  # = 49
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self._last_action = np.zeros(12, dtype=np.float64)

        if render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        else:
            self.renderer = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 立ち姿勢から少しランダムノイズを加えて開始
        self.data.qpos[:] = self.model.key_qpos[0]  # keyframe "home"
        noise = self.np_random.uniform(-0.05, 0.05, size=12)
        self.data.qpos[7:19] += noise
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._last_action = np.zeros(12, dtype=np.float64)
        return self._get_obs().astype(np.float32), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self._apply_pd_control(action)

        # 物理ステップ（制御周期: 0.02s = 50Hz、substep 5回）
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self._last_action = action.copy()
        self._step_count += 1

        obs = self._get_obs().astype(np.float32)
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self._step_count >= self.max_episode_steps

        if terminated:
            reward -= self.reward_config.fall_penalty

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.renderer is None:
            return None
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        pos = self.data.qpos[:3].copy()            # 胴体位置 xyz
        quat = self.data.qpos[3:7].copy()          # 四元数 (w, x, y, z)
        lin_vel = self.data.qvel[:3].copy()        # 線速度
        ang_vel = self.data.qvel[3:6].copy()       # 角速度
        jpos = self.data.qpos[7:19] - STANDING_POS # 相対関節角度
        jvel = self.data.qvel[6:18].copy()         # 関節角速度
        last_act = self._last_action.copy()        # 前回行動
        return np.concatenate([pos, quat, lin_vel, ang_vel, jpos, jvel, last_act])

    def _apply_pd_control(self, action: np.ndarray) -> None:
        q_target = STANDING_POS + action * ACTION_SCALE
        q_cur = self.data.qpos[7:19]
        dq_cur = self.data.qvel[6:18]
        tau = KP * (q_target - q_cur) - KD * dq_cur
        # モータートルク上限でクランプ
        ctrl_limits = self.model.actuator_ctrlrange  # shape (12, 2)
        tau = np.clip(tau, ctrl_limits[:, 0], ctrl_limits[:, 1])
        self.data.ctrl[:] = tau

    def _compute_reward(self, action: np.ndarray) -> float:
        cfg = self.reward_config
        # 生存ボーナス
        r_survival = cfg.survival_weight
        # 制御コストペナルティ
        r_ctrl = -cfg.control_weight * float(np.sum(action ** 2))
        # 前進報酬（x 方向速度）
        r_forward = cfg.forward_weight * float(self.data.qvel[0])
        return r_survival + r_ctrl + r_forward

    def _is_terminated(self) -> bool:
        # 胴体高度 < 0.15m で転倒判定
        return bool(self.data.qpos[2] < 0.15)

    # 派生クラスが利用できる位置アクセサ
    @property
    def robot_xy(self) -> np.ndarray:
        return self.data.qpos[:2].copy()

    @property
    def robot_height(self) -> float:
        return float(self.data.qpos[2])
