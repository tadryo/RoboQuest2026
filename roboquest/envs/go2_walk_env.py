"""
Go2 四足歩行ロボットの強化学習環境（歩行事前学習用）

unitree_rl_mjlab の velocity task を参考にした報酬設計:
  - 速度コマンドへの追従（線速度・角速度）
  - 姿勢の安定性（重力方向の傾き）
  - エネルギー効率（トルク・アクション変化）
  - 足のスリップ防止

観測空間: 45次元
行動空間: 12次元（正規化された目標関節角度オフセット [-1, 1]）
"""
import os

from typing import Optional

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from roboquest.utils.reward_utils import WalkRewardConfig

# デフォルト立ち姿勢 keyframe "home" より
# 順序: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
#       RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
STANDING_POS = np.array([
    0.0,  0.9, -1.8,
    0.0,  0.9, -1.8,
    0.0,  0.9, -1.8,
    0.0,  0.9, -1.8,
], dtype=np.float64)

ACTION_SCALE = 0.3   # action * ACTION_SCALE + STANDING_POS = 目標関節角度
KP = 20.0            # 位置ゲイン
KD = 0.5             # 微分ゲイン

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "go2")
# go2.xml には床がないため walk_scene.xml（床付き）をデフォルトとして使用
MODEL_XML = os.path.join(_MODEL_DIR, "walk_scene.xml")

# 速度コマンドのサンプリング範囲
VEL_CMD_RANGE = {
    "vx":    (-1.0,  1.0),   # 前後 (m/s 相当)
    "vy":    (-0.5,  0.5),   # 左右
    "omega": (-1.0,  1.0),   # 回転 (rad/s 相当)
}

# 足のゼオム名（foot slip 計算用）
FOOT_GEOM_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]


class Go2WalkEnv(gym.Env):
    """Go2 四足歩行の基本環境（速度コマンド追従）。

    観測 (45次元):
      [0:3]   速度コマンド (vx, vy, omega)
      [3:6]   胴体角速度 xyz
      [6:9]   重力方向ベクトル（ボディフレーム）
      [9:21]  関節角度（立ち姿勢からの相対値）
      [21:33] 関節角速度
      [33:45] 前回行動
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        reward_config: Optional[WalkRewardConfig] = None,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        xml_path: Optional[str] = None,
        randomize_cmd: bool = True,
    ):
        super().__init__()
        self.reward_config = reward_config or WalkRewardConfig()
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.randomize_cmd = randomize_cmd
        self._step_count = 0

        xml = xml_path or MODEL_XML
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)

        obs_dim = 3 + 3 + 3 + 12 + 12 + 12  # = 45
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self._last_action = np.zeros(12, dtype=np.float64)
        self._vel_cmd = np.zeros(3, dtype=np.float64)   # [vx, vy, omega]

        # 足ゼオム ID の取得（foot slip 報酬用）
        self._foot_geom_ids = []
        for name in FOOT_GEOM_NAMES:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._foot_geom_ids.append(gid)

        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        else:
            self._renderer = None

    # ── 公開 API ─────────────────────────────────────────────────────────

    def set_vel_cmd(self, vx: float, vy: float, omega: float) -> None:
        """外部（高レベルポリシー）から速度コマンドを設定する。"""
        self._vel_cmd = np.array([vx, vy, omega], dtype=np.float64)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # keyframe "home" から起動＋微小ノイズ
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.qpos[7:19] += self.np_random.uniform(-0.05, 0.05, 12)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # 速度コマンドをランダムサンプリング
        if self.randomize_cmd:
            self._vel_cmd = np.array([
                self.np_random.uniform(*VEL_CMD_RANGE["vx"]),
                self.np_random.uniform(*VEL_CMD_RANGE["vy"]),
                self.np_random.uniform(*VEL_CMD_RANGE["omega"]),
            ], dtype=np.float64)

        self._step_count = 0
        self._last_action = np.zeros(12, dtype=np.float64)
        return self._get_obs().astype(np.float32), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self._apply_pd_control(action)

        # 物理サブステップ（制御周期 0.02s, substep 5回）
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs().astype(np.float32)
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self._step_count >= self.max_episode_steps

        if terminated:
            reward -= self.reward_config.fall_penalty

        self._last_action = action.copy()
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()

    # ── 観測 ─────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        ang_vel = self.data.qvel[3:6].copy()
        proj_grav = self._projected_gravity()
        jpos = self.data.qpos[7:19] - STANDING_POS
        jvel = self.data.qvel[6:18].copy()
        return np.concatenate([
            self._vel_cmd,
            ang_vel,
            proj_grav,
            jpos,
            jvel,
            self._last_action,
        ])

    def _projected_gravity(self) -> np.ndarray:
        """重力ベクトル [0,0,-1] をボディフレームに回転。"""
        qw, qx, qy, qz = self.data.qpos[3:7]
        gravity_world = np.array([0.0, 0.0, -1.0])
        # 四元数の逆回転: R^T @ gravity_world
        R = np.array([
            [1-2*(qy**2+qz**2),  2*(qx*qy+qw*qz),  2*(qx*qz-qw*qy)],
            [2*(qx*qy-qw*qz),  1-2*(qx**2+qz**2),   2*(qy*qz+qw*qx)],
            [2*(qx*qz+qw*qy),    2*(qy*qz-qw*qx),  1-2*(qx**2+qy**2)],
        ])
        return R.T @ gravity_world

    # ── 制御 ─────────────────────────────────────────────────────────────

    def _apply_pd_control(self, action: np.ndarray) -> None:
        q_target = STANDING_POS + action * ACTION_SCALE
        q_cur = self.data.qpos[7:19]
        dq_cur = self.data.qvel[6:18]
        tau = KP * (q_target - q_cur) - KD * dq_cur
        limits = self.model.actuator_ctrlrange
        self.data.ctrl[:] = np.clip(tau, limits[:, 0], limits[:, 1])

    # ── 報酬 ─────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray) -> float:
        cfg = self.reward_config

        # 1. 線速度追跡（Gaussian）
        lin_vel_body = self.data.qvel[:2]  # x,y 速度（近似: ワールドフレーム）
        lin_err = np.sum((self._vel_cmd[:2] - lin_vel_body) ** 2)
        r_lin = cfg.lin_vel_weight * float(np.exp(-lin_err / 0.25))

        # 2. 角速度追跡（Gaussian）
        ang_vel_z = self.data.qvel[5]
        ang_err = (self._vel_cmd[2] - ang_vel_z) ** 2
        r_ang = cfg.ang_vel_weight * float(np.exp(-ang_err / 0.5))

        # 3. 姿勢ペナルティ（重力方向の xy 傾き）
        proj_grav = self._projected_gravity()
        r_orient = cfg.orientation_weight * float(np.sum(proj_grav[:2] ** 2))

        # 4. トルクペナルティ
        r_torque = cfg.torques_weight * float(np.sum(self.data.ctrl ** 2))

        # 5. アクション変化ペナルティ
        r_rate = cfg.action_rate_weight * float(np.sum((action - self._last_action) ** 2))

        # 6. 足スリップペナルティ
        r_slip = self._foot_slip_penalty(cfg)

        return r_lin + r_ang + r_orient + r_torque + r_rate + r_slip

    def _foot_slip_penalty(self, cfg: WalkRewardConfig) -> float:
        if not self._foot_geom_ids or cfg.foot_slip_weight == 0:
            return 0.0
        penalty = 0.0
        for geom_id in self._foot_geom_ids:
            body_id = self.model.geom_bodyid[geom_id]
            # 接触チェック
            in_contact = any(
                c.geom1 == geom_id or c.geom2 == geom_id
                for c in self.data.contact[:self.data.ncon]
            )
            if in_contact:
                # 足の水平速度
                foot_vel = np.zeros(6)
                mujoco.mj_objectVelocity(
                    self.model, self.data,
                    mujoco.mjtObj.mjOBJ_BODY, body_id, foot_vel, 0
                )
                slip = float(np.sum(foot_vel[3:5] ** 2))
                penalty += slip
        return cfg.foot_slip_weight * penalty

    # ── 終了判定 ──────────────────────────────────────────────────────────

    def _is_terminated(self) -> bool:
        return bool(self.data.qpos[2] < 0.15)  # 胴体高さ < 0.15m = 転倒

    # ── プロパティ ────────────────────────────────────────────────────────

    @property
    def robot_xy(self) -> np.ndarray:
        return self.data.qpos[:2].copy()

    @property
    def robot_height(self) -> float:
        return float(self.data.qpos[2])

    @property
    def vel_cmd(self) -> np.ndarray:
        return self._vel_cmd.copy()
