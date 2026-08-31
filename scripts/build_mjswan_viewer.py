"""
build_mjswan_viewer.py — mjswan を使って RoboQuest 2026 のブラウザビューアーをビルドする。

使い方（Colab セルから）:
    from scripts.build_mjswan_viewer import build_walk, build_flee
    import mjswan

    # Walk ビューアー（学習済みポリシーで自律歩行）
    app = build_walk(
        walk_onnx_path='webapp/models/walk_policy_normalized.onnx',
        output_dir='/tmp/rq_walk_dist',
    )
    app.launch(height=620)

    # Flee ビューアー（アリーナ + 手動 WASD 操作）
    app = build_flee(
        walk_onnx_path='webapp/models/walk_policy_normalized.onnx',
        output_dir='/tmp/rq_flee_dist',
    )
    app.launch(height=620)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
# MjSpec.from_file が include を解決するにはリポジトリルートが cwd である必要がある
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# ── 関節 / アクチュエータ定義 ──────────────────────────────────────────────────
# go2_posctrl.xml のアクチュエータ順と一致させる（訓練時の obs/action 順序）
JOINT_NAMES: list[str] = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
# go2_posctrl.xml の <position> アクチュエータ名（JOINT_NAMES と同順）。
# mjswan の joint_position アクションは関節名でマッチするため直接は使わないが、
# 訓練側（アクチュエータ順の obs/action）との対応を示すために残す。
ACTUATOR_NAMES: list[str] = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

# 訓練時の STANDING_POS (JOINT_NAMES 順)
_STANDING_VALUES = [0, 0.9, -1.8] * 4  # FR/FL/RR/RL × hip/thigh/calf
STANDING_POS: dict[str, float] = dict(zip(JOINT_NAMES, _STANDING_VALUES))

# PD ゲイン（訓練と完全一致させる）
KP = 20.0   # stiffness
KD = 0.5    # damping
ACTION_SCALE = 0.3

# add_policy の commands キー / generated_commands の command_name
VELOCITY_COMMAND_NAME = "velocity"


def _ensure_normalized_onnx(
    onnx_path: Path,
    trained_model: str = "/tmp/walk_model",
    vecnorm: str = "/tmp/walk_model_vecnorm.pkl",
) -> Path:
    """normalized ONNX が無ければ学習済みモデルから自動エクスポートする。

    古い学習セルで学習した場合や、再学習せずにビューアーだけ動かしたい場合に、
    /tmp/walk_model.zip から _normalized.onnx を生成する。
    """
    if onnx_path.exists():
        return onnx_path

    model_zip = Path(str(trained_model) + ".zip")
    if not (Path(trained_model).exists() or model_zip.exists()):
        raise FileNotFoundError(
            f"Walk ポリシーが見つかりません: {onnx_path}\n"
            f"学習済みモデルもありません: {model_zip}\n"
            "「🚀 歩行学習スタート！」セルを先に実行してください。"
        )

    print(f"⚙  {onnx_path.name} が無いので {model_zip.name} から自動生成します...")
    from scripts.export_for_web import export_normalized_policy_onnx
    export_normalized_policy_onnx(trained_model, vecnorm, onnx_path)
    return onnx_path


def _make_walk_obs():
    """Walk ポリシーの観測定義（45次元、訓練時の順序と完全一致）。

    訓練時の obs 順:
        vel_cmd(3) | ang_vel(3) | projected_gravity(3)
        | joint_pos_rel(12) | joint_vel(12) | last_action(12)
    """
    from mjswan import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjswan.envs.mdp.observations import (
        base_ang_vel,
        generated_commands,
        joint_pos_rel,
        joint_vel_rel,
        last_action,
        projected_gravity,
    )

    return ObservationGroupCfg(terms={
        # generated_commands は params で command 名を指定する必要がある
        # （add_policy の commands={"velocity": ...} のキーと一致させる）
        "vel_cmd":           ObservationTermCfg(
            func=generated_commands,
            params={"command_name": VELOCITY_COMMAND_NAME},
        ),                                                               # 3
        "ang_vel":           ObservationTermCfg(func=base_ang_vel),          # 3
        "projected_gravity": ObservationTermCfg(func=projected_gravity),     # 3
        "joint_pos":         ObservationTermCfg(func=joint_pos_rel),         # 12
        "joint_vel":         ObservationTermCfg(func=joint_vel_rel),         # 12
        "last_action":       ObservationTermCfg(func=last_action),           # 12
    })  # 合計 45次元


def _make_walk_action():
    """Walk ポリシーのアクション定義（KP/KD/scale を訓練時と一致させる）。

    訓練時: q_target = STANDING_POS + action * 0.3 → PD 制御 (KP=20, KD=0.5)
    mjswan: q_target = default_offset + action * scale（use_default_offset=True）
    """
    from mjswan.envs.mdp.actions import JointPositionActionCfg

    # 注意: type="joint_position" の場合、mjswan のブラウザ実行時は
    # actuator_names を *policy_joint_names（= 関節名）* に対して正規表現マッチする
    # （アクチュエータ名でマッチするのは muscle_activation のみ）。
    # ここに <actuator> 名（FR_hip 等）を渡すと 1つもマッチせず
    # 「no joints matched patterns」で action term ごと無視され、ロボットが動かない。
    return JointPositionActionCfg(
        actuator_names=tuple(JOINT_NAMES),
        scale=ACTION_SCALE,
        stiffness=KP,
        damping=KD,
        use_default_offset=True,
    )


def _make_velocity_command():
    """WASD キーボード / UI スライダーによる速度コマンド設定。"""
    from mjswan import velocity_command

    return velocity_command(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-1.0, 1.0),
        default_lin_vel_x=0.0,
        default_lin_vel_y=0.0,
        default_ang_vel_z=0.0,
    )


def _select_reset_keyframe(spec, keep: str) -> None:
    """`keep` 以外の keyframe を削除して、それを keyframe 0 にする。

    mjswan のブラウザランタイムはリセット時に必ず keyframe 0
    （mj_resetDataKeyframe(..., 0)）を使う。arena_web.xml は
    go2_posctrl.xml の "home"（鬼の座標を含まない）を include するため、
    そのままだと "home" が index 0 になり arena_home が無視されて
    鬼がロボットと同じ原点に出てしまう。
    """
    for key in list(spec.keys):
        if key.name != keep:
            spec.delete(key)


# ── Public API ────────────────────────────────────────────────────────────────

def build_walk(
    walk_onnx_path: str | Path = "webapp/models/walk_policy_normalized.onnx",
    output_dir: str | Path = "/tmp/rq_walk_dist",
) -> "mjswan.mjswanApp":
    """Walk ビューアーをビルドして mjswanApp を返す。

    学習済み Walk ポリシー（WASD で速度コマンドを与えると歩く）。
    VecNormalize が埋め込まれた _normalized.onnx を必ず渡すこと。
    """
    import mujoco
    import onnx
    import mjswan

    walk_onnx_path = _ensure_normalized_onnx(Path(walk_onnx_path))
    output_dir     = Path(output_dir)

    print(f"🔧 Walk ビューアーをビルド中... ({walk_onnx_path.name})")
    # walk_web.xml = go2_simple（メッシュなし）+ 床（ブラウザ互換）
    spec   = mujoco.MjSpec.from_file(str(ROOT / "models" / "go2" / "walk_web.xml"))
    policy = onnx.load(str(walk_onnx_path))

    builder = mjswan.Builder()
    (
        builder
        .add_project(name="RoboQuest 2026 — 歩行ビューアー")
        .add_scene(name="Go2 Walk", spec=spec)
        .add_policy(
            name="Walk Policy",
            policy=policy,
            observations={"policy": _make_walk_obs()},
            actions={"joint_pos": _make_walk_action()},
            policy_joint_names=JOINT_NAMES,
            default_joint_pos=STANDING_POS,
            commands={VELOCITY_COMMAND_NAME: _make_velocity_command()},
        )
    )

    app = builder.build(output_dir=str(output_dir))
    print("✅ Walk ビューアービルド完了")
    return app


def build_flee(
    walk_onnx_path: str | Path = "webapp/models/walk_policy_normalized.onnx",
    output_dir: str | Path = "/tmp/rq_flee_dist",
) -> "mjswan.mjswanApp":
    """Flee（鬼ごっこ）ビューアーをビルドして mjswanApp を返す。

    アリーナ（壁 + 鬼ボディ）を表示し、Walk ポリシーで WASD 手動操作する。
    ※ 高レベル Flee ポリシーの統合は将来実装予定。
    """
    import mujoco
    import onnx
    import mjswan

    walk_onnx_path = _ensure_normalized_onnx(Path(walk_onnx_path))
    output_dir     = Path(output_dir)

    print(f"🔧 Flee ビューアーをビルド中... ({walk_onnx_path.name})")
    # arena_web.xml = go2_simple（メッシュなし）+ 壁 + 鬼ボディ
    spec   = mujoco.MjSpec.from_file(str(ROOT / "models" / "go2" / "arena_web.xml"))
    # ブラウザ側は keyframe 0 でリセットするので arena_home を先頭に持ってくる
    _select_reset_keyframe(spec, keep="arena_home")
    policy = onnx.load(str(walk_onnx_path))

    builder = mjswan.Builder()
    (
        builder
        .add_project(name="RoboQuest 2026 — 鬼ごっこビューアー")
        .add_scene(name="Go2 vs Oni", spec=spec)
        .add_policy(
            name="Walk Policy (Manual)",
            policy=policy,
            observations={"policy": _make_walk_obs()},
            actions={"joint_pos": _make_walk_action()},
            policy_joint_names=JOINT_NAMES,
            default_joint_pos=STANDING_POS,
            commands={VELOCITY_COMMAND_NAME: _make_velocity_command()},
        )
    )

    app = builder.build(output_dir=str(output_dir))
    print("✅ Flee ビューアービルド完了")
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="mjswan ビューアーをビルド")
    parser.add_argument("mode", choices=["walk", "flee", "both"], default="both", nargs="?")
    parser.add_argument("--walk_onnx", default="webapp/models/walk_policy_normalized.onnx")
    parser.add_argument("--out_walk",  default="/tmp/rq_walk_dist")
    parser.add_argument("--out_flee",  default="/tmp/rq_flee_dist")
    args = parser.parse_args()

    if args.mode in ("walk", "both"):
        build_walk(args.walk_onnx, args.out_walk)
    if args.mode in ("flee", "both"):
        build_flee(args.walk_onnx, args.out_flee)
