"""
報酬関数の設定クラスとプリセット

Tier1 のスライダーパラメータがそのままこのクラスのフィールドに対応する。
"""
from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """逃げロボット用の報酬重みを管理するクラス。

    各フィールドが Tier1 のスライダーパラメータに対応。
    """
    # 鬼から離れるほど高い報酬
    distance_weight: float = 1.0

    # 転倒・タグされずに生き延びる毎ステップのボーナス
    survival_weight: float = 0.1

    # 制御量の二乗和ペナルティ（省エネ・滑らかな動き）
    control_weight: float = 0.05

    # 前進速度に対する報酬（直線的な逃げを促進）
    forward_weight: float = 0.5

    # 転倒時のペナルティ（エピソード終了時に引かれる）
    fall_penalty: float = 10.0

    # タグされた（鬼に捕まった）時のペナルティ
    tag_penalty: float = 20.0

    # アリーナ境界への接近ペナルティ
    boundary_penalty: float = 5.0


# --- プリセット ---

def make_aggressive_config() -> RewardConfig:
    """逃げることを最優先する設定（リスクを顧みず距離を稼ぐ）"""
    return RewardConfig(
        distance_weight=3.0,
        survival_weight=0.05,
        control_weight=0.01,
        forward_weight=1.5,
    )


def make_balanced_config() -> RewardConfig:
    """デフォルトの均衡設定"""
    return RewardConfig()


def make_efficient_config() -> RewardConfig:
    """エネルギー効率優先（スタミナ型・長期生存重視）"""
    return RewardConfig(
        distance_weight=0.8,
        survival_weight=0.5,
        control_weight=0.3,
        forward_weight=0.3,
    )
