#!/bin/bash
# Google Colab 用セットアップスクリプト
# Colab のセルで実行: !bash setup_colab.sh

set -e

echo "=== RoboQuest2026 セットアップ ==="

# 1. リポジトリのクローン（まだの場合）
if [ ! -d "RoboQuest2026" ]; then
  echo "[1/4] リポジトリをクローン..."
  git clone https://github.com/SingularityBattleQuest/RoboQuest2026.git
  cd RoboQuest2026
else
  cd RoboQuest2026
  echo "[1/4] リポジトリは既に存在します"
fi

# 2. 依存ライブラリのインストール
echo "[2/4] ライブラリをインストール中..."
# mjswan==0.8.2 固定（0.9 系は API 破壊 + mjlab との mujoco pin 衝突）。
# mujoco は mjswan の pin（==3.8.1）に従うのでここでは指定しない。
pip install -q gymnasium "stable-baselines3[extra]" mediapy tqdm pandas matplotlib \
  imageio imageio-ffmpeg "onnx>=1.20.0" onnxruntime "mjswan==0.8.2"

# 3. Go2 モデルファイルのダウンロード
echo "[3/4] Go2 モデルをダウンロード中..."
python scripts/download_models.py

# 4. パッケージのインストール
echo "[4/4] roboquest パッケージをインストール中..."
pip install -q -e .

# ブラウザビューアーは mjswan がノートブック内でビルドする（vendor assets 不要）。
# mjswan 0.8.2 の wheel はビルド済み dist を同梱しているので Node は不要。

echo ""
echo "✅ セットアップ完了！ノートブックを実行できます。"
