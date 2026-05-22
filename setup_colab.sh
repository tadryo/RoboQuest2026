#!/bin/bash
# Google Colab 用セットアップスクリプト
# Colab のセルで実行: !bash setup_colab.sh

set -e

echo "=== RoboQuest2026 セットアップ ==="

# 1. リポジトリのクローン（まだの場合）
if [ ! -d "RoboQuest2026" ]; then
  echo "[1/4] リポジトリをクローン..."
  git clone https://github.com/tadryo/RoboQuest2026.git
  cd RoboQuest2026
else
  cd RoboQuest2026
  echo "[1/4] リポジトリは既に存在します"
fi

# 2. 依存ライブラリのインストール
echo "[2/4] ライブラリをインストール中..."
pip install -q mujoco gymnasium stable-baselines3 mediapy tqdm pandas matplotlib imageio imageio-ffmpeg

# 3. Go2 モデルファイルのダウンロード
echo "[3/4] Go2 モデルをダウンロード中..."
python scripts/download_models.py

# 4. パッケージのインストール
echo "[4/4] roboquest パッケージをインストール中..."
pip install -q -e .

echo ""
echo "✅ セットアップ完了！ノートブックを実行できます。"
