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
pip install -q gymnasium "stable-baselines3[extra]" mediapy tqdm pandas matplotlib \
  imageio imageio-ffmpeg "onnx>=1.20.0" onnxruntime

# mjswan（ブラウザビューアー）は別途インストールする。
# - 0.8.2 固定：0.9 系は API 破壊（観測関数が mjlab へ移動）＋ mjlab との mujoco pin 衝突。
# - mjswan の wheel は Requires-Python が <3.13 だが、中身は純 Python + ビルド済み
#   フロントエンドで、依存（mujoco==3.8.1 / onnx）にも cp313 wheel があるため
#   Python 3.13（現在の Colab）でも動作する。3.13 では pip が候補から外すので
#   --ignore-requires-python を付ける。
# - mujoco は mjswan の pin（==3.8.1）に従うのでここでは指定しない。
MJSWAN_FLAGS=""
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)'; then
  MJSWAN_FLAGS="--ignore-requires-python"
fi
if ! pip install -q $MJSWAN_FLAGS "mjswan==0.8.2"; then
  echo "⚠ mjswan のインストールに失敗しました。ブラウザビューアーのセルのみ使えません。"
  echo "  学習・動画のセルはそのまま実行できます。"
fi

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
