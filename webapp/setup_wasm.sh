#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_wasm.sh — mujoco_wasm dist ファイルをダウンロードする
#
# 使い方:
#   cd webapp
#   bash setup_wasm.sh
#
# 完了後 webapp/dist/ に以下のファイルが作成されます:
#   mujoco_wasm.js   (ES module エントリポイント)
#   mujoco_wasm.wasm (WebAssembly バイナリ)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="zalo/mujoco_wasm"
# 最新リリースのタグを自動取得（GitHub API）
LATEST=$(curl -s "https://api.github.com/repos/${REPO}/releases/latest" \
         | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": "\(.*\)".*/\1/')

if [ -z "$LATEST" ]; then
  echo "❌ GitHub API からバージョン取得に失敗しました。"
  echo "   手動で https://github.com/${REPO}/releases から"
  echo "   mujoco_wasm.js と mujoco_wasm.wasm を webapp/dist/ にコピーしてください。"
  exit 1
fi

echo "📦 mujoco_wasm ${LATEST} をダウンロード中…"

DIST_DIR="$(dirname "$0")/dist"
mkdir -p "$DIST_DIR"

BASE_URL="https://github.com/${REPO}/releases/download/${LATEST}"

for f in mujoco_wasm.js mujoco_wasm.wasm; do
  URL="${BASE_URL}/${f}"
  echo "  ↓ ${f}"
  curl -L -o "${DIST_DIR}/${f}" "${URL}"
done

echo ""
echo "✅ 完了！ファイルは ${DIST_DIR}/ に保存されました。"
echo ""
echo "次のステップ:"
echo "  1. ローカルサーバーを起動: python3 -m http.server 8080 (リポジトリルートから)"
echo "  2. ブラウザで開く: http://localhost:8080/webapp/"
echo ""
echo "GitHub Pages にデプロイする場合:"
echo "  git add webapp/dist/ && git commit -m 'add mujoco_wasm dist'"
echo "  git push"
