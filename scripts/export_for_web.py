"""
export_for_web.py — 学習済みモデルを Web ビューアー用にエクスポート

出力:
  <save_dir>/policy.onnx          高レベル逃げポリシー (ONNX)
  <save_dir>/vecnorm_stats.json   VecNormalize 統計 (観測正規化)

使い方:
  python scripts/export_for_web.py \\
      --flee_model /tmp/flee_model \\
      --vecnorm    /tmp/flee_model_vecnorm.pkl \\
      --save_dir   webapp/models

Web ビューアーでの読み込み:
  index.html の「AIポリシー」パネルから
    - policy.onnx          → 「モデルを読み込む」
    - vecnorm_stats.json   → 「VecNorm stats」
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def export_policy_onnx(flee_model_path: str, save_dir: str) -> str:
    """SB3 PPO ポリシーを ONNX へエクスポート。"""
    from stable_baselines3 import PPO

    print(f"📂 モデル読み込み: {flee_model_path}.zip")
    model = PPO.load(flee_model_path, device="cpu")

    policy = model.policy
    policy.eval()

    # 観測次元（高レベルポリシー: 10次元）
    obs_dim = policy.observation_space.shape[0]
    dummy   = torch.zeros(1, obs_dim, dtype=torch.float32)

    onnx_path = os.path.join(save_dir, "policy.onnx")
    torch.onnx.export(
        policy,
        dummy,
        onnx_path,
        opset_version=12,
        input_names=["obs"],
        output_names=["actions", "values", "log_probs"],
        dynamic_axes={"obs": {0: "batch_size"}},
        do_constant_folding=True,
    )
    print(f"✅ ONNX エクスポート完了: {onnx_path}")
    return onnx_path


def export_vecnorm_stats(vecnorm_path: str, save_dir: str) -> str:
    """VecNormalize の観測統計を JSON へエクスポート。"""
    from stable_baselines3.common.vec_env import VecNormalize

    print(f"📂 VecNormalize 読み込み: {vecnorm_path}")
    with open(vecnorm_path, "rb") as f:
        vec_norm = pickle.load(f)

    obs_rms = vec_norm.obs_rms
    stats = {
        "obs_rms_mean": obs_rms.mean.tolist(),
        "obs_rms_var":  obs_rms.var.tolist(),
        "clip_obs":     float(vec_norm.clip_obs),
        "obs_dim":      int(obs_rms.mean.shape[0]),
    }

    json_path = os.path.join(save_dir, "vecnorm_stats.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ VecNorm 統計エクスポート完了: {json_path}")
    return json_path


def verify_onnx(onnx_path: str) -> None:
    """ONNX モデルの動作確認。"""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, 10), dtype=np.float32)
        out = sess.run(None, {"obs": dummy})
        actions = out[0][0]
        print(f"🧪 推論テスト OK — 出力 actions: {actions}")
    except ImportError:
        print("⚠  onnxruntime がインストールされていません（確認スキップ）")
        print("   pip install onnxruntime  でインストールできます")
    except Exception as e:
        print(f"❌ 推論テスト失敗: {e}")


def main():
    parser = argparse.ArgumentParser(description="SB3 モデルを Web エクスポート")
    parser.add_argument("--flee_model", type=str, default="/tmp/flee_model",
                        help="逃げポリシーのパス（.zip なし）")
    parser.add_argument("--vecnorm", type=str, default="/tmp/flee_model_vecnorm.pkl",
                        help="VecNormalize の pkl ファイル")
    parser.add_argument("--save_dir", type=str, default="webapp/models",
                        help="エクスポート先ディレクトリ")
    parser.add_argument("--skip_vecnorm", action="store_true",
                        help="VecNorm エクスポートをスキップ")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Policy → ONNX
    if not os.path.exists(args.flee_model + ".zip"):
        print(f"❌ モデルが見つかりません: {args.flee_model}.zip")
        sys.exit(1)
    onnx_path = export_policy_onnx(args.flee_model, args.save_dir)

    # 2. VecNormalize stats → JSON
    if not args.skip_vecnorm:
        if os.path.exists(args.vecnorm):
            export_vecnorm_stats(args.vecnorm, args.save_dir)
        else:
            print(f"⚠  VecNorm ファイルが見つかりません: {args.vecnorm}（スキップ）")

    # 3. Verify
    print("\n🧪 ONNX 動作確認…")
    verify_onnx(onnx_path)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎉 エクスポート完了！

ファイル:
  {args.save_dir}/policy.onnx
  {args.save_dir}/vecnorm_stats.json

Web ビューアーでの使い方:
  1. webapp/index.html をブラウザで開く
  2. 右パネル「AIポリシー」→「モデルを読み込む」で policy.onnx を選択
  3. 「VecNorm stats」で vecnorm_stats.json を選択
  4. ▶ 開始 ボタンでシミュレーション開始！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    main()
