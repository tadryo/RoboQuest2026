"""
Export SB3 policies and VecNormalize statistics for the browser viewer.

The Colab viewer runs entirely in the browser:

  flee_policy.onnx  : 10-dim high-level obs -> [vx, vy, omega]
  walk_policy.onnx  : 45-dim walk obs -> 12 joint actions
  *_vecnorm.json    : observation normalization statistics

The exported ONNX graph contains only deterministic actions, which keeps the
browser side simple and matches `model.predict(..., deterministic=True)`.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class DeterministicPolicyONNX(torch.nn.Module):
    """Small wrapper around an SB3 ActorCriticPolicy for ONNX export."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.register_buffer(
            "action_low",
            torch.as_tensor(policy.action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.as_tensor(policy.action_space.high, dtype=torch.float32),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        actions = self.policy._predict(obs, deterministic=True)
        return torch.clamp(actions, self.action_low, self.action_high)


def _model_zip_exists(model_path: str | Path) -> bool:
    path = Path(model_path)
    return path.exists() or path.with_suffix(".zip").exists()


def export_policy_onnx(model_path: str | Path, output_path: str | Path) -> Path:
    """Export an SB3 PPO policy to a deterministic-action ONNX file."""
    from stable_baselines3 import PPO

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading PPO model: {model_path}")
    model = PPO.load(str(model_path), device="cpu")
    model.policy.eval()

    obs_dim = int(model.policy.observation_space.shape[0])
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    wrapper = DeterministicPolicyONNX(model.policy)

    export_kwargs = {
        "opset_version": 17,
        "input_names": ["obs"],
        "output_names": ["action"],
        "dynamic_axes": {"obs": {0: "batch"}, "action": {0: "batch"}},
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(wrapper, dummy_obs, str(output_path), **export_kwargs)
    print(f"Exported ONNX: {output_path}")
    return output_path


def export_vecnorm_stats(vecnorm_path: str | Path, output_path: str | Path) -> Optional[Path]:
    """Export VecNormalize observation statistics to JSON."""
    vecnorm_path = Path(vecnorm_path)
    output_path = Path(output_path)
    if not vecnorm_path.exists():
        print(f"VecNormalize stats not found, skipping: {vecnorm_path}")
        return None

    with vecnorm_path.open("rb") as f:
        vec_norm = pickle.load(f)

    obs_rms = vec_norm.obs_rms
    stats = {
        "obs_rms_mean": np.asarray(obs_rms.mean, dtype=np.float64).tolist(),
        "obs_rms_var": np.asarray(obs_rms.var, dtype=np.float64).tolist(),
        "clip_obs": float(vec_norm.clip_obs),
        "epsilon": float(getattr(vec_norm, "epsilon", 1e-8)),
        "obs_dim": int(np.asarray(obs_rms.mean).shape[0]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Exported VecNormalize JSON: {output_path}")
    return output_path


def verify_policy_export(model_path: str | Path, onnx_path: str | Path) -> None:
    """Compare SB3 deterministic action with ONNX Runtime on one sample."""
    try:
        import onnxruntime as ort
        from stable_baselines3 import PPO
    except ImportError:
        print("ONNX verification skipped: install onnxruntime to enable it.")
        return

    model = PPO.load(str(model_path), device="cpu")
    obs_dim = int(model.policy.observation_space.shape[0])
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(1, obs_dim)).astype(np.float32)

    with torch.no_grad():
        torch_action = DeterministicPolicyONNX(model.policy)(torch.as_tensor(obs))
        torch_action = torch_action.detach().cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_action = sess.run(None, {"obs": obs})[0]
    max_abs = float(np.max(np.abs(torch_action - ort_action)))
    print(f"Verification max abs diff: {max_abs:.6g}")
    if max_abs > 1e-4:
        raise RuntimeError(f"ONNX verification failed: max abs diff {max_abs:.6g}")


def export_named_policy(
    name: str,
    model_path: str | Path,
    vecnorm_path: str | Path,
    save_dir: str | Path,
    verify: bool = True,
) -> bool:
    """Export one policy family. Returns False when the model is absent."""
    if not _model_zip_exists(model_path):
        print(f"{name} model not found, skipping: {model_path}.zip")
        return False

    save_dir = Path(save_dir)
    onnx_path = export_policy_onnx(model_path, save_dir / f"{name}_policy.onnx")
    export_vecnorm_stats(vecnorm_path, save_dir / f"{name}_vecnorm.json")
    if verify:
        verify_policy_export(model_path, onnx_path)
    return True


def export_all_for_web(
    walk_model: str | Path = "/tmp/walk_model",
    walk_vecnorm: str | Path = "/tmp/walk_model_vecnorm.pkl",
    flee_model: str | Path = "/tmp/flee_model",
    flee_vecnorm: str | Path = "/tmp/flee_model_vecnorm.pkl",
    save_dir: str | Path = "webapp/models",
    verify: bool = True,
) -> dict[str, bool]:
    """Export every model that exists and return which ones were written."""
    results = {
        "walk": export_named_policy("walk", walk_model, walk_vecnorm, save_dir, verify),
        "flee": export_named_policy("flee", flee_model, flee_vecnorm, save_dir, verify),
    }
    print(f"Web export summary: {results}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RoboQuest policies for web viewer")
    parser.add_argument("--walk_model", default="/tmp/walk_model")
    parser.add_argument("--walk_vecnorm", default="/tmp/walk_model_vecnorm.pkl")
    parser.add_argument("--flee_model", default="/tmp/flee_model")
    parser.add_argument("--flee_vecnorm", default="/tmp/flee_model_vecnorm.pkl")
    parser.add_argument("--save_dir", default=str(ROOT / "webapp" / "models"))
    parser.add_argument(
        "--only",
        choices=["all", "walk", "flee"],
        default="all",
        help="Export a subset of policies",
    )
    parser.add_argument("--no_verify", action="store_true", help="Skip ONNX Runtime check")
    args = parser.parse_args()

    verify = not args.no_verify
    if args.only == "walk":
        ok = export_named_policy(
            "walk", args.walk_model, args.walk_vecnorm, args.save_dir, verify
        )
        if not ok:
            sys.exit(1)
    elif args.only == "flee":
        ok = export_named_policy(
            "flee", args.flee_model, args.flee_vecnorm, args.save_dir, verify
        )
        if not ok:
            sys.exit(1)
    else:
        results = export_all_for_web(
            args.walk_model,
            args.walk_vecnorm,
            args.flee_model,
            args.flee_vecnorm,
            args.save_dir,
            verify,
        )
        if not any(results.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()
