from setuptools import setup, find_packages

setup(
    name="roboquest",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mujoco>=3.1.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.3.0",
        "numpy>=1.24.0",
        "onnx>=1.20.0",
        "onnxruntime>=1.17.0",
        # ブラウザビューアー。0.9 系は API 破壊 + mjlab との mujoco pin 衝突のため固定
        "mjswan==0.8.2",
        "mediapy>=1.1.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.66.0",
        "pandas>=2.0.0",
    ],
)
