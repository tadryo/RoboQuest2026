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
        "onnx>=1.16.0",
        "onnxruntime>=1.17.0",
        "mediapy>=1.1.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.66.0",
        "pandas>=2.0.0",
    ],
)
