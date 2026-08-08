from setuptools import setup, find_packages

setup(
    name="modality_gap",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
    ],
    author="Expert PyTorch Developer",
    description="Multimodal representation learning project comparing Sufficient Statistic Distillation (SSD) and CLIP baseline for modality gap elimination",
    python_requires=">=3.8",
)
