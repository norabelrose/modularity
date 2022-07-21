from setuptools import setup


setup(
    name="modularity",
    version="0.1.0",
    description="Experiments with induced modularity in neural networks",
    author="Nora Belrose",
    install_requires=[
        "numpy",
        "pytorch_lightning",
        "scipy",
        "torch",
        "torchvision",
        "tqdm"
    ],
)
