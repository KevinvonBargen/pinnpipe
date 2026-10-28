import os

from setuptools import find_packages, setup

_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_CURRENT_DIR, "README.md"), encoding="utf-8") as f:
    README = f.read()

setup(
    name="pinnpipe",
    version="1.0.0",
    packages=find_packages(include=["pinnpipe", "pinnpipe.*"]),
    python_requires=">=3.9",
    install_requires=[
        "chex",
        "flax",
        "jax",
        "jaxlib",
        "jaxtyping",
        "matplotlib",
        "ml-collections",
        "numpy",
        "optax",
        "scipy",
        "tabulate",
        "wandb",
    ],
    extras_require={
        "testing": ["pytest"],
    },
    license="Apache-2.0",
    description="A JAX/Flax library and benchmark suite for evaluating training choices of physics-informed neural networks.",
    long_description=README,
    long_description_content_type="text/markdown",
)
