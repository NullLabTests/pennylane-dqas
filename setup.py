from setuptools import setup, find_packages

setup(
    name="pennylane-dqas",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pennylane>=0.38",
        "torch>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
    ],
    python_requires=">=3.10",
)
