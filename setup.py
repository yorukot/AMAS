from setuptools import setup, find_packages

setup(
    name="amas",
    version="0.1.0",
    packages=find_packages(include=["."]),
    python_requires=">=3.12",
    install_requires=[
        "ollama",
    ],
) 