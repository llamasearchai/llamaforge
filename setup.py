import os
from setuptools import setup, find_packages

# Read version from __init__.py
with open(os.path.join("llamaforge", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"\'')
            break
    else:
        version = "0.0.1"

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="llamaforge",
    version=version,
    description="Ultimate Language Model Command Interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LlamaSearch AI",
    author_email="info@llamasearch.ai",
    url="https://github.com/llamasearch/llamaforge",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "typing-extensions>=4.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "llama.cpp": ["llama-cpp-python>=0.1.70"],
        "huggingface": ["transformers>=4.25.0", "torch>=1.13.1"],
        "openai": ["openai>=0.27.0"],
        "server": ["fastapi>=0.95.0", "uvicorn>=0.21.0", "pydantic>=1.10.7"],
        "dev": [
            "pytest>=7.3.1",
            "black>=23.3.0",
            "isort>=5.12.0",
            "mypy>=1.2.0",
        ],
        "all": [
            "llama-cpp-python>=0.1.70",
            "transformers>=4.25.0",
            "torch>=1.13.1",
            "openai>=0.27.0",
            "fastapi>=0.95.0",
            "uvicorn>=0.21.0",
            "pydantic>=1.10.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "llamaforge=llamaforge.main:main",
        ],
    },
) 