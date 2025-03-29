#!/usr/bin/env python3
"""
Setup script for LlamaForge.
"""

import os
from setuptools import setup, find_packages

# Read version from version.py
version = {}
with open(os.path.join("llamaforge", "version.py")) as f:
    exec(f.read(), version)

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if not line.startswith("#") and line.strip()]

# Read long description from README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Optional dependencies
extras_require = {
    "llama": ["llama-cpp-python>=0.1.0"],
    "mlx": ["mlx>=0.0.3"],
    "transformers": ["transformers>=4.20.0", "torch>=1.10.0"],
    "all": [
        "llama-cpp-python>=0.1.0",
        "mlx>=0.0.3",
        "transformers>=4.20.0",
        "torch>=1.10.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.10.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "build>=0.10.0",
        "twine>=4.0.0",
    ],
}

setup(
    name="llamaforge",
    version=version["__version__"],
    description=version["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=version["__author__"],
    author_email=version["__email__"],
    url=version["__url__"],
    packages=find_packages(),
    package_data={
        "llamaforge": ["py.typed"],
    },
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "llamaforge=llamaforge.main:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm, ai, language model, cli, inference",
) 