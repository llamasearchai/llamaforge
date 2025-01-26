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
    name="llamaforge-llamasearch",
    version=version,
    description="Ultimate Language Model Command Interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LlamaSearch AI",
    author_email="nikjois@llamasearch.ai",
    url="https://llamasearch.ai",
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
    package_dir={"": "src"},
    packages=find_packages(where="src"),
) 
# Updated in commit 5 - 2025-04-04 17:04:00

# Updated in commit 13 - 2025-04-04 17:04:07

# Updated in commit 21 - 2025-04-04 17:04:11

# Updated in commit 29 - 2025-04-04 17:04:17

# Updated in commit 5 - 2025-04-05 14:24:48

# Updated in commit 13 - 2025-04-05 14:24:49

# Updated in commit 21 - 2025-04-05 14:24:49

# Updated in commit 29 - 2025-04-05 14:24:49

# Updated in commit 5 - 2025-04-05 15:00:42

# Updated in commit 13 - 2025-04-05 15:00:42

# Updated in commit 21 - 2025-04-05 15:00:43

# Updated in commit 29 - 2025-04-05 15:00:43

# Updated in commit 5 - 2025-04-05 15:10:25

# Updated in commit 13 - 2025-04-05 15:10:26
