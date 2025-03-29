# Installation Guide

There are several ways to install LlamaForge depending on your needs and environment.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Using pip

The simplest way to install LlamaForge is using pip:

```bash
pip install llamaforge
```

For the latest development version, you can install directly from GitHub:

```bash
pip install git+https://github.com/llamasearch/llamaforge.git
```

## From Source

To install from source:

```bash
git clone https://github.com/llamasearch/llamaforge.git
cd llamaforge
pip install -e .
```

This installs LlamaForge in development mode, allowing you to modify the code and see changes immediately.

## Optional Dependencies

Depending on which backends you want to use, you may need to install additional dependencies:

### llama.cpp Backend

```bash
pip install llamaforge[llama-cpp]
```

### Hugging Face Backend

```bash
pip install llamaforge[huggingface]
```

### All Backends

```bash
pip install llamaforge[all]
```

## Verifying Installation

To verify that LlamaForge is installed correctly, run:

```bash
python -c "import llamaforge; print(llamaforge.__version__)"
```

This should print the version number of LlamaForge.

## Next Steps

Once you have LlamaForge installed, check out the [Quick Start Guide](quick-start.md) to learn how to use it. 