# Backends Guide

LlamaForge supports multiple backends to run language models. Each backend has its own advantages and use cases. This guide explains how to work with the different backends.

## Available Backends

LlamaForge currently supports the following backends:

1. **llama.cpp**: For running local models in GGUF format
2. **Hugging Face**: For running models from the Hugging Face Hub or local Hugging Face models
3. **OpenAI**: For using OpenAI's API models

## llama.cpp Backend

The llama.cpp backend is ideal for running models locally on your machine.

### Setup

To use the llama.cpp backend, you need to install the llama-cpp-python package:

```bash
pip install llamaforge[llama-cpp]
```

For GPU acceleration:

```bash
# For CUDA
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade

# For Metal (MacOS)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --upgrade
```

### Usage

```python
from llamaforge import LlamaForge

forge = LlamaForge()
forge.add_model(
    name="llama-2-7b", 
    path="/path/to/llama-2-7b.gguf", 
    backend="llama.cpp",
    parameters={
        "n_gpu_layers": 32,  # Number of layers to offload to GPU
        "n_ctx": 4096,       # Context window size
        "temperature": 0.7
    }
)
forge.config.save()

forge.load_model("llama-2-7b")
response = forge.generate("Explain quantum computing in simple terms")
print(response)
```

### Key Parameters

- `n_gpu_layers`: Number of layers to offload to GPU (0 for CPU only)
- `n_ctx`: Context length
- `n_batch`: Batch size for prompt evaluation
- `f16_kv`: Use half-precision for key/value cache

## Hugging Face Backend

The Hugging Face backend allows you to use models from the Hugging Face Hub or local Hugging Face models.

### Setup

```bash
pip install llamaforge[huggingface]
```

### Usage

```python
from llamaforge import LlamaForge

forge = LlamaForge()
forge.add_model(
    name="mistral-7b", 
    path="mistralai/Mistral-7B-Instruct-v0.1", 
    backend="huggingface",
    parameters={
        "device": "cuda",
        "torch_dtype": "bfloat16",
        "trust_remote_code": True
    }
)
forge.config.save()

forge.load_model("mistral-7b")
response = forge.generate("Explain quantum computing in simple terms")
print(response)
```

### Key Parameters

- `device`: Device to use (cuda, cpu, etc.)
- `torch_dtype`: Data type for PyTorch (bfloat16, float16, etc.)
- `trust_remote_code`: Whether to trust remote code from Hugging Face
- `cache_dir`: Directory to cache downloaded models

## OpenAI Backend

The OpenAI backend allows you to use OpenAI's API models.

### Setup

```bash
pip install llamaforge[openai]
```

### Usage

```python
from llamaforge import LlamaForge

forge = LlamaForge()
forge.add_model(
    name="gpt-4", 
    path="gpt-4", 
    backend="openai",
    parameters={
        "api_key": "your-api-key",
        "temperature": 0.7,
        "max_tokens": 1000
    }
)
forge.config.save()

forge.load_model("gpt-4")
response = forge.generate("Explain quantum computing in simple terms")
print(response)
```

### Key Parameters

- `api_key`: Your OpenAI API key
- `api_base`: Custom API endpoint
- `organization`: Your OpenAI organization ID
- `temperature`: Controls randomness
- `max_tokens`: Maximum length of generated text

## Selecting a Backend

When choosing a backend, consider:

1. **Performance**: llama.cpp is optimized for local execution
2. **Flexibility**: Hugging Face offers a wide range of models
3. **Quality**: OpenAI's models often provide state-of-the-art results
4. **Cost**: Local models have no usage costs after download
5. **Privacy**: Local execution keeps your data private

## Creating Custom Backends

You can create custom backends by extending the `BaseBackend` class:

```python
from llamaforge.backends.base import BaseBackend

class MyCustomBackend(BaseBackend):
    def load_model(self, model_path, parameters=None):
        # Implementation here
        pass

    def generate(self, prompt, parameters=None):
        # Implementation here
        pass

    def generate_stream(self, prompt, parameters=None):
        # Implementation here
        yield "Streaming response"

    def is_model_loaded(self):
        # Implementation here
        return True
```

Then register your custom backend:

```python
from llamaforge.backends import register_backend

register_backend("my_custom", MyCustomBackend)
```

## Next Steps

- Learn about the [Plugins Guide](plugins.md) to extend LlamaForge's capabilities
- See the [API Reference](../api/backends.md) for detailed information on the backends API 