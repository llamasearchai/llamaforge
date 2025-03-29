# Working with Models

In LlamaForge, models are the core component for generating text. This guide explains how to manage and use different language models.

## Models Overview

LlamaForge supports various types of models through different backends:

- **Local models** via llama.cpp (GGUF format)
- **Hugging Face models** (local or remote)
- **OpenAI models** (via API)

## Listing Available Models

You can view all configured models:

```python
from llamaforge import LlamaForge

forge = LlamaForge()
models = forge.list_models()
print(models)
```

From the command line:

```bash
llamaforge models list
```

## Adding Models

### Via Python API

```python
from llamaforge import LlamaForge

forge = LlamaForge()

# Add a llama.cpp model
forge.add_model(
    name="llama-2-7b", 
    path="/path/to/llama-2-7b.gguf", 
    backend="llama.cpp",
    parameters={
        "context_length": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    }
)

# Add a Hugging Face model
forge.add_model(
    name="mistral-7b", 
    path="mistralai/Mistral-7B-Instruct-v0.1", 
    backend="huggingface",
    parameters={
        "device": "cuda",
        "torch_dtype": "bfloat16"
    }
)

# Add an OpenAI model
forge.add_model(
    name="gpt-4", 
    path="gpt-4", 
    backend="openai",
    parameters={
        "api_key": "your-api-key"
    }
)

# Save configuration
forge.config.save()
```

### Via Command Line

```bash
# Add a llama.cpp model
llamaforge models add --name "llama-2-7b" --path "/path/to/llama-2-7b.gguf" --backend "llama.cpp"

# Add a Hugging Face model
llamaforge models add --name "mistral-7b" --path "mistralai/Mistral-7B-Instruct-v0.1" --backend "huggingface"

# Add an OpenAI model
llamaforge models add --name "gpt-4" --path "gpt-4" --backend "openai" --param "api_key=your-api-key"
```

## Loading Models

Before generating text, you need to load a model:

```python
from llamaforge import LlamaForge

forge = LlamaForge()
forge.load_model("llama-2-7b")
```

You can also provide parameters to override the default configuration:

```python
forge.load_model("llama-2-7b", parameters={
    "temperature": 0.5,
    "top_p": 0.95
})
```

## Model Parameters

Different backends support different parameters. Here are some common ones:

### Common Parameters

- `temperature`: Controls randomness (higher = more random)
- `top_p`: Nucleus sampling probability
- `top_k`: Limits vocabulary to top k tokens
- `max_tokens`: Maximum length of generated text

### llama.cpp Specific Parameters

- `n_gpu_layers`: Number of layers to offload to GPU
- `n_ctx`: Context length
- `n_batch`: Batch size for prompt evaluation
- `f16_kv`: Use half-precision for key/value cache

### Hugging Face Specific Parameters

- `device`: Device to use (cuda, cpu, etc.)
- `torch_dtype`: Data type for PyTorch
- `trust_remote_code`: Whether to trust remote code from Hugging Face

### OpenAI Specific Parameters

- `api_key`: Your OpenAI API key
- `api_base`: Custom API endpoint
- `organization`: Your OpenAI organization ID

## Removing Models

```python
from llamaforge import LlamaForge

forge = LlamaForge()
forge.remove_model("llama-2-7b")
forge.config.save()
```

From the command line:

```bash
llamaforge models remove --name "llama-2-7b"
```

## Best Practices

- **Set a default model**: Use `forge.config.set("default_model", "your-model")` to define a default model.
- **Optimize for your hardware**: Adjust parameters based on your available memory and computational resources.
- **Use quantized models**: For local usage, quantized GGUF models provide a good balance of quality and resource usage.
- **Stream responses**: For interactive applications, use the streaming API to provide real-time feedback.

## Next Steps

- Check out the [Backends Guide](backends.md) to learn more about the different backends
- Learn about the [Plugins Guide](plugins.md) to extend LlamaForge's capabilities 