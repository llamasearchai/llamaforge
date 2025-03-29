# Backends API Reference

LlamaForge backends provide the interface for different model implementations. This reference explains how to use and extend LlamaForge backends.

## Available Backends

LlamaForge includes several built-in backends:

- `llama.cpp`: For running local models in GGUF format
- `huggingface`: For running models from the Hugging Face Hub or local Hugging Face models
- `openai`: For using OpenAI's API models

## BaseBackend Class

All backend implementations inherit from the `BaseBackend` abstract class, which defines the required interface.

```python
from llamaforge.backends.base import BaseBackend

class BaseBackend:
    """Base class for all backends."""
    
    def load_model(self, model_path, parameters=None):
        """Load a model from the given path with the given parameters."""
        raise NotImplementedError
    
    def generate(self, prompt, parameters=None):
        """Generate text from the given prompt with the given parameters."""
        raise NotImplementedError
    
    def generate_stream(self, prompt, parameters=None):
        """Stream generated text from the given prompt with the given parameters."""
        raise NotImplementedError
    
    def is_model_loaded(self):
        """Check if a model is loaded."""
        raise NotImplementedError
```

## Using Backends

Backends are typically used through the `LlamaForge` class rather than directly:

```python
from llamaforge import LlamaForge

forge = LlamaForge()

# Add a model with a specific backend
forge.add_model(
    name="llama-2-7b",
    path="/path/to/llama-2-7b.gguf",
    backend="llama.cpp",
    parameters={
        "n_gpu_layers": 32
    }
)

# Load the model (uses the specified backend)
forge.load_model("llama-2-7b")

# Generate text (uses the loaded backend)
response = forge.generate("Explain quantum computing in simple terms")
```

If you need to access backends directly:

```python
from llamaforge.backends import get_backend

# Get a backend class
llama_cpp_backend_class = get_backend("llama.cpp")

# Instantiate the backend
backend = llama_cpp_backend_class()

# Load a model
backend.load_model("/path/to/llama-2-7b.gguf", parameters={
    "n_gpu_layers": 32,
    "n_ctx": 4096
})

# Generate text
response = backend.generate("Explain quantum computing in simple terms", parameters={
    "temperature": 0.7,
    "max_tokens": 500
})
```

## LlamaCppBackend

The `LlamaCppBackend` class provides integration with the [llama.cpp](https://github.com/ggerganov/llama.cpp) library through its Python bindings.

### Load Model

```python
from llamaforge.backends.llama_cpp import LlamaCppBackend

backend = LlamaCppBackend()
backend.load_model(
    model_path="/path/to/llama-2-7b.gguf",
    parameters={
        "n_gpu_layers": 32,
        "n_ctx": 4096,
        "f16_kv": True
    }
)
```

Parameters:
- `model_path` (str): Path to the model file
- `parameters` (dict, optional): Model parameters
  - `n_gpu_layers` (int): Number of layers to offload to GPU
  - `n_ctx` (int): Context length
  - `n_batch` (int): Batch size for prompt evaluation
  - `f16_kv` (bool): Use half-precision for key/value cache

### Generate

```python
response = backend.generate(
    prompt="Explain quantum computing in simple terms",
    parameters={
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 500
    }
)
```

Parameters:
- `prompt` (str): Input prompt
- `parameters` (dict, optional): Generation parameters
  - `temperature` (float): Controls randomness
  - `top_p` (float): Nucleus sampling probability
  - `top_k` (int): Limits vocabulary to top k tokens
  - `max_tokens` (int): Maximum length of generated text
  - `stop` (list): List of strings that stop generation when encountered

### Generate Stream

```python
for chunk in backend.generate_stream(
    prompt="Explain quantum computing in simple terms",
    parameters={
        "temperature": 0.7,
        "max_tokens": 500
    }
):
    print(chunk, end="", flush=True)
```

Parameters are the same as for `generate`.

## HuggingFaceBackend

The `HuggingFaceBackend` class provides integration with the [Hugging Face Transformers](https://huggingface.co/transformers/) library.

### Load Model

```python
from llamaforge.backends.huggingface import HuggingFaceBackend

backend = HuggingFaceBackend()
backend.load_model(
    model_path="mistralai/Mistral-7B-Instruct-v0.1",
    parameters={
        "device": "cuda",
        "torch_dtype": "bfloat16",
        "trust_remote_code": True
    }
)
```

Parameters:
- `model_path` (str): Path to the model file or model identifier
- `parameters` (dict, optional): Model parameters
  - `device` (str): Device to use (cuda, cpu, etc.)
  - `torch_dtype` (str): Data type for PyTorch (bfloat16, float16, etc.)
  - `trust_remote_code` (bool): Whether to trust remote code from Hugging Face
  - `cache_dir` (str): Directory to cache downloaded models

### Generate

```python
response = backend.generate(
    prompt="Explain quantum computing in simple terms",
    parameters={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 500
    }
)
```

Parameters:
- `prompt` (str): Input prompt
- `parameters` (dict, optional): Generation parameters
  - `temperature` (float): Controls randomness
  - `top_p` (float): Nucleus sampling probability
  - `max_new_tokens` (int): Maximum length of generated text
  - `do_sample` (bool): Whether to use sampling

### Generate Stream

```python
for chunk in backend.generate_stream(
    prompt="Explain quantum computing in simple terms",
    parameters={
        "temperature": 0.7,
        "max_new_tokens": 500
    }
):
    print(chunk, end="", flush=True)
```

Parameters are the same as for `generate`.

## OpenAIBackend

The `OpenAIBackend` class provides integration with the [OpenAI API](https://platform.openai.com/docs/api-reference).

### Load Model

```python
from llamaforge.backends.openai_api import OpenAIBackend

backend = OpenAIBackend()
backend.load_model(
    model_path="gpt-4",
    parameters={
        "api_key": "your-api-key",
        "api_base": "https://api.openai.com/v1",
        "organization": "your-org-id"
    }
)
```

Parameters:
- `model_path` (str): OpenAI model name
- `parameters` (dict, optional): Model parameters
  - `api_key` (str): Your OpenAI API key
  - `api_base` (str): Custom API endpoint
  - `organization` (str): Your OpenAI organization ID

### Generate

```python
response = backend.generate(
    prompt="Explain quantum computing in simple terms",
    parameters={
        "temperature": 0.7,
        "max_tokens": 500,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    }
)
```

Parameters:
- `prompt` (str): Input prompt
- `parameters` (dict, optional): Generation parameters
  - `temperature` (float): Controls randomness
  - `max_tokens` (int): Maximum length of generated text
  - `presence_penalty` (float): Penalty for token presence
  - `frequency_penalty` (float): Penalty for token frequency
  - `stop` (list): List of strings that stop generation when encountered

### Generate Stream

```python
for chunk in backend.generate_stream(
    prompt="Explain quantum computing in simple terms",
    parameters={
        "temperature": 0.7,
        "max_tokens": 500
    }
):
    print(chunk, end="", flush=True)
```

Parameters are the same as for `generate`.

## Creating Custom Backends

You can create custom backends by extending the `BaseBackend` class:

```python
from llamaforge.backends.base import BaseBackend

class MyCustomBackend(BaseBackend):
    """A custom backend implementation."""
    
    def __init__(self):
        super().__init__()
        self.model = None
    
    def load_model(self, model_path, parameters=None):
        """Load a model from the given path with the given parameters."""
        parameters = parameters or {}
        # Implement model loading logic
        self.model = load_your_model(model_path, **parameters)
    
    def generate(self, prompt, parameters=None):
        """Generate text from the given prompt with the given parameters."""
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        parameters = parameters or {}
        # Implement text generation logic
        return your_generation_function(self.model, prompt, **parameters)
    
    def generate_stream(self, prompt, parameters=None):
        """Stream generated text from the given prompt with the given parameters."""
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        parameters = parameters or {}
        # Implement streaming text generation logic
        for chunk in your_streaming_function(self.model, prompt, **parameters):
            yield chunk
    
    def is_model_loaded(self):
        """Check if a model is loaded."""
        return self.model is not None
```

Then register your backend:

```python
from llamaforge.backends import register_backend

register_backend("my_custom", MyCustomBackend)
```

Now you can use your custom backend:

```python
from llamaforge import LlamaForge

forge = LlamaForge()
forge.add_model(
    name="my-model",
    path="/path/to/my-model",
    backend="my_custom",
    parameters={
        "custom_param": "value"
    }
)
```

## Backend Registry

LlamaForge maintains a registry of available backends. You can access and modify this registry:

```python
from llamaforge.backends import register_backend, get_backend, list_backends

# Register a backend
register_backend("my_custom", MyCustomBackend)

# Get a backend class
backend_class = get_backend("my_custom")

# List all available backends
available_backends = list_backends()  # ["llama.cpp", "huggingface", "openai", "my_custom"]
``` 