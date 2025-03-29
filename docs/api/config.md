# Config API Reference

The `Config` class in LlamaForge provides a centralized configuration management system. It handles loading, saving, and accessing configuration settings for models, backends, and other components.

## Initialization

```python
from llamaforge.config import Config

# Default initialization (uses ~/.llamaforge/config.json)
config = Config()

# Custom configuration path
config = Config(config_path="/path/to/custom/config.json")
```

Parameters:
- `config_path` (str, optional): Path to the configuration file. Defaults to `~/.llamaforge/config.json`.

## Properties

### `config_path`

Returns the path to the configuration file.

```python
path = config.config_path  # Returns the full path to the configuration file
```

### `config`

Returns the entire configuration dictionary.

```python
all_config = config.config  # Returns the entire configuration as a dictionary
```

## Methods

### `get`

Retrieves a configuration value by key. Supports nested keys using dot notation.

```python
# Get a top-level key
models = config.get("models")

# Get a nested key
model_path = config.get("models.llama-2-7b.path")

# Provide a default value if the key doesn't exist
api_key = config.get("api_keys.openai", default="no-key-set")
```

Parameters:
- `key` (str): The configuration key to retrieve. Use dot notation for nested keys.
- `default` (any, optional): Default value to return if the key doesn't exist.

Returns:
- The configuration value if the key exists, otherwise the default value.

### `set`

Sets a configuration value by key. Supports nested keys using dot notation.

```python
# Set a top-level key
config.set("default_model", "llama-2-7b")

# Set a nested key
config.set("models.llama-2-7b.parameters.n_gpu_layers", 32)
```

Parameters:
- `key` (str): The configuration key to set. Use dot notation for nested keys.
- `value` (any): The value to set.

### `add_model`

Adds a model to the configuration.

```python
config.add_model(
    name="llama-2-7b",
    path="/path/to/llama-2-7b.gguf",
    backend="llama.cpp",
    parameters={
        "n_gpu_layers": 32,
        "n_ctx": 4096
    },
    metadata={
        "description": "Llama 2 7B model",
        "version": "2",
        "license": "llama2"
    }
)
```

Parameters:
- `name` (str): The name of the model.
- `path` (str): The path to the model file.
- `backend` (str, optional): The backend to use. Defaults to "llama.cpp".
- `parameters` (dict, optional): Additional model parameters.
- `metadata` (dict, optional): Model metadata.

Returns:
- The model configuration as a dictionary.

### `remove_model`

Removes a model from the configuration.

```python
result = config.remove_model("llama-2-7b")  # True if the model was removed
```

Parameters:
- `name` (str): The name of the model to remove.

Returns:
- `True` if the model was removed, `False` otherwise.

### `get_model`

Retrieves a model configuration by name.

```python
model_config = config.get_model("llama-2-7b")
```

Parameters:
- `name` (str): The name of the model.

Returns:
- The model configuration as a dictionary if the model exists, otherwise `None`.

### `list_models`

Lists all models in the configuration.

```python
models = config.list_models()  # Returns a list of model names
```

Returns:
- A list of model names.

### `save`

Saves the current configuration to the configuration file.

```python
config.save()  # Saves the configuration to the default path

# Or save to a different path
config.save("/path/to/new/config.json")
```

Parameters:
- `path` (str, optional): Path to save the configuration to. Defaults to the current configuration path.

### `load`

Loads the configuration from the configuration file.

```python
config.load()  # Loads the configuration from the default path

# Or load from a different path
config.load("/path/to/other/config.json")
```

Parameters:
- `path` (str, optional): Path to load the configuration from. Defaults to the current configuration path.

## Usage in LlamaForge

The `Config` class is used internally by the `LlamaForge` class to manage configurations:

```python
from llamaforge import LlamaForge

forge = LlamaForge()

# Access the config directly
config = forge.config

# Add a model (uses config.add_model internally)
forge.add_model(
    name="llama-2-7b",
    path="/path/to/llama-2-7b.gguf",
    backend="llama.cpp"
)

# Get configuration settings
default_model = forge.config.get("default_model")
```

## Advanced Usage

### Creating a Config From Scratch

```python
from llamaforge.config import Config

# Create a new configuration
config = Config(config_path="/path/to/new/config.json")

# Initialize basic structure
config.set("models", {})
config.set("default_model", "")
config.set("api_keys", {})

# Add models
config.add_model(
    name="llama-2-7b",
    path="/path/to/llama-2-7b.gguf"
)

# Save the configuration
config.save()
```

### Merging Configurations

```python
import json
from llamaforge.config import Config

# Load an existing configuration
config = Config()

# Load another configuration from a file
with open("/path/to/other/config.json", "r") as f:
    other_config = json.load(f)

# Merge configurations
for model_name, model_config in other_config.get("models", {}).items():
    if model_name not in config.get("models", {}):
        config.add_model(
            name=model_name,
            path=model_config.get("path", ""),
            backend=model_config.get("backend", "llama.cpp"),
            parameters=model_config.get("parameters", {}),
            metadata=model_config.get("metadata", {})
        )

# Save the merged configuration
config.save()
```

### Environment Variables

LlamaForge respects environment variables for configuration overrides:

```python
import os
from llamaforge.config import Config

# Set environment variables
os.environ["LLAMAFORGE_CONFIG_PATH"] = "/path/to/custom/config.json"
os.environ["LLAMAFORGE_DEFAULT_MODEL"] = "llama-2-7b"
os.environ["LLAMAFORGE_OPENAI_API_KEY"] = "your-api-key"

# Create a config instance (will use the environment variable for the path)
config = Config()

# Environment variables override file-based configuration
default_model = config.get("default_model")  # Will be "llama-2-7b"
api_key = config.get("api_keys.openai")  # Will be "your-api-key"
```

## Configuration File Structure

The LlamaForge configuration file follows this structure:

```json
{
  "models": {
    "llama-2-7b": {
      "path": "/path/to/llama-2-7b.gguf",
      "backend": "llama.cpp",
      "parameters": {
        "n_gpu_layers": 32,
        "n_ctx": 4096
      },
      "metadata": {
        "description": "Llama 2 7B model",
        "version": "2",
        "license": "llama2"
      }
    },
    "mistral-7b": {
      "path": "mistralai/Mistral-7B-Instruct-v0.1",
      "backend": "huggingface",
      "parameters": {
        "device": "cuda",
        "torch_dtype": "bfloat16"
      }
    }
  },
  "default_model": "llama-2-7b",
  "api_keys": {
    "openai": "your-openai-api-key"
  },
  "plugins": {
    "enabled": [
      "text_formatter",
      "text_cleaner",
      "calculator"
    ]
  },
  "logging": {
    "level": "INFO",
    "file": "~/.llamaforge/llamaforge.log"
  }
} 