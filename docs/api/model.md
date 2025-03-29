# Model API Reference

The `Model` class represents a language model in LlamaForge. It encapsulates all the information needed to load and use a model.

## Initialization

```python
from llamaforge.model import Model

# Create a model
model = Model(
    name="llama-2-7b",
    path="/path/to/llama-2-7b.gguf",
    config={
        "backend": "llama.cpp",
        "parameters": {
            "temperature": 0.7,
            "top_p": 0.9
        },
        "metadata": {
            "description": "Llama 2 7B chat model",
            "version": "2.0"
        }
    }
)
```

Parameters:
- `name` (str): Name of the model
- `path` (str): Path to the model file or model identifier
- `config` (dict): Configuration dictionary for the model

## Properties

### Name

Get the name of the model:

```python
name = model.name  # "llama-2-7b"
```

### Path

Get the path of the model:

```python
path = model.path  # "/path/to/llama-2-7b.gguf"
```

### Config

Get the configuration dictionary:

```python
config = model.config
```

### Backend

Get the backend name:

```python
backend = model.backend  # "llama.cpp"
```

### Metadata

Get the metadata dictionary:

```python
metadata = model.metadata
```

## Methods

### Get Parameter

Get a parameter value from the model's configuration:

```python
# Get a top-level parameter
temperature = model.get_param("temperature")

# Get a nested parameter
value = model.get_param("nested.parameter")

# Get with a default value
default_value = model.get_param("nonexistent", "default")
```

Parameters:
- `key` (str): Parameter key (use dots for nested keys)
- `default` (any, optional): Default value to return if the key doesn't exist

### Get Parameters

Get parameters from the model's configuration:

```python
# Get all parameters
all_params = model.get_parameters()

# Get a specific parameter
temperature = model.get_parameters("temperature")

# Get with a default value
default_value = model.get_parameters("nonexistent", "default")
```

Parameters:
- `key` (str, optional): Parameter key. If not provided, returns all parameters.
- `default` (any, optional): Default value to return if the key doesn't exist

### Get Metadata

Get metadata from the model's configuration:

```python
# Get a specific metadata value
description = model.get_metadata("description")

# Get with a default value
version = model.get_metadata("version", "1.0")
```

Parameters:
- `key` (str): Metadata key
- `default` (any, optional): Default value to return if the key doesn't exist

### Is Local

Check if the model is a local file:

```python
if model.is_local():
    print("Model is a local file")
else:
    print("Model is remote or not a file path")
```

### Update Config

Update the model's configuration:

```python
model.update_config({
    "parameters": {
        "temperature": 0.5
    }
})
```

Parameters:
- `config` (dict): New configuration to merge with the existing one

### To Dict

Convert the model to a dictionary:

```python
model_dict = model.to_dict()
```

Returns a dictionary representation of the model.

## Usage in LlamaForge

The `Model` class is typically used by the `LlamaForge` class rather than directly by users:

```python
from llamaforge import LlamaForge

forge = LlamaForge()

# Add a model (creates a Model instance internally)
forge.add_model(
    name="llama-2-7b",
    path="/path/to/llama-2-7b.gguf",
    backend="llama.cpp",
    parameters={
        "temperature": 0.7
    }
)

# Load the model
forge.load_model("llama-2-7b")
```

## Advanced Usage

### Creating a Model from JSON

```python
import json
from llamaforge.model import Model

# Load model configuration from a JSON file
with open("model_config.json", "r") as f:
    config = json.load(f)

# Create a model from the configuration
model = Model(
    name=config["name"],
    path=config["path"],
    config=config
)
```

### Serializing a Model

```python
import json
from llamaforge.model import Model

# Create a model
model = Model(
    name="llama-2-7b",
    path="/path/to/llama-2-7b.gguf",
    config={
        "backend": "llama.cpp",
        "parameters": {
            "temperature": 0.7
        }
    }
)

# Convert the model to a dictionary
model_dict = model.to_dict()

# Save the model configuration to a JSON file
with open("model_config.json", "w") as f:
    json.dump(model_dict, f, indent=2)
```

## Custom Model Subclasses

You can extend the `Model` class for specific backends:

```python
from llamaforge.model import Model

class LlamaCppModel(Model):
    """A model specific to the llama.cpp backend."""
    
    def __init__(self, name, path, config=None):
        super().__init__(name, path, config)
        self.config["backend"] = "llama.cpp"
    
    def get_n_gpu_layers(self):
        """Get the number of GPU layers."""
        return self.get_parameters("n_gpu_layers", 0)
    
    def set_n_gpu_layers(self, n_layers):
        """Set the number of GPU layers."""
        params = self.get_parameters()
        params["n_gpu_layers"] = n_layers
``` 