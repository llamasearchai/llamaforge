# LlamaForge API Reference

The `LlamaForge` class is the main entry point for working with models in the LlamaForge library. It provides an integrated interface for managing models, handling configurations, and running text generation with various backends.

## Initialization

```python
from llamaforge import LlamaForge

# Default initialization
forge = LlamaForge()

# Initialize with a custom configuration path
forge = LlamaForge(config_path="/path/to/custom/config.json")

# Initialize with logging configured
forge = LlamaForge(log_level="DEBUG")
```

Parameters:
- `config_path` (str, optional): Path to the configuration file. Defaults to `~/.llamaforge/config.json`.
- `log_level` (str, optional): Logging level. Defaults to "INFO".

## Properties

### `config`

Access the underlying `Config` instance.

```python
# Access configuration
config = forge.config

# Get a configuration value
default_model = forge.config.get("default_model")
```

### `current_model`

Access the currently loaded model.

```python
model = forge.current_model
print(f"Current model: {model.name}")
```

## Model Management

### `add_model`

Add a model to the configuration.

```python
forge.add_model(
    name="llama-2-7b",
    path="/path/to/llama-2-7b.gguf",
    backend="llama.cpp",
    parameters={
        "n_gpu_layers": 32,
        "n_ctx": 4096
    }
)
```

Parameters:
- `name` (str): Name of the model.
- `path` (str): Path to the model file or model identifier.
- `backend` (str, optional): Backend to use. Defaults to "llama.cpp".
- `parameters` (dict, optional): Model-specific parameters.
- `metadata` (dict, optional): Model metadata.

### `remove_model`

Remove a model from the configuration.

```python
result = forge.remove_model("llama-2-7b")
print(f"Model removed: {result}")
```

Parameters:
- `name` (str): Name of the model to remove.

Returns:
- `True` if the model was removed, `False` otherwise.

### `list_models`

List all available models.

```python
models = forge.list_models()
for model_name in models:
    print(model_name)
```

Returns:
- A list of model names.

### `get_model`

Get a model instance by name.

```python
model = forge.get_model("llama-2-7b")
print(model.path)
```

Parameters:
- `name` (str): Name of the model.

Returns:
- A `Model` instance if the model exists, otherwise `None`.

### `load_model`

Load a model for text generation.

```python
forge.load_model("llama-2-7b")
print(f"Loaded model: {forge.current_model.name}")
```

Parameters:
- `name` (str): Name of the model to load.

### `is_model_loaded`

Check if a model is loaded.

```python
is_loaded = forge.is_model_loaded("llama-2-7b")
print(f"Model loaded: {is_loaded}")
```

Parameters:
- `name` (str, optional): Name of the model to check. If not provided, checks if any model is loaded.

Returns:
- `True` if the model is loaded, `False` otherwise.

## Text Generation

### `generate`

Generate text from a prompt.

```python
# Basic generation
response = forge.generate("Explain quantum computing in simple terms")

# Generation with parameters
response = forge.generate(
    prompt="Explain quantum computing in simple terms",
    parameters={
        "temperature": 0.7,
        "max_tokens": 500
    }
)
```

Parameters:
- `prompt` (str): Input prompt.
- `parameters` (dict, optional): Generation parameters.

Returns:
- The generated text as a string.

### `generate_stream`

Stream generated text from a prompt.

```python
# Basic streaming generation
for chunk in forge.generate_stream("Explain quantum computing in simple terms"):
    print(chunk, end="", flush=True)

# Streaming with parameters
for chunk in forge.generate_stream(
    prompt="Explain quantum computing in simple terms",
    parameters={
        "temperature": 0.7,
        "max_tokens": 500
    }
):
    print(chunk, end="", flush=True)
```

Parameters:
- `prompt` (str): Input prompt.
- `parameters` (dict, optional): Generation parameters.

Returns:
- An iterator yielding chunks of generated text.

## Plugin Management

### `list_plugins`

List all available plugins.

```python
plugins = forge.list_plugins()
print(plugins)  # ['text_formatter', 'text_cleaner', 'calculator', 'benchmark']
```

Returns:
- A list of plugin names.

### `list_enabled_plugins`

List all enabled plugins.

```python
enabled_plugins = forge.list_enabled_plugins()
print(enabled_plugins)  # ['text_formatter', 'text_cleaner']
```

Returns:
- A list of enabled plugin names.

### `enable_plugin`

Enable a plugin.

```python
forge.enable_plugin("calculator")
```

Parameters:
- `name` (str): Name of the plugin to enable.

### `disable_plugin`

Disable a plugin.

```python
forge.disable_plugin("calculator")
```

Parameters:
- `name` (str): Name of the plugin to disable.

### `is_plugin_enabled`

Check if a plugin is enabled.

```python
status = forge.is_plugin_enabled("text_formatter")
print(f"Plugin enabled: {status}")
```

Parameters:
- `name` (str): Name of the plugin to check.

Returns:
- `True` if the plugin is enabled, `False` otherwise.

### `get_plugin_config`

Get configuration for a plugin.

```python
config = forge.get_plugin_config("text_formatter")
print(config)
```

Parameters:
- `name` (str): Name of the plugin.

Returns:
- The plugin configuration as a dictionary.

### `set_plugin_config`

Set configuration for a plugin.

```python
forge.set_plugin_config("text_formatter", {
    "trim_whitespace": True,
    "add_system_instruction": True,
    "system_instruction": "You are a helpful assistant."
})
```

Parameters:
- `name` (str): Name of the plugin.
- `config` (dict): Configuration for the plugin.

### `get_all_plugin_configs`

Get configuration for all plugins.

```python
configs = forge.get_all_plugin_configs()
print(configs)
```

Returns:
- A dictionary mapping plugin names to configurations.

### `run_command`

Run a command plugin.

```python
results = forge.run_command("benchmark", {
    "model": "llama-2-7b",
    "dataset": "alpaca_eval",
    "num_samples": 100
})
print(results)
```

Parameters:
- `name` (str): Name of the command.
- `args` (dict): Arguments for the command.

Returns:
- The result of the command.

## Chat Interface

### `create_chat_interface`

Create a chat interface for interactive conversation.

```python
chat = forge.create_chat_interface(
    model="llama-2-7b",
    system_prompt="You are a helpful assistant."
)

# Start the chat
chat.start()
```

Parameters:
- `model` (str, optional): Name of the model to use. Defaults to the current model or the default model.
- `system_prompt` (str, optional): System prompt to use.

Returns:
- A `ChatInterface` instance.

## Advanced Usage

### Loading Different Backends

```python
# Load a local GGUF model with llama.cpp
forge.add_model(
    name="llama-2-7b",
    path="/path/to/llama-2-7b.gguf",
    backend="llama.cpp",
    parameters={
        "n_gpu_layers": 32,
        "n_ctx": 4096
    }
)

# Load a Hugging Face model
forge.add_model(
    name="mistral-7b",
    path="mistralai/Mistral-7B-Instruct-v0.1",
    backend="huggingface",
    parameters={
        "device": "cuda",
        "torch_dtype": "bfloat16"
    }
)

# Use OpenAI models
forge.add_model(
    name="gpt-4",
    path="gpt-4",
    backend="openai",
    parameters={
        "api_key": "your-api-key"
    }
)
```

### Customizing Generation Parameters

```python
# Load the model
forge.load_model("llama-2-7b")

# Generate with custom parameters
response = forge.generate(
    prompt="Write a story about a robot.",
    parameters={
        "temperature": 0.8,        # Controls randomness
        "top_p": 0.9,              # Nucleus sampling probability
        "top_k": 40,               # Limit vocabulary to top k tokens
        "max_tokens": 1000,        # Maximum length of generated text
        "stop": ["THE END"],       # Stop sequences
        "repeat_penalty": 1.1      # Penalty for repeating tokens
    }
)
```

### Using Plugins for Text Processing

```python
# Enable preprocessing and postprocessing plugins
forge.enable_plugin("text_formatter")
forge.enable_plugin("text_cleaner")

# Configure the text formatter plugin
forge.set_plugin_config("text_formatter", {
    "trim_whitespace": True,
    "add_system_instruction": True,
    "system_instruction": "You are a helpful assistant who responds in the style of William Shakespeare.",
    "format_as_chat": True,
    "chat_format": "llama2"
})

# Configure the text cleaner plugin
forge.set_plugin_config("text_cleaner", {
    "trim_whitespace": True,
    "remove_special_tokens": True,
    "remove_incomplete_sentences": True
})

# Generate text with automatic formatting and cleaning
response = forge.generate("Tell me about artificial intelligence.")
```

### Batch Processing

```python
# Load the model
forge.load_model("llama-2-7b")

# Define a list of prompts
prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "What is reinforcement learning?"
]

# Process prompts in batch
results = []
for prompt in prompts:
    results.append(forge.generate(prompt))

# Print results
for i, result in enumerate(results):
    print(f"Prompt {i+1}: {prompts[i]}")
    print(f"Response: {result}")
    print("---")
```

### Creating an API Server

```python
from llamaforge import LlamaForge
from llamaforge.server import create_api_server

# Initialize LlamaForge
forge = LlamaForge()

# Add some models
forge.add_model(
    name="llama-2-7b",
    path="/path/to/llama-2-7b.gguf",
    backend="llama.cpp"
)

# Create an API server
app = create_api_server(forge)

# Run the server (in a production environment, use a WSGI server)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Error Handling

```python
from llamaforge import LlamaForge
from llamaforge.exceptions import ModelNotFoundError, ModelLoadError

forge = LlamaForge()

try:
    # Try to load a non-existent model
    forge.load_model("non-existent-model")
except ModelNotFoundError:
    print("Model not found in configuration.")
    
    # Add the model
    forge.add_model(
        name="llama-2-7b", 
        path="/path/to/llama-2-7b.gguf"
    )
    
    try:
        # Try to load the model
        forge.load_model("llama-2-7b")
    except ModelLoadError as e:
        print(f"Failed to load model: {e}")
        
        # Provide more information about the error
        print("Please check the model path and backend compatibility.") 