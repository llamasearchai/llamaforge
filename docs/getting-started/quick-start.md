# Quick Start Guide

This guide will help you get started with LlamaForge quickly. We'll cover basic usage patterns for both the Python API and command-line interface.

## Using the Python API

### Basic Text Generation

```python
from llamaforge import LlamaForge

# Initialize LlamaForge
forge = LlamaForge()

# List available models
print(forge.list_models())

# Load a model
forge.load_model("llama-2-7b-chat")

# Generate text
response = forge.generate("Explain quantum computing in simple terms")
print(response)
```

### Streaming Text Generation

```python
from llamaforge import LlamaForge

forge = LlamaForge()
forge.load_model("llama-2-7b-chat")

# Stream the response
for chunk in forge.generate_stream("Write a short poem about AI"):
    print(chunk, end="", flush=True)
print()
```

### Chat Interface

```python
from llamaforge import LlamaForge

forge = LlamaForge()
forge.load_model("llama-2-7b-chat")

# Start a chat session
chat_history = []
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    response = forge.chat(user_input, chat_history)
    print(f"AI: {response}")
    
    # Update chat history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})
```

## Using the Command Line Interface

### Basic Text Generation

```bash
llamaforge run "Explain quantum computing in simple terms" --model llama-2-7b-chat
```

### Interactive Chat

```bash
llamaforge chat --model llama-2-7b-chat
```

### Starting the API Server

```bash
llamaforge serve --model llama-2-7b-chat --host 127.0.0.1 --port 8000
```

The API server is compatible with OpenAI's API, so you can use it with existing tools.

### Model Management

List available models:

```bash
llamaforge models list
```

Add a new model:

```bash
llamaforge models add --name "my-llama" --path "/path/to/model.gguf" --backend "llama.cpp"
```

## Configuration

LlamaForge uses a configuration file located at `~/.llamaforge/config.json`. You can edit this file directly or use the Python API:

```python
from llamaforge import LlamaForge

forge = LlamaForge()

# Get configuration value
print(forge.config.get("default_model"))

# Set configuration value
forge.config.set("default_model", "llama-2-7b-chat")
forge.config.save()
```

## Next Steps

- [Model Guide](../guides/models.md): Learn more about working with different models
- [Backends Guide](../guides/backends.md): Understand the different backend options
- [Plugins Guide](../guides/plugins.md): Learn how to extend functionality with plugins
- [API Reference](../api/llamaforge.md): Detailed API documentation 