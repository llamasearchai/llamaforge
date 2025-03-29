# Quick Start Guide

This guide will help you get started with LlamaForge, showing how to install the library, set up your first model, and generate text.

## Installation

Install LlamaForge using pip:

```bash
pip install llamaforge
```

For optional dependencies:

```bash
# With llama.cpp support
pip install llamaforge[llama.cpp]

# With Hugging Face support
pip install llamaforge[huggingface]

# With all dependencies
pip install llamaforge[all]
```

## Setting Up Your First Model

### Using a Local GGUF Model

If you have a GGUF model file locally (e.g., from [TheBloke](https://huggingface.co/TheBloke)), you can set it up like this:

```python
from llamaforge import LlamaForge

# Initialize LlamaForge
forge = LlamaForge()

# Add a local GGUF model
forge.add_model(
    name="llama-2-7b",
    path="/path/to/llama-2-7b.gguf",
    backend="llama.cpp",
    parameters={
        "n_gpu_layers": 32,  # Use GPU acceleration if available
        "n_ctx": 4096        # Set context length
    }
)

# Save the configuration
forge.config.save()
```

### Using a Hugging Face Model

To use a model from the Hugging Face Hub:

```python
from llamaforge import LlamaForge

# Initialize LlamaForge
forge = LlamaForge()

# Add a Hugging Face model
forge.add_model(
    name="mistral-7b",
    path="mistralai/Mistral-7B-Instruct-v0.1",
    backend="huggingface",
    parameters={
        "device": "cuda",        # Use GPU if available
        "torch_dtype": "bfloat16"  # Use BF16 precision
    }
)

# Save the configuration
forge.config.save()
```

### Using OpenAI Models

To use OpenAI models:

```python
from llamaforge import LlamaForge

# Initialize LlamaForge
forge = LlamaForge()

# Add an OpenAI model
forge.add_model(
    name="gpt-4",
    path="gpt-4",
    backend="openai",
    parameters={
        "api_key": "your-api-key"  # Your OpenAI API key
    }
)

# Save the configuration
forge.config.save()
```

## Generating Text

Once you've set up a model, you can generate text:

```python
from llamaforge import LlamaForge

# Initialize LlamaForge
forge = LlamaForge()

# Load a model
forge.load_model("llama-2-7b")

# Generate text
response = forge.generate(
    prompt="Explain quantum computing in simple terms",
    parameters={
        "temperature": 0.7,
        "max_tokens": 500
    }
)

print(response)
```

## Streaming Text Generation

For streaming generation (getting output tokens as they're generated):

```python
from llamaforge import LlamaForge

# Initialize LlamaForge
forge = LlamaForge()

# Load a model
forge.load_model("llama-2-7b")

# Stream generated text
for chunk in forge.generate_stream(
    prompt="Write a short story about an AI assistant",
    parameters={
        "temperature": 0.8,
        "max_tokens": 1000
    }
):
    print(chunk, end="", flush=True)
```

## Using the Command Line Interface

LlamaForge provides a convenient command-line interface:

```bash
# List all available models
llamaforge list-models

# Add a model
llamaforge add-model --name llama-2-7b --path /path/to/llama-2-7b.gguf --backend llama.cpp

# Generate text
llamaforge generate --model llama-2-7b "Explain quantum computing in simple terms"

# Generate with parameters
llamaforge generate --model llama-2-7b --temperature 0.7 --max-tokens 500 "Explain quantum computing in simple terms"

# Start a chat session
llamaforge chat --model llama-2-7b

# Start the API server
llamaforge serve --host 127.0.0.1 --port 8000
```

## Using Plugins

LlamaForge includes plugins for preprocessing inputs and postprocessing outputs:

```python
from llamaforge import LlamaForge

# Initialize LlamaForge
forge = LlamaForge()

# Load a model
forge.load_model("llama-2-7b")

# List available plugins
print(forge.list_plugins())

# Enable plugins
forge.enable_plugin("text_formatter")
forge.enable_plugin("text_cleaner")

# Configure the text formatter plugin
forge.set_plugin_config("text_formatter", {
    "trim_whitespace": True,
    "add_system_instruction": True,
    "system_instruction": "You are a helpful assistant.",
    "format_as_chat": True,
    "chat_format": "llama2"
})

# Configure the text cleaner plugin
forge.set_plugin_config("text_cleaner", {
    "trim_whitespace": True,
    "remove_special_tokens": True,
    "remove_incomplete_sentences": True
})

# Generate text (will be processed by the plugins)
response = forge.generate("Tell me about large language models")
print(response)
```

## Creating a Simple Chat Interface

LlamaForge makes it easy to create a chat interface:

```python
from llamaforge import LlamaForge

# Initialize LlamaForge
forge = LlamaForge()

# Load a model
forge.load_model("llama-2-7b")

# Create a chat interface
chat = forge.create_chat_interface(
    system_prompt="You are a helpful assistant who specializes in explaining complex topics in simple terms."
)

# Start the chat
chat.start()
```

## Running an API Server

You can run an API server to expose your models via a REST API:

```python
from llamaforge import LlamaForge
from llamaforge.server import create_api_server

# Initialize LlamaForge
forge = LlamaForge()

# Add a model
forge.add_model(
    name="llama-2-7b",
    path="/path/to/llama-2-7b.gguf",
    backend="llama.cpp"
)

# Create an API server
app = create_api_server(forge)

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

The API server provides endpoints compatible with OpenAI's API, so you can use it with existing tools and libraries.

## Next Steps

Now that you're familiar with the basics of LlamaForge, you can:

- Check the [API Reference](../../api/llamaforge.md) for detailed documentation
- Learn about [model management](../models.md)
- Explore [backends](../../api/backends.md) and their configurations
- Learn about the [plugin system](../../api/plugins.md) and how to create custom plugins
- Set up [configuration management](../configuration.md) for your projects

Happy coding with LlamaForge! 