# LlamaForge

Welcome to the official documentation for LlamaForge, a powerful and flexible framework for working with large language models.

## What is LlamaForge?

LlamaForge is a Python library and command-line tool that simplifies working with various large language models (LLMs). It provides a unified interface for:

- Managing multiple models across different backends
- Text generation with various parameters
- Streaming text generation
- Preprocessing and postprocessing text
- Adding tools for advanced model capabilities
- Creating custom plugins
- Running an API server for model inference

LlamaForge supports multiple backends including:

- **llama.cpp**: For running GGUF models locally
- **Hugging Face**: For using models from the Hugging Face Hub
- **OpenAI API**: For accessing OpenAI's models

## Quick Start

### Installation

```bash
pip install llamaforge
```

### Basic Usage

```python
from llamaforge import LlamaForge

# Initialize LlamaForge
forge = LlamaForge()

# Add a model
forge.add_model(
    name="llama-2-7b",
    path="/path/to/llama-2-7b.gguf",
    backend="llama.cpp",
    parameters={
        "n_gpu_layers": 32
    }
)

# Load the model
forge.load_model("llama-2-7b")

# Generate text
response = forge.generate("Explain quantum computing in simple terms")
print(response)

# Stream text
for chunk in forge.generate_stream("Write a short story about AI"):
    print(chunk, end="", flush=True)
```

### Command Line Interface

LlamaForge also provides a command-line interface:

```bash
# Add a model
llamaforge add-model --name llama-2-7b --path /path/to/llama-2-7b.gguf

# Generate text
llamaforge generate --model llama-2-7b "Explain quantum computing in simple terms"

# Start the API server
llamaforge serve --host 127.0.0.1 --port 8000
```

## Features

### Multi-Backend Support

LlamaForge provides a unified interface for working with models from different backends:

```python
# Add a local GGUF model
forge.add_model(
    name="llama-2-7b",
    path="/path/to/llama-2-7b.gguf",
    backend="llama.cpp"
)

# Add a Hugging Face model
forge.add_model(
    name="mistral-7b",
    path="mistralai/Mistral-7B-Instruct-v0.1",
    backend="huggingface"
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
```

### Plugin System

LlamaForge includes a powerful plugin system that allows you to extend its functionality:

```python
# Enable preprocessing and postprocessing plugins
forge.enable_plugin("text_formatter")
forge.enable_plugin("text_cleaner")

# Configure plugins
forge.set_plugin_config("text_formatter", {
    "add_system_instruction": True,
    "system_instruction": "You are a helpful assistant."
})

# Generate text with preprocessing and postprocessing
response = forge.generate("Explain quantum computing")
```

### API Server

LlamaForge can run as an API server:

```python
from llamaforge import LlamaForge
from llamaforge.server import create_api_server

# Initialize LlamaForge
forge = LlamaForge()

# Add models
forge.add_model("llama-2-7b", "/path/to/llama-2-7b.gguf")

# Create and run the API server
app = create_api_server(forge)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Project Status

LlamaForge is under active development. While it's already useful for many tasks, we're continuously adding new features and improvements.

## Getting Help

- Check the [API Reference](api/llamaforge.md) for detailed documentation
- Follow the [tutorials](guides/getting_started/quick_start.md) to learn how to use LlamaForge
- Join our [GitHub Discussions](https://github.com/llamasearch/llamaforge/discussions) for questions and support

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing/development.md) for details on how to get started.

## License

LlamaForge is licensed under the MIT License. See the [LICENSE](https://github.com/llamasearch/llamaforge/blob/main/LICENSE) file for details. 