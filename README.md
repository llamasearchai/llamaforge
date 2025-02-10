# LlamaForge

Ultimate Language Model Command Interface

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Version](https://img.shields.io/badge/version-0.2.0-green)

LlamaForge is a powerful command-line tool and Python library designed to streamline working with large language models. It provides a unified interface for managing, running, and optimizing various language models from different providers.

## Features

- **Multiple Backends**: Support for llama.cpp, Hugging Face, and OpenAI models
- **Plugin System**: Extend functionality with preprocessing, postprocessing, tools, and commands
- **API Server**: OpenAI-compatible API for easy integration with existing tools
- **Command Line Interface**: Simple and intuitive CLI for working with models
- **Configuration Management**: Flexible configuration system for models and backends
- **Streaming Support**: Stream generated text for real-time applications

## Installation

### Basic Installation

```bash
pip install llamaforge
```

### With Backend Support

```bash
# For llama.cpp models
pip install "llamaforge[llama.cpp]"

# For Hugging Face models
pip install "llamaforge[huggingface]"

# For OpenAI API
pip install "llamaforge[openai]"

# For API server
pip install "llamaforge[server]"

# For all features
pip install "llamaforge[all]"
```

## Quick Start

### Python API

```python
from llamaforge import LlamaForge

# Initialize LlamaForge
forge = LlamaForge()

# Load a model
forge.load_model("llama-2-7b-chat")

# Generate text
response = forge.generate("Explain quantum computing in simple terms")
print(response)
```

### Command Line

```bash
# Initialize configuration
llamaforge config init

# List available models
llamaforge list models

# Chat with a model
llamaforge chat --model llama-2-7b-chat

# Generate text from a prompt
llamaforge run "Explain quantum computing in simple terms" --model llama-2-7b-chat

# Start API server
llamaforge serve --model llama-2-7b-chat
```

## Configuration

LlamaForge uses a configuration file located at `~/.llamaforge/config.json`. You can initialize it with the following command:

```bash
llamaforge config init
```

To set configuration values:

```bash
llamaforge config set default_model llama-2-7b-chat
```

To get configuration values:

```bash
llamaforge config get default_model
```

## Plugin System

LlamaForge includes a powerful plugin system for extending functionality:

- **Preprocessing**: Modify input text before sending to the model
- **Postprocessing**: Clean and format model outputs
- **Tools**: Add capabilities like web search, code execution, etc.
- **Commands**: Extend the CLI with custom commands

Example configuration for plugins:

```json
{
  "plugins": [
    {
      "type": "preprocessing",
      "name": "text_formatter",
      "config": {
        "trim_whitespace": true,
        "format_as_chat": true,
        "chat_format": "llama2"
      }
    },
    {
      "type": "postprocessing",
      "name": "text_cleaner",
      "config": {
        "remove_special_tokens": true,
        "trim_whitespace": true
      }
    }
  ]
}
```

## API Server

LlamaForge includes an OpenAI-compatible API server:

```bash
llamaforge serve --host 127.0.0.1 --port 8000 --model llama-2-7b-chat
```

Then you can use it with OpenAI-compatible clients:

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # Not used but required
    base_url="http://127.0.0.1:8000/v1"
)

response = client.chat.completions.create(
    model="llama-2-7b-chat",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
print(response.choices[0].message.content)
```

## License

LlamaForge is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
# Updated in commit 1 - 2025-04-04 17:03:57

# Updated in commit 9 - 2025-04-04 17:04:03

# Updated in commit 17 - 2025-04-04 17:04:09

# Updated in commit 25 - 2025-04-04 17:04:14

# Updated in commit 1 - 2025-04-05 14:24:48

# Updated in commit 9 - 2025-04-05 14:24:49

# Updated in commit 17 - 2025-04-05 14:24:49

# Updated in commit 25 - 2025-04-05 14:24:49

# Updated in commit 1 - 2025-04-05 15:00:42

# Updated in commit 9 - 2025-04-05 15:00:42

# Updated in commit 17 - 2025-04-05 15:00:43

# Updated in commit 25 - 2025-04-05 15:00:43

# Updated in commit 1 - 2025-04-05 15:10:25

# Updated in commit 9 - 2025-04-05 15:10:26

# Updated in commit 17 - 2025-04-05 15:10:26
