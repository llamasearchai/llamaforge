# LlamaForge

<div align="center">
  <img src="docs/images/llamaforge_logo.png" alt="LlamaForge Logo" width="200"/>
  <h3>A Comprehensive Language Model Command-Line Interface</h3>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

LlamaForge provides a unified command-line interface for working with language models, supporting multiple backends including llama.cpp, MLX, and Transformers.

## Features

- 🚀 **Multi-backend Support**: Use your model with llama.cpp, MLX, or Transformers.
- 🔄 **Model Management**: Download, import, and manage local models.
- 💬 **Interactive Chat**: Rich interactive chat mode with command support.
- 🌐 **API Server**: Compatible with OpenAI's API for integration with existing tools.
- 🔌 **Plugin System**: Extend functionality with custom plugins.
- 🛠️ **Configuration Wizard**: Easy setup with an interactive configuration wizard.
- 📊 **Benchmarking**: Measure inference speed and performance.
- 📦 **Fine-tuning**: Fine-tune models to your specific use case (experimental).

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/llamaforge.git
cd llamaforge

# Install LlamaForge
python scripts/install_llamaforge.py
```

### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llamaforge.git
cd llamaforge
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the setup script:
```bash
python scripts/install_llamaforge.py
```

## Getting Started

1. Run the configuration wizard to set up LlamaForge:
```bash
llamaforge config
```

2. Download a model:
```bash
llamaforge model add --name TheBloke/Llama-2-7B-Chat-GGUF
```

3. Start chatting:
```bash
llamaforge chat --model Llama-2-7B-Chat
```

## Usage

### Interactive Chat

```bash
llamaforge chat --model MODEL_NAME
```

### Text Generation

```bash
llamaforge generate --model MODEL_NAME --prompt "Once upon a time"
```

### API Server

```bash
llamaforge api --host localhost --port 8000
```

### Model Management

```bash
# List available models
llamaforge model list

# Add a model
llamaforge model add --name TheBloke/Llama-2-7B-Chat-GGUF

# Remove a model
llamaforge model remove --name Llama-2-7B-Chat
```

### Configuration

```bash
llamaforge config
```

## Plugin System

LlamaForge includes a flexible plugin system allowing you to extend its functionality.

### Available Plugin Types

- **Preprocessor**: Modify prompts before sending to the model
- **Postprocessor**: Modify completions before returning to the user
- **Formatter**: Format model outputs in a specific way
- **Command**: Add custom commands to the CLI
- **Tool**: Add custom tools for chat mode
- **Adapter**: Adapt different model formats and APIs

### Creating a Plugin

Create a new Python file in the `~/.llamaforge/plugins` directory:

```python
from llamaforge.plugin_manager import PreprocessorPlugin

class MyPlugin(PreprocessorPlugin):
    def __init__(self):
        super().__init__(
            name="my_plugin",
            description="A sample plugin that modifies the prompt"
        )
    
    def process(self, prompt):
        return f"Enhanced prompt: {prompt}"
```

## Configuration

LlamaForge's configuration is stored in `~/.llamaforge/config.json`. You can modify it directly or use the configuration wizard:

```bash
llamaforge config
```

## Backend Support

LlamaForge supports multiple backends:

- **llama.cpp**: Fast C++ implementation, optimized for CPU inference
- **MLX**: Apple's ML framework, optimized for Apple Silicon
- **Transformers**: Hugging Face's Transformers library with PyTorch/TensorFlow

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [MLX](https://github.com/ml-explore/mlx)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Rich](https://github.com/Textualize/rich)

## Testing

LlamaForge includes a test suite to verify core functionality. To run the tests:

```bash
# Run all tests
python tests/run_tests.py

# Run a specific test file
python tests/test_core.py
```

## Development and Publishing

### Setting up Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llamaforge.git
   cd llamaforge
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   pip install -e ".[dev]"  # Installs development dependencies
   ```

### Publishing to PyPI

1. Ensure all tests pass:
   ```bash
   python tests/run_tests.py
   ```

2. Update version in `llamaforge/version.py`

3. Build the package:
   ```bash
   python setup.py sdist bdist_wheel
   ```

4. Upload to PyPI:
   ```bash
   twine upload dist/*
   ``` 