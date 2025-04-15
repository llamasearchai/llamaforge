# Configuration Guide

LlamaForge uses a configuration system to manage settings, models, and preferences. This guide explains how to work with the configuration.

## Configuration File

By default, LlamaForge stores its configuration in a JSON file at `~/.llamaforge/config.json`. When you first run LlamaForge, this file is created automatically with default settings.

## Viewing the Configuration

### Using Python

```python
from llamaforge import LlamaForge

forge = LlamaForge()
config = forge.config.get_all()
print(config)
```

### Using Command Line

```bash
llamaforge config list
```

## Modifying Configuration

### Using Python

```python
from llamaforge import LlamaForge

forge = LlamaForge()

# Set a top-level configuration value
forge.config.set("default_model", "llama-2-7b-chat")

# Set a nested configuration value
forge.config.set("logging.level", "INFO")

# Save the changes
forge.config.save()
```

### Using Command Line

```bash
# Set a top-level configuration value
llamaforge config set default_model llama-2-7b-chat

# Set a nested configuration value
llamaforge config set logging.level INFO
```

## Configuration Options

Here are the main configuration options:

### General Settings

- `default_model`: The default model to use when none is specified
- `models_dir`: Directory where models are stored
- `cache_dir`: Directory for caching model data
- `temp_dir`: Directory for temporary files

### Logging Settings

- `logging.level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Log file path
- `logging.format`: Log message format

### Model Configurations

Models are configured in the `models` section:

```json
{
  "models": {
    "llama-2-7b-chat": {
      "path": "/path/to/llama-2-7b-chat.gguf",
      "backend": "llama.cpp",
      "parameters": {
        "temperature": 0.7,
        "top_p": 0.9,
        "n_ctx": 4096
      }
    }
  }
}
```

### Backend Configurations

Backend-specific settings are in the `backends` section:

```json
{
  "backends": {
    "llama.cpp": {
      "default_parameters": {
        "n_gpu_layers": 32,
        "n_ctx": 2048
      }
    },
    "openai": {
      "default_parameters": {
        "api_key": "your-api-key",
        "organization": "your-org-id"
      }
    }
  }
}
```

### Server Settings

Settings for the API server:

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8000,
    "enable_cors": true,
    "request_timeout": 300
  }
}
```

## Using a Custom Configuration File

You can specify a custom configuration file:

### Using Python

```python
from llamaforge import LlamaForge

forge = LlamaForge(config_path="/path/to/custom/config.json")
```

### Using Command Line

```bash
llamaforge --config /path/to/custom/config.json <command>
```

## Environment Variables

LlamaForge respects the following environment variables:

- `LLAMAFORGE_CONFIG`: Path to the configuration file
- `LLAMAFORGE_DEFAULT_MODEL`: Default model to use
- `LLAMAFORGE_OPENAI_API_KEY`: OpenAI API key
- `LLAMAFORGE_MODELS_DIR`: Directory where models are stored

Environment variables take precedence over the configuration file.

## Configuration Profiles

LlamaForge supports multiple configuration profiles:

### Creating a Profile

```python
from llamaforge import LlamaForge

forge = LlamaForge()

# Create a new profile
forge.config.create_profile("production")

# Switch to the profile
forge.config.set_active_profile("production")

# Configure the profile
forge.config.set("default_model", "gpt-4")
forge.config.save()
```

### Using Command Line

```bash
# Create a profile
llamaforge config create-profile production

# Switch to a profile
llamaforge config set-profile production

# Run commands with a specific profile
llamaforge --profile production chat
```

## Best Practices

1. **Keep sensitive information in environment variables**: API keys should be set via environment variables rather than in the configuration file.
2. **Use profiles for different environments**: Create separate profiles for development, testing, and production.
3. **Use absolute paths for models**: To ensure portability, specify absolute paths for model files.
4. **Backup your configuration**: Keep a backup of your configuration in a secure location.
5. **Version control your configuration**: If you're developing with LlamaForge, consider version controlling your configuration (with sensitive information removed).

## Next Steps

- Learn about [Working with Models](../guides/models.md)
- Explore the [Plugins Guide](../guides/plugins.md) 