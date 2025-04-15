# Plugins Guide

LlamaForge's plugin system allows you to extend its functionality in various ways. This guide explains how to use and create plugins.

## Plugin Types

LlamaForge supports several types of plugins:

1. **Preprocessing**: Modify the input before sending it to the model
2. **Postprocessing**: Modify the output from the model
3. **Tools**: Add capabilities like web search, calculation, or other functions
4. **Commands**: Add new commands to the CLI

## Using Built-in Plugins

### Preprocessing Plugins

```python
from llamaforge import LlamaForge
from llamaforge.plugins.preprocessing import TextFormatterPlugin

forge = LlamaForge()
forge.load_model("llama-2-7b-chat")

# Create and configure the plugin
formatter = TextFormatterPlugin()
formatter.configure({
    "trim_whitespace": True,
    "add_system_instruction": True,
    "system_instruction": "You are a helpful assistant.",
    "format_as_chat": True,
    "chat_format": "llama2"
})

# Register the plugin
forge.register_plugin(formatter)

# Generate text (input will be preprocessed)
response = forge.generate("Explain quantum computing in simple terms")
print(response)
```

### Postprocessing Plugins

```python
from llamaforge import LlamaForge
from llamaforge.plugins.postprocessing import TextCleanerPlugin

forge = LlamaForge()
forge.load_model("llama-2-7b-chat")

# Create and configure the plugin
cleaner = TextCleanerPlugin()
cleaner.configure({
    "trim_whitespace": True,
    "remove_special_tokens": True,
    "remove_incomplete_sentences": True
})

# Register the plugin
forge.register_plugin(cleaner)

# Generate text (output will be postprocessed)
response = forge.generate("Write a short story about a robot")
print(response)
```

### Tool Plugins

```python
from llamaforge import LlamaForge
from llamaforge.plugins.tools import CalculatorPlugin

forge = LlamaForge()
forge.load_model("llama-2-7b-chat")

# Create and register the calculator tool
calculator = CalculatorPlugin()
forge.register_plugin(calculator)

# Generate text (model can use the calculator tool)
prompt = "What is 1234 * 5678? Calculate the answer step by step."
response = forge.generate(prompt)
print(response)
```

## Creating Custom Plugins

You can create custom plugins by extending the `BasePlugin` class.

### Example: Custom Preprocessing Plugin

```python
from llamaforge.plugins.base import BasePlugin

class MyCustomPreprocessor(BasePlugin):
    @property
    def name(self):
        return "my_custom_preprocessor"
    
    @property
    def description(self):
        return "A custom preprocessing plugin"
    
    def process(self, text):
        # Add your preprocessing logic here
        # For example, convert all text to lowercase
        return text.lower()
    
    def configure(self, config):
        # Store configuration
        self.config = config
```

### Example: Custom Postprocessing Plugin

```python
from llamaforge.plugins.base import BasePlugin

class MyCustomPostprocessor(BasePlugin):
    @property
    def name(self):
        return "my_custom_postprocessor"
    
    @property
    def description(self):
        return "A custom postprocessing plugin"
    
    @property
    def supports_streaming(self):
        return True  # Set to True if your plugin can process streaming output
    
    def process(self, text):
        # Add your postprocessing logic here
        # For example, capitalize the first letter of each sentence
        return '. '.join(s.strip().capitalize() for s in text.split('.') if s.strip())
    
    def configure(self, config):
        # Store configuration
        self.config = config
```

### Example: Custom Tool Plugin

```python
from llamaforge.plugins.base import BasePlugin
import json

class WeatherPlugin(BasePlugin):
    @property
    def name(self):
        return "weather"
    
    @property
    def description(self):
        return "Get weather information for a location"
    
    def get_tool_description(self):
        return {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or zip code"
                    }
                },
                "required": ["location"]
            }
        }
    
    def process_tool_call(self, tool_name, parameters):
        if tool_name != "get_weather":
            return {"error": "Unknown tool"}
        
        location = parameters.get("location")
        # Implement weather API call here
        # For example purposes, returning mock data
        return {
            "location": location,
            "temperature": "72°F",
            "conditions": "Sunny",
            "humidity": "45%"
        }
```

## Registering Custom Plugins

After creating your custom plugin, you need to register it:

```python
from llamaforge import LlamaForge
from my_plugins import MyCustomPreprocessor

forge = LlamaForge()
forge.load_model("llama-2-7b-chat")

# Create and register the custom plugin
my_plugin = MyCustomPreprocessor()
forge.register_plugin(my_plugin)

# Use it
response = forge.generate("Hello World!")
print(response)
```

## Plugin Configuration

Plugins can be configured through the `configure` method:

```python
my_plugin = MyCustomPreprocessor()
my_plugin.configure({
    "option1": "value1",
    "option2": "value2"
})
```

## Plugin Chaining

Multiple plugins can be registered and will be executed in the order they were registered:

```python
# Preprocessing plugins run in this order
forge.register_plugin(plugin1)
forge.register_plugin(plugin2)

# Postprocessing plugins run in reverse order
forge.register_plugin(post_plugin1)
forge.register_plugin(post_plugin2)  # Runs first
```

## Best Practices

1. **Keep plugins focused**: Each plugin should do one thing well
2. **Handle errors gracefully**: Don't let plugin failures crash the main application
3. **Document your plugins**: Provide clear descriptions and parameter documentation
4. **Support streaming**: When possible, design plugins to work with streaming outputs
5. **Test thoroughly**: Ensure your plugins work with different models and inputs

## Next Steps

- See the [API Reference](../api/plugins.md) for detailed information on the plugins API
- Check out the [API Server Guide](api-server.md) to learn how to use plugins with the API server 