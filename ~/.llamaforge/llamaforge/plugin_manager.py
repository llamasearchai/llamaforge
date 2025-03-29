#!/usr/bin/env python3
"""
LlamaForge Plugin Manager
This module provides a plugin system for extending LlamaForge's functionality.
Plugins can be used to add custom preprocessors, postprocessors, formatters,
commands, tools, and model adapters.
"""

import os
import sys
import json
import inspect
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Type, Tuple

# Set up logging
logger = logging.getLogger("llamaforge.plugin_manager")


# Plugin types definition
PLUGIN_TYPES = {
    "preprocessor": "Modify prompts before sending to the model",
    "postprocessor": "Modify completions before returning to the user",
    "formatter": "Format model outputs in a specific way",
    "command": "Add custom commands to the CLI",
    "tool": "Add custom tools for chat mode",
    "adapter": "Adapt different model formats and APIs"
}


class PluginBase:
    """Base class for all LlamaForge plugins."""
    
    plugin_type = None  # To be defined by subclasses
    plugin_name = None  # To be defined by plugin implementations
    plugin_description = None  # To be defined by plugin implementations
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Dictionary containing plugin configuration
        """
        self.config = config
    
    def cleanup(self):
        """
        Clean up any resources used by the plugin.
        This is called when the plugin is unloaded.
        """
        pass


class PreprocessorPlugin(PluginBase):
    """Plugin for modifying prompts before they are sent to the model."""
    
    plugin_type = "preprocessor"
    
    def process(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Process the prompt before it is sent to the model.
        
        Args:
            prompt: The original prompt
            context: Additional context about the request
            
        Returns:
            str: The modified prompt
        """
        raise NotImplementedError("Preprocessor plugins must implement process()")


class PostprocessorPlugin(PluginBase):
    """Plugin for modifying completions before they are returned to the user."""
    
    plugin_type = "postprocessor"
    
    def process(self, completion: str, prompt: str = None, context: Dict[str, Any] = None) -> str:
        """
        Process the completion before it is returned to the user.
        
        Args:
            completion: The completion text from the model
            prompt: The original prompt (if available)
            context: Additional context about the request
            
        Returns:
            str: The modified completion
        """
        raise NotImplementedError("Postprocessor plugins must implement process()")


class FormatterPlugin(PluginBase):
    """Plugin for formatting text in a specific way."""
    
    plugin_type = "formatter"
    
    def format(self, text: str, format_type: str = None, context: Dict[str, Any] = None) -> str:
        """
        Format the text according to a specific format type.
        
        Args:
            text: The text to format
            format_type: The type of formatting to apply (optional)
            context: Additional context about the request
            
        Returns:
            str: The formatted text
        """
        raise NotImplementedError("Formatter plugins must implement format()")


class CommandPlugin(PluginBase):
    """Plugin for adding custom commands to the CLI."""
    
    plugin_type = "command"
    
    def get_command_info(self) -> Dict[str, Any]:
        """
        Get information about the command this plugin provides.
        
        Returns:
            Dict containing command information:
                name: Command name
                description: Command description
                help: Help text for the command
                args: List of argument definitions
        """
        raise NotImplementedError("Command plugins must implement get_command_info()")
    
    def execute(self, args: List[str], context: Dict[str, Any] = None) -> Any:
        """
        Execute the command with the given arguments.
        
        Args:
            args: Arguments passed to the command
            context: Additional context about the CLI environment
            
        Returns:
            Any: The result of the command execution
        """
        raise NotImplementedError("Command plugins must implement execute()")


class ToolPlugin(PluginBase):
    """Plugin for adding custom tools in chat mode."""
    
    plugin_type = "tool"
    
    def get_tool_info(self) -> Dict[str, Any]:
        """
        Get information about the tool this plugin provides.
        
        Returns:
            Dict containing tool information:
                name: Tool name
                description: Tool description
                parameters: Tool parameters description (JSON schema format)
        """
        raise NotImplementedError("Tool plugins must implement get_tool_info()")
    
    def execute(self, params: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """
        Execute the tool with the given parameters.
        
        Args:
            params: Parameters passed to the tool
            context: Additional context about the chat environment
            
        Returns:
            Any: The result of the tool execution
        """
        raise NotImplementedError("Tool plugins must implement execute()")


class AdapterPlugin(PluginBase):
    """Plugin for adapting different model formats and APIs."""
    
    plugin_type = "adapter"
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """
        Get information about the adapter.
        
        Returns:
            Dict containing adapter information:
                supported_formats: List of supported model formats
                capabilities: List of capabilities (e.g., 'chat', 'completion', etc.)
        """
        raise NotImplementedError("Adapter plugins must implement get_adapter_info()")
    
    def adapt_input(self, input_data: Any, target_format: str, context: Dict[str, Any] = None) -> Any:
        """
        Adapt input data to the target format.
        
        Args:
            input_data: The input data to adapt
            target_format: The target format to adapt to
            context: Additional context
            
        Returns:
            Any: The adapted input data
        """
        raise NotImplementedError("Adapter plugins must implement adapt_input()")
    
    def adapt_output(self, output_data: Any, source_format: str, target_format: str = None, context: Dict[str, Any] = None) -> Any:
        """
        Adapt output data from the source format to the target format.
        
        Args:
            output_data: The output data to adapt
            source_format: The source format of the data
            target_format: The target format to adapt to (optional)
            context: Additional context
            
        Returns:
            Any: The adapted output data
        """
        raise NotImplementedError("Adapter plugins must implement adapt_output()")


class PluginManager:
    """Manager for loading, unloading, and running plugins."""
    
    # Mapping of plugin types to their classes
    PLUGIN_CLASSES = {
        "preprocessor": PreprocessorPlugin,
        "postprocessor": PostprocessorPlugin,
        "formatter": FormatterPlugin,
        "command": CommandPlugin,
        "tool": ToolPlugin,
        "adapter": AdapterPlugin
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the plugin manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.plugins_dir = os.path.expanduser(config.get("directories", {}).get("plugins", "~/.llamaforge/plugins"))
        self.plugins = {}  # Type: Dict[str, Dict[str, PluginBase]]
        
        # Create plugins directory if it doesn't exist
        os.makedirs(self.plugins_dir, exist_ok=True)
        
        # Initialize plugin categories
        for plugin_type in PLUGIN_TYPES:
            self.plugins[plugin_type] = {}
        
        # Load plugins if directory exists
        if os.path.exists(self.plugins_dir):
            self.discover_plugins()
    
    def discover_plugins(self):
        """Discover available plugins in the plugins directory."""
        logger.info(f"Discovering plugins in {self.plugins_dir}")
        
        # Ensure the plugins directory exists
        if not os.path.exists(self.plugins_dir):
            logger.warning(f"Plugins directory {self.plugins_dir} does not exist")
            return
        
        # Find all Python files in the plugins directory
        for plugin_file in Path(self.plugins_dir).glob("**/*.py"):
            if plugin_file.name.startswith("_"):
                continue  # Skip files that start with underscore
                
            try:
                # Get relative path to plugin file
                rel_path = plugin_file.relative_to(self.plugins_dir)
                plugin_id = str(rel_path).replace("/", ".").replace("\\", ".")[:-3]  # Remove .py extension
                
                # Check if plugin is already loaded
                if any(plugin_id in self.plugins[plugin_type] for plugin_type in self.plugins):
                    logger.debug(f"Plugin {plugin_id} already loaded, skipping")
                    continue
                
                # Load plugin
                self.load_plugin(str(plugin_file), plugin_id)
                
            except Exception as e:
                logger.error(f"Error discovering plugin {plugin_file}: {e}")
    
    def load_plugin(self, plugin_path: str, plugin_id: str = None) -> Optional[PluginBase]:
        """
        Load a plugin from the given path.
        
        Args:
            plugin_path: Path to the plugin file
            plugin_id: ID to use for the plugin (if None, derived from the path)
            
        Returns:
            The loaded plugin instance, or None if loading failed
        """
        try:
            if plugin_id is None:
                # Generate plugin ID from path
                plugin_id = os.path.basename(plugin_path)
                if plugin_id.endswith(".py"):
                    plugin_id = plugin_id[:-3]  # Remove .py extension
            
            logger.info(f"Loading plugin {plugin_id} from {plugin_path}")
            
            # Load module
            spec = importlib.util.spec_from_file_location(plugin_id, plugin_path)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to load plugin {plugin_id}: Invalid module specification")
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                    
                attr = getattr(module, attr_name)
                if not inspect.isclass(attr) or not issubclass(attr, PluginBase) or attr is PluginBase:
                    continue
                
                # Skip abstract plugin classes
                if attr in self.PLUGIN_CLASSES.values():
                    continue
                
                # Create plugin instance
                plugin_instance = attr(self.config)
                if not hasattr(plugin_instance, "plugin_type") or not plugin_instance.plugin_type:
                    logger.warning(f"Plugin class {attr_name} in {plugin_id} has no plugin_type defined")
                    continue
                
                if not hasattr(plugin_instance, "plugin_name") or not plugin_instance.plugin_name:
                    logger.warning(f"Plugin class {attr_name} in {plugin_id} has no plugin_name defined")
                    continue
                
                # Store the plugin
                plugin_type = plugin_instance.plugin_type
                if plugin_type not in self.plugins:
                    logger.warning(f"Plugin {plugin_id} has unknown type {plugin_type}")
                    continue
                
                plugin_name = plugin_instance.plugin_name
                self.plugins[plugin_type][plugin_name] = plugin_instance
                logger.info(f"Loaded plugin {plugin_name} of type {plugin_type} from {plugin_id}")
                
                return plugin_instance
                
            logger.warning(f"No valid plugin classes found in {plugin_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_id}: {e}")
            return None
    
    def unload_plugin(self, plugin_type: str, plugin_name: str) -> bool:
        """
        Unload a plugin by type and name.
        
        Args:
            plugin_type: Type of the plugin
            plugin_name: Name of the plugin
            
        Returns:
            bool: True if unloaded successfully, False otherwise
        """
        if plugin_type not in self.plugins:
            logger.warning(f"Unknown plugin type {plugin_type}")
            return False
        
        if plugin_name not in self.plugins[plugin_type]:
            logger.warning(f"Plugin {plugin_name} of type {plugin_type} not found")
            return False
        
        try:
            # Call cleanup method
            self.plugins[plugin_type][plugin_name].cleanup()
            
            # Remove from plugins dict
            del self.plugins[plugin_type][plugin_name]
            logger.info(f"Unloaded plugin {plugin_name} of type {plugin_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name} of type {plugin_type}: {e}")
            return False
    
    def get_plugins(self, plugin_type: str = None) -> Dict[str, Union[PluginBase, Dict[str, PluginBase]]]:
        """
        Get loaded plugins, optionally filtered by type.
        
        Args:
            plugin_type: Type of plugins to get (if None, get all)
            
        Returns:
            Dict of plugin instances by name, or dict of plugin types if no type specified
        """
        if plugin_type is not None:
            if plugin_type not in self.plugins:
                logger.warning(f"Unknown plugin type {plugin_type}")
                return {}
            return self.plugins[plugin_type]
        
        return self.plugins
    
    def run_preprocessors(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Run all preprocessor plugins on the given prompt.
        
        Args:
            prompt: The original prompt
            context: Additional context
            
        Returns:
            str: The processed prompt
        """
        processed = prompt
        for plugin_name, plugin in self.plugins["preprocessor"].items():
            try:
                processed = plugin.process(processed, context)
                logger.debug(f"Preprocessor {plugin_name} processed prompt")
            except Exception as e:
                logger.error(f"Error running preprocessor {plugin_name}: {e}")
        return processed
    
    def run_postprocessors(self, completion: str, prompt: str = None, context: Dict[str, Any] = None) -> str:
        """
        Run all postprocessor plugins on the given completion.
        
        Args:
            completion: The completion from the model
            prompt: The original prompt (if available)
            context: Additional context
            
        Returns:
            str: The processed completion
        """
        processed = completion
        for plugin_name, plugin in self.plugins["postprocessor"].items():
            try:
                processed = plugin.process(processed, prompt, context)
                logger.debug(f"Postprocessor {plugin_name} processed completion")
            except Exception as e:
                logger.error(f"Error running postprocessor {plugin_name}: {e}")
        return processed
    
    def run_formatter(self, formatter_name: str, text: str, format_type: str = None, context: Dict[str, Any] = None) -> Optional[str]:
        """
        Run a specific formatter plugin on the given text.
        
        Args:
            formatter_name: Name of the formatter to use
            text: Text to format
            format_type: Type of formatting to apply (optional)
            context: Additional context
            
        Returns:
            str: The formatted text, or None if formatter not found or error occurred
        """
        if formatter_name not in self.plugins["formatter"]:
            logger.warning(f"Formatter {formatter_name} not found")
            return None
        
        try:
            formatted = self.plugins["formatter"][formatter_name].format(text, format_type, context)
            logger.debug(f"Formatter {formatter_name} formatted text")
            return formatted
        except Exception as e:
            logger.error(f"Error running formatter {formatter_name}: {e}")
            return None
    
    def execute_command(self, command_name: str, args: List[str], context: Dict[str, Any] = None) -> Any:
        """
        Execute a command plugin with the given arguments.
        
        Args:
            command_name: Name of the command to execute
            args: Arguments to pass to the command
            context: Additional context
            
        Returns:
            Any: The result of the command execution, or None if command not found or error occurred
        """
        if command_name not in self.plugins["command"]:
            logger.warning(f"Command {command_name} not found")
            return None
        
        try:
            result = self.plugins["command"][command_name].execute(args, context)
            logger.debug(f"Command {command_name} executed")
            return result
        except Exception as e:
            logger.error(f"Error executing command {command_name}: {e}")
            return None
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """
        Execute a tool plugin with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool
            context: Additional context
            
        Returns:
            Any: The result of the tool execution, or None if tool not found or error occurred
        """
        if tool_name not in self.plugins["tool"]:
            logger.warning(f"Tool {tool_name} not found")
            return None
        
        try:
            result = self.plugins["tool"][tool_name].execute(params, context)
            logger.debug(f"Tool {tool_name} executed")
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return None
    
    def get_available_commands(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available command plugins.
        
        Returns:
            Dict mapping command names to their info dictionaries
        """
        commands = {}
        for command_name, plugin in self.plugins["command"].items():
            try:
                command_info = plugin.get_command_info()
                commands[command_name] = command_info
            except Exception as e:
                logger.error(f"Error getting info for command {command_name}: {e}")
        return commands
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available tool plugins.
        
        Returns:
            Dict mapping tool names to their info dictionaries
        """
        tools = {}
        for tool_name, plugin in self.plugins["tool"].items():
            try:
                tool_info = plugin.get_tool_info()
                tools[tool_name] = tool_info
            except Exception as e:
                logger.error(f"Error getting info for tool {tool_name}: {e}")
        return tools
    
    def create_sample_plugin(self, plugin_type: str, plugin_name: str) -> str:
        """
        Create a sample plugin file with the given type and name.
        
        Args:
            plugin_type: Type of plugin to create
            plugin_name: Name for the plugin
            
        Returns:
            str: Path to the created plugin file, or None if creation failed
        """
        if plugin_type not in PLUGIN_TYPES:
            logger.error(f"Unknown plugin type {plugin_type}")
            return None
        
        # Create sanitized filename from plugin name
        safe_name = "".join(c if c.isalnum() else "_" for c in plugin_name.lower())
        filename = f"{safe_name}_{plugin_type}.py"
        filepath = os.path.join(self.plugins_dir, filename)
        
        # Check if file already exists
        if os.path.exists(filepath):
            logger.warning(f"Plugin file {filepath} already exists")
            return None
        
        # Create template content based on plugin type
        try:
            # Get the base class for this plugin type
            base_class = self.PLUGIN_CLASSES[plugin_type]
            
            # Create template
            content = f"""#!/usr/bin/env python3
\"\"\"
{plugin_name} - {PLUGIN_TYPES[plugin_type]}
A sample {plugin_type} plugin for LlamaForge.
\"\"\"

from llamaforge.plugin_manager import {base_class.__name__}


class {plugin_name.replace(" ", "")}({base_class.__name__}):
    \"\"\"
    {plugin_name} - {PLUGIN_TYPES[plugin_type]}
    \"\"\"
    
    # Required plugin attributes
    plugin_type = "{plugin_type}"
    plugin_name = "{plugin_name}"
    plugin_description = "A sample {plugin_type} plugin for LlamaForge"
    
    def __init__(self, config):
        super().__init__(config)
        # Add your initialization code here
        
"""
            
            # Add additional methods based on plugin type
            if plugin_type == "preprocessor":
                content += """    def process(self, prompt, context=None):
        \"\"\"Process the prompt before it is sent to the model.\"\"\"
        # Example: Add a prefix to all prompts
        return f"Enhanced query: {prompt}"
"""
            elif plugin_type == "postprocessor":
                content += """    def process(self, completion, prompt=None, context=None):
        \"\"\"Process the completion after it is returned from the model.\"\"\"
        # Example: Add a suffix to all completions
        return f"{completion}\\n\\nGenerated by LlamaForge"
"""
            elif plugin_type == "formatter":
                content += """    def format(self, text, format_type=None, context=None):
        \"\"\"Format the text according to a specific format type.\"\"\"
        if format_type == "uppercase":
            return text.upper()
        elif format_type == "lowercase":
            return text.lower()
        else:
            return text
"""
            elif plugin_type == "command":
                content += """    def get_command_info(self):
        \"\"\"Get information about the command this plugin provides.\"\"\"
        return {
            "name": "{plugin_name}",
            "description": "A sample command plugin",
            "help": "Usage: {plugin_name} [arguments]",
            "args": [
                {"name": "arg1", "type": "string", "description": "First argument"},
                {"name": "arg2", "type": "int", "description": "Second argument"}
            ]
        }
    
    def execute(self, args, context=None):
        \"\"\"Execute the command with the given arguments.\"\"\"
        return f"Executed {self.plugin_name} with args: {args}"
""".format(plugin_name=plugin_name.lower().replace(" ", "-"))
            elif plugin_type == "tool":
                content += """    def get_tool_info(self):
        \"\"\"Get information about the tool this plugin provides.\"\"\"
        return {
            "name": "{plugin_name}",
            "description": "A sample tool plugin",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "First parameter"
                    },
                    "param2": {
                        "type": "number",
                        "description": "Second parameter"
                    }
                },
                "required": ["param1"]
            }
        }
    
    def execute(self, params, context=None):
        \"\"\"Execute the tool with the given parameters.\"\"\"
        return f"Executed {self.plugin_name} with params: {params}"
""".format(plugin_name=plugin_name.lower().replace(" ", "_"))
            elif plugin_type == "adapter":
                content += """    def get_adapter_info(self):
        \"\"\"Get information about the adapter.\"\"\"
        return {
            "supported_formats": ["format1", "format2"],
            "capabilities": ["chat", "completion"]
        }
    
    def adapt_input(self, input_data, target_format, context=None):
        \"\"\"Adapt input data to the target format.\"\"\"
        # Implement your input adaptation logic here
        return input_data
    
    def adapt_output(self, output_data, source_format, target_format=None, context=None):
        \"\"\"Adapt output data from the source format to the target format.\"\"\"
        # Implement your output adaptation logic here
        return output_data
"""
            
            # Add cleanup method
            content += """    def cleanup(self):
        \"\"\"Clean up any resources used by the plugin.\"\"\"
        # Add your cleanup code here
        pass
"""
            
            # Write template to file
            with open(filepath, "w") as f:
                f.write(content)
            
            logger.info(f"Created sample {plugin_type} plugin at {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error creating sample plugin: {e}")
            return None 