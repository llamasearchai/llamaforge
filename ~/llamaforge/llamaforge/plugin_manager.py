#!/usr/bin/env python3
"""
LlamaForge Plugin Manager

This module provides a plugin system for extending LlamaForge's functionality.
Plugins can be used to:
- Preprocess prompts before sending to the model
- Postprocess completions before returning to the user
- Add custom commands to the CLI
- Add custom tools for chat mode
- Format model outputs in a specific way
- Adapt to different model formats and APIs
"""

import os
import sys
import json
import inspect
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Type, Union, Tuple

logger = logging.getLogger("llamaforge.plugin_manager")

# Plugin types dictionary with descriptions
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
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the plugin with configuration.
        
        Args:
            config: Plugin configuration
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.description = config.get("description", "")
        self.version = config.get("version", "0.1.0")
        self.author = config.get("author", "Unknown")
        self.enabled = config.get("enabled", True)
    
    def initialize(self) -> bool:
        """Initialize the plugin.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        logger.debug(f"Initializing plugin: {self.name}")
        return True
    
    def cleanup(self) -> bool:
        """Clean up when the plugin is unloaded.
        
        Returns:
            True if cleanup was successful, False otherwise
        """
        logger.debug(f"Cleaning up plugin: {self.name}")
        return True
    
    def enable(self) -> bool:
        """Enable the plugin.
        
        Returns:
            True if enabling was successful, False otherwise
        """
        logger.debug(f"Enabling plugin: {self.name}")
        self.enabled = True
        return True
    
    def disable(self) -> bool:
        """Disable the plugin.
        
        Returns:
            True if disabling was successful, False otherwise
        """
        logger.debug(f"Disabling plugin: {self.name}")
        self.enabled = False
        return True


class PreprocessorPlugin(PluginBase):
    """Plugin for preprocessing prompts before sending to the model."""
    
    def process_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Process the prompt.
        
        Args:
            prompt: The prompt to process
            context: Additional context
            
        Returns:
            Processed prompt
        """
        logger.debug(f"Plugin {self.name} processing prompt")
        return prompt


class PostprocessorPlugin(PluginBase):
    """Plugin for postprocessing completions before returning to the user."""
    
    def process_completion(self, completion: str, context: Dict[str, Any]) -> str:
        """Process the completion.
        
        Args:
            completion: The completion to process
            context: Additional context
            
        Returns:
            Processed completion
        """
        logger.debug(f"Plugin {self.name} processing completion")
        return completion


class FormatterPlugin(PluginBase):
    """Plugin for formatting text in a specific way."""
    
    def format_text(self, text: str, format_type: str, context: Dict[str, Any]) -> str:
        """Format the text.
        
        Args:
            text: The text to format
            format_type: The type of formatting to apply
            context: Additional context
            
        Returns:
            Formatted text
        """
        logger.debug(f"Plugin {self.name} formatting text as {format_type}")
        return text


class CommandPlugin(PluginBase):
    """Plugin for adding custom commands to the CLI."""
    
    def get_commands(self) -> Dict[str, Callable]:
        """Get the commands provided by this plugin.
        
        Returns:
            Dictionary mapping command names to functions
        """
        logger.debug(f"Plugin {self.name} providing commands")
        return {}
    
    def execute_command(self, command: str, args: List[str], context: Dict[str, Any]) -> Any:
        """Execute a command.
        
        Args:
            command: The command to execute
            args: Command arguments
            context: Additional context
            
        Returns:
            Command result
        """
        logger.debug(f"Plugin {self.name} executing command: {command}")
        commands = self.get_commands()
        if command in commands:
            return commands[command](*args)
        raise ValueError(f"Command {command} not found in plugin {self.name}")


class ToolPlugin(PluginBase):
    """Plugin for adding custom tools for chat mode."""
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get the tools provided by this plugin.
        
        Returns:
            Dictionary mapping tool names to tool definitions
        """
        logger.debug(f"Plugin {self.name} providing tools")
        return {}
    
    def execute_tool(self, tool: str, args: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute a tool.
        
        Args:
            tool: The tool to execute
            args: Tool arguments
            context: Additional context
            
        Returns:
            Tool result
        """
        logger.debug(f"Plugin {self.name} executing tool: {tool}")
        raise NotImplementedError(f"Tool {tool} not implemented in plugin {self.name}")


class AdapterPlugin(PluginBase):
    """Plugin for adapting to different model formats and APIs."""
    
    def adapt_input(self, input_data: Any, model_type: str, context: Dict[str, Any]) -> Any:
        """Adapt input data to the format expected by the model.
        
        Args:
            input_data: Input data to adapt
            model_type: Type of model
            context: Additional context
            
        Returns:
            Adapted input data
        """
        logger.debug(f"Plugin {self.name} adapting input for model type: {model_type}")
        return input_data
    
    def adapt_output(self, output_data: Any, model_type: str, context: Dict[str, Any]) -> Any:
        """Adapt output data from the model to the format expected by the user.
        
        Args:
            output_data: Output data to adapt
            model_type: Type of model
            context: Additional context
            
        Returns:
            Adapted output data
        """
        logger.debug(f"Plugin {self.name} adapting output for model type: {model_type}")
        return output_data


class PluginManager:
    """Manager for loading, unloading, and running plugins."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the plugin manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.plugins_dir = Path(os.path.expanduser(config.get("dirs", {}).get("plugins", "~/.llamaforge/plugins")))
        self.plugins: Dict[str, PluginBase] = {}
        self.plugins_by_type: Dict[str, List[PluginBase]] = {
            plugin_type: [] for plugin_type in PLUGIN_TYPES.keys()
        }
        self.enabled = config.get("advanced", {}).get("plugins_enabled", True)
        
        # Create plugins directory if it doesn't exist
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Plugin manager initialized with plugins directory: {self.plugins_dir}")
    
    def discover_plugins(self) -> Dict[str, Dict[str, Any]]:
        """Discover available plugins.
        
        Returns:
            Dictionary mapping plugin IDs to plugin metadata
        """
        plugins = {}
        
        if not self.enabled:
            logger.info("Plugin system is disabled")
            return plugins
        
        logger.info(f"Discovering plugins in {self.plugins_dir}")
        
        # Check if the directory exists
        if not self.plugins_dir.exists():
            logger.warning(f"Plugins directory {self.plugins_dir} does not exist")
            return plugins
        
        # Scan all .py files in the plugins directory
        for plugin_file in self.plugins_dir.glob("*.py"):
            try:
                # Try to load the plugin module
                module_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                if spec is None or spec.loader is None:
                    logger.warning(f"Failed to load plugin spec: {plugin_file}")
                    continue
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check if the module defines a get_plugin_metadata function
                if hasattr(module, "get_plugin_metadata"):
                    metadata = module.get_plugin_metadata()
                    plugins[module_name] = metadata
                    logger.info(f"Discovered plugin: {module_name} - {metadata.get('name', 'Unnamed')}")
                else:
                    logger.warning(f"Plugin {module_name} does not define get_plugin_metadata()")
            
            except Exception as e:
                logger.error(f"Error loading plugin {plugin_file}: {e}")
        
        return plugins
    
    def load_plugin(self, plugin_id: str) -> Optional[PluginBase]:
        """Load a plugin.
        
        Args:
            plugin_id: ID of the plugin to load
            
        Returns:
            Loaded plugin instance, or None if loading failed
        """
        if not self.enabled:
            logger.warning("Plugin system is disabled, cannot load plugin")
            return None
        
        logger.info(f"Loading plugin: {plugin_id}")
        
        # Check if the plugin is already loaded
        if plugin_id in self.plugins:
            logger.warning(f"Plugin {plugin_id} is already loaded")
            return self.plugins[plugin_id]
        
        # Try to load the plugin module
        plugin_path = self.plugins_dir / f"{plugin_id}.py"
        if not plugin_path.exists():
            logger.error(f"Plugin {plugin_id} not found at {plugin_path}")
            return None
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(plugin_id, plugin_path)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to load plugin spec: {plugin_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if the module defines a plugin class
            if hasattr(module, "Plugin"):
                plugin_class = module.Plugin
                if not issubclass(plugin_class, PluginBase):
                    logger.error(f"Plugin {plugin_id} does not inherit from PluginBase")
                    return None
                
                # Create plugin instance
                plugin = plugin_class(self.config)
                
                # Initialize the plugin
                if not plugin.initialize():
                    logger.error(f"Failed to initialize plugin {plugin_id}")
                    return None
                
                # Register the plugin
                self.plugins[plugin_id] = plugin
                
                # Add to plugins by type
                for plugin_type, plugin_list in self.plugins_by_type.items():
                    if hasattr(module, "get_plugin_type") and module.get_plugin_type() == plugin_type:
                        plugin_list.append(plugin)
                        logger.info(f"Registered plugin {plugin_id} as {plugin_type}")
                    elif plugin_type == "preprocessor" and isinstance(plugin, PreprocessorPlugin):
                        plugin_list.append(plugin)
                        logger.info(f"Registered plugin {plugin_id} as {plugin_type}")
                    elif plugin_type == "postprocessor" and isinstance(plugin, PostprocessorPlugin):
                        plugin_list.append(plugin)
                        logger.info(f"Registered plugin {plugin_id} as {plugin_type}")
                    elif plugin_type == "formatter" and isinstance(plugin, FormatterPlugin):
                        plugin_list.append(plugin)
                        logger.info(f"Registered plugin {plugin_id} as {plugin_type}")
                    elif plugin_type == "command" and isinstance(plugin, CommandPlugin):
                        plugin_list.append(plugin)
                        logger.info(f"Registered plugin {plugin_id} as {plugin_type}")
                    elif plugin_type == "tool" and isinstance(plugin, ToolPlugin):
                        plugin_list.append(plugin)
                        logger.info(f"Registered plugin {plugin_id} as {plugin_type}")
                    elif plugin_type == "adapter" and isinstance(plugin, AdapterPlugin):
                        plugin_list.append(plugin)
                        logger.info(f"Registered plugin {plugin_id} as {plugin_type}")
                
                logger.info(f"Loaded plugin: {plugin_id}")
                return plugin
            
            logger.error(f"Plugin {plugin_id} does not define a Plugin class")
            return None
        
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_id}: {e}")
            return None
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin.
        
        Args:
            plugin_id: ID of the plugin to unload
            
        Returns:
            True if unloading was successful, False otherwise
        """
        logger.info(f"Unloading plugin: {plugin_id}")
        
        # Check if the plugin is loaded
        if plugin_id not in self.plugins:
            logger.warning(f"Plugin {plugin_id} is not loaded")
            return False
        
        plugin = self.plugins[plugin_id]
        
        # Clean up the plugin
        if not plugin.cleanup():
            logger.warning(f"Failed to clean up plugin {plugin_id}")
        
        # Remove from plugins by type
        for plugin_list in self.plugins_by_type.values():
            if plugin in plugin_list:
                plugin_list.remove(plugin)
        
        # Remove from plugins
        del self.plugins[plugin_id]
        
        logger.info(f"Unloaded plugin: {plugin_id}")
        return True
    
    def run_preprocessors(self, prompt: str, context: Dict[str, Any]) -> str:
        """Run all preprocessor plugins on a prompt.
        
        Args:
            prompt: The prompt to process
            context: Additional context
            
        Returns:
            Processed prompt
        """
        if not self.enabled:
            return prompt
        
        logger.debug("Running preprocessor plugins")
        result = prompt
        
        for plugin in self.plugins_by_type["preprocessor"]:
            if not plugin.enabled:
                continue
            
            try:
                result = plugin.process_prompt(result, context)
                logger.debug(f"Preprocessor {plugin.name} processed prompt")
            except Exception as e:
                logger.error(f"Error in preprocessor {plugin.name}: {e}")
        
        return result
    
    def run_postprocessors(self, completion: str, context: Dict[str, Any]) -> str:
        """Run all postprocessor plugins on a completion.
        
        Args:
            completion: The completion to process
            context: Additional context
            
        Returns:
            Processed completion
        """
        if not self.enabled:
            return completion
        
        logger.debug("Running postprocessor plugins")
        result = completion
        
        for plugin in self.plugins_by_type["postprocessor"]:
            if not plugin.enabled:
                continue
            
            try:
                result = plugin.process_completion(result, context)
                logger.debug(f"Postprocessor {plugin.name} processed completion")
            except Exception as e:
                logger.error(f"Error in postprocessor {plugin.name}: {e}")
        
        return result
    
    def format_text(self, text: str, format_type: str, context: Dict[str, Any]) -> str:
        """Format text using formatter plugins.
        
        Args:
            text: The text to format
            format_type: The type of formatting to apply
            context: Additional context
            
        Returns:
            Formatted text
        """
        if not self.enabled:
            return text
        
        logger.debug(f"Formatting text as {format_type}")
        result = text
        
        for plugin in self.plugins_by_type["formatter"]:
            if not plugin.enabled:
                continue
            
            try:
                result = plugin.format_text(result, format_type, context)
                logger.debug(f"Formatter {plugin.name} formatted text")
            except Exception as e:
                logger.error(f"Error in formatter {plugin.name}: {e}")
        
        return result
    
    def get_commands(self) -> Dict[str, Tuple[CommandPlugin, Callable]]:
        """Get all commands from command plugins.
        
        Returns:
            Dictionary mapping command names to (plugin, function) tuples
        """
        if not self.enabled:
            return {}
        
        logger.debug("Getting commands from plugins")
        commands = {}
        
        for plugin in self.plugins_by_type["command"]:
            if not plugin.enabled:
                continue
            
            try:
                plugin_commands = plugin.get_commands()
                for cmd_name, cmd_func in plugin_commands.items():
                    if cmd_name in commands:
                        logger.warning(f"Command {cmd_name} already exists, skipping")
                        continue
                    
                    commands[cmd_name] = (plugin, cmd_func)
                    logger.debug(f"Added command {cmd_name} from plugin {plugin.name}")
            except Exception as e:
                logger.error(f"Error getting commands from plugin {plugin.name}: {e}")
        
        return commands
    
    def execute_command(self, command: str, args: List[str], context: Dict[str, Any]) -> Any:
        """Execute a command.
        
        Args:
            command: The command to execute
            args: Command arguments
            context: Additional context
            
        Returns:
            Command result
        """
        if not self.enabled:
            raise ValueError("Plugin system is disabled")
        
        logger.debug(f"Executing command: {command}")
        
        commands = self.get_commands()
        if command not in commands:
            raise ValueError(f"Command {command} not found")
        
        plugin, func = commands[command]
        try:
            return plugin.execute_command(command, args, context)
        except Exception as e:
            logger.error(f"Error executing command {command} from plugin {plugin.name}: {e}")
            raise
    
    def get_tools(self) -> Dict[str, Tuple[ToolPlugin, Dict[str, Any]]]:
        """Get all tools from tool plugins.
        
        Returns:
            Dictionary mapping tool names to (plugin, tool_definition) tuples
        """
        if not self.enabled:
            return {}
        
        logger.debug("Getting tools from plugins")
        tools = {}
        
        for plugin in self.plugins_by_type["tool"]:
            if not plugin.enabled:
                continue
            
            try:
                plugin_tools = plugin.get_tools()
                for tool_name, tool_def in plugin_tools.items():
                    if tool_name in tools:
                        logger.warning(f"Tool {tool_name} already exists, skipping")
                        continue
                    
                    tools[tool_name] = (plugin, tool_def)
                    logger.debug(f"Added tool {tool_name} from plugin {plugin.name}")
            except Exception as e:
                logger.error(f"Error getting tools from plugin {plugin.name}: {e}")
        
        return tools
    
    def execute_tool(self, tool: str, args: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute a tool.
        
        Args:
            tool: The tool to execute
            args: Tool arguments
            context: Additional context
            
        Returns:
            Tool result
        """
        if not self.enabled:
            raise ValueError("Plugin system is disabled")
        
        logger.debug(f"Executing tool: {tool}")
        
        tools = self.get_tools()
        if tool not in tools:
            raise ValueError(f"Tool {tool} not found")
        
        plugin, tool_def = tools[tool]
        try:
            return plugin.execute_tool(tool, args, context)
        except Exception as e:
            logger.error(f"Error executing tool {tool} from plugin {plugin.name}: {e}")
            raise
    
    def get_loaded_plugins(self) -> Dict[str, PluginBase]:
        """Get all loaded plugins.
        
        Returns:
            Dictionary mapping plugin IDs to plugin instances
        """
        return self.plugins.copy()
    
    def create_sample_plugin(self, plugin_type: str, name: str) -> bool:
        """Create a sample plugin file.
        
        Args:
            plugin_type: Type of plugin to create
            name: Name of the plugin
            
        Returns:
            True if the plugin was created successfully, False otherwise
        """
        if plugin_type not in PLUGIN_TYPES:
            logger.error(f"Invalid plugin type: {plugin_type}")
            return False
        
        # Create plugin file name (snake_case)
        file_name = name.lower().replace(" ", "_") + ".py"
        plugin_path = self.plugins_dir / file_name
        
        # Check if the file already exists
        if plugin_path.exists():
            logger.error(f"Plugin file {plugin_path} already exists")
            return False
        
        # Create plugin class name (PascalCase)
        class_name = "".join(word.capitalize() for word in name.split())
        
        # Create plugin file content
        content = f"""#!/usr/bin/env python3
\"\"\"
LlamaForge Plugin: {name}

This is a {plugin_type} plugin for LlamaForge.
{PLUGIN_TYPES[plugin_type]}
\"\"\"

from llamaforge.plugin_manager import {plugin_type.capitalize()}Plugin


def get_plugin_metadata():
    \"\"\"Get plugin metadata.
    
    Returns:
        Plugin metadata
    \"\"\"
    return {{
        "name": "{name}",
        "description": "A {plugin_type} plugin for LlamaForge",
        "version": "0.1.0",
        "author": "Your Name",
        "type": "{plugin_type}"
    }}


def get_plugin_type():
    \"\"\"Get plugin type.
    
    Returns:
        Plugin type
    \"\"\"
    return "{plugin_type}"


class Plugin({plugin_type.capitalize()}Plugin):
    \"\"\"A {plugin_type} plugin for LlamaForge.\"\"\"
"""
        
        # Add specific methods based on plugin type
        if plugin_type == "preprocessor":
            content += """
    def process_prompt(self, prompt, context):
        \"\"\"Process the prompt.
        
        Args:
            prompt: The prompt to process
            context: Additional context
            
        Returns:
            Processed prompt
        \"\"\"
        # Add your preprocessing logic here
        return prompt
"""
        elif plugin_type == "postprocessor":
            content += """
    def process_completion(self, completion, context):
        \"\"\"Process the completion.
        
        Args:
            completion: The completion to process
            context: Additional context
            
        Returns:
            Processed completion
        \"\"\"
        # Add your postprocessing logic here
        return completion
"""
        elif plugin_type == "formatter":
            content += """
    def format_text(self, text, format_type, context):
        \"\"\"Format the text.
        
        Args:
            text: The text to format
            format_type: The type of formatting to apply
            context: Additional context
            
        Returns:
            Formatted text
        \"\"\"
        # Add your formatting logic here
        return text
"""
        elif plugin_type == "command":
            content += """
    def get_commands(self):
        \"\"\"Get the commands provided by this plugin.
        
        Returns:
            Dictionary mapping command names to functions
        \"\"\"
        return {
            "example": self.example_command
        }
    
    def example_command(self, *args):
        \"\"\"Example command.
        
        Args:
            *args: Command arguments
            
        Returns:
            Command result
        \"\"\"
        return f"Example command called with args: {args}"
"""
        elif plugin_type == "tool":
            content += """
    def get_tools(self):
        \"\"\"Get the tools provided by this plugin.
        
        Returns:
            Dictionary mapping tool names to tool definitions
        \"\"\"
        return {
            "example_tool": {
                "name": "example_tool",
                "description": "An example tool",
                "parameters": {
                    "param1": {
                        "type": "string",
                        "description": "First parameter"
                    },
                    "param2": {
                        "type": "integer",
                        "description": "Second parameter"
                    }
                }
            }
        }
    
    def execute_tool(self, tool, args, context):
        \"\"\"Execute a tool.
        
        Args:
            tool: The tool to execute
            args: Tool arguments
            context: Additional context
            
        Returns:
            Tool result
        \"\"\"
        if tool == "example_tool":
            return f"Example tool called with args: {args}"
        else:
            raise ValueError(f"Unknown tool: {tool}")
"""
        elif plugin_type == "adapter":
            content += """
    def adapt_input(self, input_data, model_type, context):
        \"\"\"Adapt input data to the format expected by the model.
        
        Args:
            input_data: Input data to adapt
            model_type: Type of model
            context: Additional context
            
        Returns:
            Adapted input data
        \"\"\"
        # Add your input adaptation logic here
        return input_data
    
    def adapt_output(self, output_data, model_type, context):
        \"\"\"Adapt output data from the model to the format expected by the user.
        
        Args:
            output_data: Output data to adapt
            model_type: Type of model
            context: Additional context
            
        Returns:
            Adapted output data
        \"\"\"
        # Add your output adaptation logic here
        return output_data
"""
        
        try:
            # Create plugins directory if it doesn't exist
            self.plugins_dir.mkdir(parents=True, exist_ok=True)
            
            # Write plugin file
            with open(plugin_path, "w") as f:
                f.write(content)
            
            logger.info(f"Created sample {plugin_type} plugin at {plugin_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error creating sample plugin: {e}")
            return False 