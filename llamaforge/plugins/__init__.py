"""
Plugin system for LlamaForge.
"""

import logging
import importlib
from typing import Dict, List, Any, Optional

from .base import BasePlugin

logger = logging.getLogger(__name__)

def load_plugins(plugin_configs: List[Dict[str, Any]]) -> Dict[str, List[BasePlugin]]:
    """
    Load plugins from configuration.
    
    Args:
        plugin_configs: List of plugin configurations
        
    Returns:
        Dict[str, List[BasePlugin]]: Dictionary of loaded plugins by type
    """
    plugins: Dict[str, List[BasePlugin]] = {
        "preprocessing": [],
        "postprocessing": [],
        "tools": [],
        "commands": []
    }
    
    for plugin_config in plugin_configs:
        plugin_type = plugin_config.get("type")
        plugin_name = plugin_config.get("name")
        plugin_path = plugin_config.get("path")
        
        if not plugin_type or not plugin_name:
            logger.warning(f"Invalid plugin configuration: {plugin_config}")
            continue
        
        if plugin_type not in plugins:
            logger.warning(f"Unknown plugin type: {plugin_type}")
            continue
        
        try:
            # Try to load built-in plugin first
            plugin_class = _load_built_in_plugin(plugin_type, plugin_name)
            
            # If not built-in, try to load from custom path
            if plugin_class is None and plugin_path:
                plugin_class = _load_custom_plugin(plugin_path, plugin_name)
            
            # If still None, log error and continue
            if plugin_class is None:
                logger.error(f"Plugin not found: {plugin_name}")
                continue
            
            # Initialize plugin with config
            plugin = plugin_class(plugin_config.get("config", {}))
            plugins[plugin_type].append(plugin)
            logger.info(f"Loaded {plugin_type} plugin: {plugin_name}")
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {str(e)}")
    
    return plugins

def _load_built_in_plugin(plugin_type: str, plugin_name: str) -> Optional[type]:
    """
    Load a built-in plugin.
    
    Args:
        plugin_type: Type of plugin
        plugin_name: Name of plugin
        
    Returns:
        Optional[type]: Plugin class or None if not found
    """
    try:
        if plugin_type == "preprocessing":
            from .preprocessing import PREPROCESSORS
            return PREPROCESSORS.get(plugin_name)
        elif plugin_type == "postprocessing":
            from .postprocessing import POSTPROCESSORS
            return POSTPROCESSORS.get(plugin_name)
        elif plugin_type == "tools":
            from .tools import TOOLS
            return TOOLS.get(plugin_name)
        elif plugin_type == "commands":
            from .commands import COMMANDS
            return COMMANDS.get(plugin_name)
        return None
    except ImportError:
        logger.warning(f"Plugin module for {plugin_type} not found")
        return None

def _load_custom_plugin(plugin_path: str, plugin_name: str) -> Optional[type]:
    """
    Load a custom plugin from path.
    
    Args:
        plugin_path: Path to plugin module
        plugin_name: Name of plugin class
        
    Returns:
        Optional[type]: Plugin class or None if not found
    """
    try:
        module = importlib.import_module(plugin_path)
        return getattr(module, plugin_name, None)
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading custom plugin {plugin_name} from {plugin_path}: {str(e)}")
        return None

__all__ = ["BasePlugin", "load_plugins"] 