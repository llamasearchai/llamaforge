"""
Base plugin class for LlamaForge.
"""

import abc
from typing import Any, Dict, Optional

class BasePlugin(abc.ABC):
    """
    Abstract base class for LlamaForge plugins.
    
    All plugins must implement this interface to provide a consistent
    way to integrate with LlamaForge.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new BasePlugin instance.
        
        Args:
            config: Plugin configuration
        """
        self.config = config or {}
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Get the name of the plugin.
        
        Returns:
            str: Plugin name
        """
        pass
    
    @property
    @abc.abstractmethod
    def description(self) -> str:
        """
        Get the description of the plugin.
        
        Returns:
            str: Plugin description
        """
        pass
    
    @property
    def version(self) -> str:
        """
        Get the version of the plugin.
        
        Returns:
            str: Plugin version
        """
        return "0.1.0"
    
    @property
    def supports_streaming(self) -> bool:
        """
        Check if the plugin supports streaming.
        
        Returns:
            bool: True if supports streaming, False otherwise
        """
        return False
    
    @abc.abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process data with the plugin.
        
        Args:
            data: Data to process
            
        Returns:
            Any: Processed data
        """
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        return self.config.get(key, default) 