"""
Configuration management for LlamaForge.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

class Config:
    """
    Configuration management for LlamaForge.
    
    This class provides methods for loading, saving, and accessing
    configuration settings for the LlamaForge application.
    """
    
    DEFAULT_CONFIG_DIR = os.path.expanduser("~/.llamaforge")
    DEFAULT_CONFIG_FILE = "config.json"
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}
        
        # If no path specified, use default
        if not self.config_path:
            self.config_path = os.path.join(
                self.DEFAULT_CONFIG_DIR,
                self.DEFAULT_CONFIG_FILE
            )
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Load configuration
        self.load()
    
    def load(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            bool: True if configuration loaded successfully, False otherwise
        """
        if not os.path.exists(self.config_path):
            logger.debug(f"Configuration file not found at {self.config_path}")
            self._create_default_config()
            return True
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config_data = json.load(f)
            logger.debug(f"Configuration loaded from {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            self._create_default_config()
            return False
    
    def save(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            bool: True if configuration saved successfully, False otherwise
        """
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config_data, f, indent=2)
            logger.debug(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        # Handle nested keys (e.g., "models.default")
        if "." in key:
            parts = key.split(".")
            value = self.config_data
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        
        return self.config_data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        # Handle nested keys (e.g., "models.default")
        if "." in key:
            parts = key.split(".")
            config = self.config_data
            
            # Navigate to the deepest dict
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                elif not isinstance(config[part], dict):
                    config[part] = {}
                
                config = config[part]
            
            # Set value in the deepest dict
            config[parts[-1]] = value
        else:
            self.config_data[key] = value
    
    def _create_default_config(self) -> None:
        """
        Create default configuration.
        """
        self.config_data = {
            "default_backend": "llama_cpp",
            "default_model": None,
            "models": {},
            "model_directories": [
                os.path.expanduser("~/.llamaforge/models"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"),
            ],
            "plugins": [],
            "generation": {
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.1,
            },
            "server": {
                "host": "127.0.0.1",
                "port": 8000,
                "cors_origins": ["*"],
            },
            "tools": {},
            "commands": {},
        }
        
        logger.debug("Default configuration created")
    
    def add_model(
        self,
        name: str,
        path: str,
        backend: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a model to the configuration.
        
        Args:
            name: Model name
            path: Model path
            backend: Backend name
            params: Model parameters
        """
        if "models" not in self.config_data:
            self.config_data["models"] = {}
        
        self.config_data["models"][name] = {
            "path": path,
            "backend": backend,
            **(params or {}),
        }
        
        logger.debug(f"Model '{name}' added to configuration")
    
    def remove_model(self, name: str) -> bool:
        """
        Remove a model from the configuration.
        
        Args:
            name: Model name
            
        Returns:
            bool: True if model removed, False otherwise
        """
        if "models" in self.config_data and name in self.config_data["models"]:
            del self.config_data["models"][name]
            logger.debug(f"Model '{name}' removed from configuration")
            return True
        
        return False

    def list_models(self) -> List[str]:
        """
        List available models.
        
        Returns:
            List[str]: Names of available models
        """
        return list(self.config_data.get("models", {}).keys())
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Optional[Dict[str, Any]]: Model configuration or None if not found
        """
        return self.config_data.get("models", {}).get(model_name) 