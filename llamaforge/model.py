"""
Model class implementation for LlamaForge.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any

class Model:
    """
    Represents a language model in LlamaForge.
    
    Attributes:
        name (str): Name of the model
        path (Optional[str]): Path to model files if local
        config (Dict[str, Any]): Model configuration
    """
    
    def __init__(
        self, 
        name: str, 
        path: Optional[str] = None, 
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize a new Model instance.
        
        Args:
            name: Name of the model
            path: Path to model files if local
            config: Model configuration
        """
        self.name = name
        self.path = path
        self.config = config or {}
    
    @property
    def is_local(self) -> bool:
        """
        Check if the model is local.
        
        Returns:
            bool: True if model is local, False otherwise
        """
        return self.path is not None and os.path.exists(self.path)
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters
        """
        return self.config.get("parameters", {})
    
    @property
    def backend_type(self) -> str:
        """
        Get the backend type for this model.
        
        Returns:
            str: Backend type
        """
        return self.config.get("backend", "llama.cpp")
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dict[str, Any]: Model metadata
        """
        return self.config.get("metadata", {})
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Get a specific model parameter.
        
        Args:
            key: Parameter key
            default: Default value if key not found
            
        Returns:
            Any: Parameter value
        """
        return self.parameters.get(key, default)
    
    def __repr__(self) -> str:
        """
        Get string representation of the model.
        
        Returns:
            str: String representation
        """
        return f"Model(name='{self.name}', backend='{self.backend_type}')" 