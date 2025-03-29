"""
Base backend class for model inference.
"""

import abc
from typing import Any, Dict, Iterator, Optional

from ..model import Model

class BaseBackend(abc.ABC):
    """
    Abstract base class for model backends.
    
    All model backends (e.g., llama.cpp, HuggingFace, OpenAI) must implement
    this interface to provide a consistent way to interact with different
    model implementations.
    """
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Get the name of the backend.
        
        Returns:
            str: Backend name
        """
        pass
    
    @abc.abstractmethod
    def load_model(self, model: Model) -> None:
        """
        Load a model.
        
        Args:
            model: Model to load
            
        Raises:
            RuntimeError: If model loading fails
        """
        pass
    
    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from the loaded model.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional parameters for text generation
            
        Returns:
            str: Generated text
            
        Raises:
            RuntimeError: If no model is loaded or generation fails
        """
        pass
    
    @abc.abstractmethod
    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """
        Stream generated text from the loaded model.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional parameters for text generation
            
        Yields:
            str: Chunks of generated text
            
        Raises:
            RuntimeError: If no model is loaded or generation fails
        """
        pass
    
    @abc.abstractmethod
    def is_model_loaded(self) -> bool:
        """
        Check if a model is loaded.
        
        Returns:
            bool: True if a model is loaded, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def unload_model(self) -> None:
        """
        Unload the current model.
        """
        pass
    
    @abc.abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information
            
        Raises:
            RuntimeError: If no model is loaded
        """
        pass 