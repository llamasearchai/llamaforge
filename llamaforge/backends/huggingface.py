"""
HuggingFace backend implementation.
"""

import logging
from typing import Any, Dict, Iterator, Optional

from .base import BaseBackend
from ..model import Model

logger = logging.getLogger(__name__)

class HuggingFaceBackend(BaseBackend):
    """
    Backend for HuggingFace Transformers models.
    
    This backend uses the transformers package to run models locally.
    """
    
    def __init__(self) -> None:
        """
        Initialize a new HuggingFaceBackend instance.
        """
        self._model = None
        self._tokenizer = None
        self._transformers = None
        self._torch = None
        self._current_model_info: Dict[str, Any] = {}
        
        # Try to import transformers
        try:
            import transformers
            import torch
            self._transformers = transformers
            self._torch = torch
            logger.info("transformers package loaded successfully")
        except ImportError:
            logger.warning(
                "transformers package not found. "
                "Models will not be loaded until the package is installed."
            )
    
    @property
    def name(self) -> str:
        """
        Get the name of the backend.
        
        Returns:
            str: Backend name
        """
        return "huggingface"
    
    def load_model(self, model: Model) -> None:
        """
        Load a model using HuggingFace Transformers.
        
        Args:
            model: Model to load
            
        Raises:
            RuntimeError: If transformers is not installed or model loading fails
        """
        # This is a stub implementation
        logger.warning("HuggingFaceBackend is not fully implemented yet")
        raise NotImplementedError("HuggingFaceBackend is not fully implemented yet")
    
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text using the loaded HuggingFace model.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional parameters for text generation
            
        Returns:
            str: Generated text
            
        Raises:
            RuntimeError: If no model is loaded
        """
        # This is a stub implementation
        raise NotImplementedError("HuggingFaceBackend is not fully implemented yet")
    
    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """
        Stream generated text using the loaded HuggingFace model.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional parameters for text generation
            
        Yields:
            str: Chunks of generated text
            
        Raises:
            RuntimeError: If no model is loaded
        """
        # This is a stub implementation
        raise NotImplementedError("HuggingFaceBackend is not fully implemented yet")
        yield ""  # This is just to make the type checker happy
    
    def is_model_loaded(self) -> bool:
        """
        Check if a model is loaded.
        
        Returns:
            bool: True if a model is loaded, False otherwise
        """
        return self._model is not None and self._tokenizer is not None
    
    def unload_model(self) -> None:
        """
        Unload the current model.
        """
        if self._model is not None:
            logger.info("Unloading model")
            self._model = None
            self._tokenizer = None
            self._current_model_info = {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information
            
        Raises:
            RuntimeError: If no model is loaded
        """
        if not self.is_model_loaded():
            raise RuntimeError("No model loaded")
        
        return self._current_model_info 