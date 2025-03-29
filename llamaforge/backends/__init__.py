"""
Backends for model inference in LlamaForge.
"""

import logging
from typing import Any, Dict, Type

from .base import BaseBackend
from .llama_cpp import LlamaCppBackend
from .huggingface import HuggingFaceBackend
from .openai_api import OpenAIBackend

logger = logging.getLogger(__name__)

# Registry of available backends
BACKENDS: Dict[str, Type[BaseBackend]] = {
    "llama.cpp": LlamaCppBackend,
    "huggingface": HuggingFaceBackend,
    "openai": OpenAIBackend,
}

def get_backend(backend_name: str) -> BaseBackend:
    """
    Get a backend instance by name.
    
    Args:
        backend_name: Name of the backend
        
    Returns:
        BaseBackend: Backend instance
        
    Raises:
        ValueError: If backend is not found
    """
    if backend_name not in BACKENDS:
        logger.error(f"Backend '{backend_name}' not found. Available backends: {list(BACKENDS.keys())}")
        raise ValueError(f"Backend '{backend_name}' not found")
    
    logger.info(f"Creating backend: {backend_name}")
    return BACKENDS[backend_name]()

__all__ = [
    "BaseBackend", 
    "LlamaCppBackend", 
    "HuggingFaceBackend", 
    "OpenAIBackend", 
    "get_backend",
    "BACKENDS"
] 