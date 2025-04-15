"""
OpenAI API backend implementation.
"""

import logging
from typing import Any, Dict, Iterator, Optional

from ..model import Model
from .base import BaseBackend

logger = logging.getLogger(__name__)


class OpenAIBackend(BaseBackend):
    """
    Backend for OpenAI API models.

    This backend connects to the OpenAI API to generate text.
    """

    def __init__(self) -> None:
        """
        Initialize a new OpenAIBackend instance.
        """
        self._model_name = None
        self._api_key = None
        self._openai = None
        self._current_model_info: Dict[str, Any] = {}

        # Try to import openai
        try:
            import openai

            self._openai = openai
            logger.info("openai package loaded successfully")
        except ImportError:
            logger.warning(
                "openai package not found. "
                "Models will not be loaded until the package is installed."
            )

    @property
    def name(self) -> str:
        """
        Get the name of the backend.

        Returns:
            str: Backend name
        """
        return "openai"

    def load_model(self, model: Model) -> None:
        """
        Setup connection for OpenAI API.

        Args:
            model: Model to use

        Raises:
            RuntimeError: If openai package is not installed or setup fails
        """
        # This is a stub implementation
        logger.warning("OpenAIBackend is not fully implemented yet")
        raise NotImplementedError("OpenAIBackend is not fully implemented yet")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text using the OpenAI API.

        Args:
            prompt: Input text prompt
            **kwargs: Additional parameters for text generation

        Returns:
            str: Generated text

        Raises:
            RuntimeError: If API is not configured
        """
        # This is a stub implementation
        raise NotImplementedError("OpenAIBackend is not fully implemented yet")

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """
        Stream generated text using the OpenAI API.

        Args:
            prompt: Input text prompt
            **kwargs: Additional parameters for text generation

        Yields:
            str: Chunks of generated text

        Raises:
            RuntimeError: If API is not configured
        """
        # This is a stub implementation
        raise NotImplementedError("OpenAIBackend is not fully implemented yet")
        yield ""  # This is just to make the type checker happy

    def is_model_loaded(self) -> bool:
        """
        Check if API is configured.

        Returns:
            bool: True if API is configured, False otherwise
        """
        return self._model_name is not None and self._api_key is not None

    def unload_model(self) -> None:
        """
        Clear API configuration.
        """
        logger.info("Clearing OpenAI API configuration")
        self._model_name = None
        self._api_key = None
        self._current_model_info = {}

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured API model.

        Returns:
            Dict[str, Any]: Model information

        Raises:
            RuntimeError: If API is not configured
        """
        if not self.is_model_loaded():
            raise RuntimeError("No API model configured")

        return self._current_model_info
