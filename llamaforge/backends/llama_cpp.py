"""
Llama.cpp backend implementation.
"""

import importlib
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..model import Model
from .base import BaseBackend

logger = logging.getLogger(__name__)


class LlamaCppBackend(BaseBackend):
    """
    Backend for llama.cpp models.

    This backend uses the llama-cpp-python package to run models locally.
    """

    def __init__(self) -> None:
        """
        Initialize a new LlamaCppBackend instance.
        """
        self._model = None
        self._llama_cpp = None
        self._current_model_info: Dict[str, Any] = {}

        # Try to import llama_cpp
        try:
            import llama_cpp

            self._llama_cpp = llama_cpp
            logger.info("llama-cpp-python package loaded successfully")
        except ImportError:
            logger.warning(
                "llama-cpp-python package not found. "
                "Models will not be loaded until the package is installed."
            )

    @property
    def name(self) -> str:
        """
        Get the name of the backend.

        Returns:
            str: Backend name
        """
        return "llama.cpp"

    def load_model(self, model: Model) -> None:
        """
        Load a model using llama.cpp.

        Args:
            model: Model to load

        Raises:
            RuntimeError: If llama-cpp-python is not installed or model loading fails
        """
        if self._llama_cpp is None:
            try:
                import llama_cpp

                self._llama_cpp = llama_cpp
                logger.info("llama-cpp-python package loaded successfully")
            except ImportError:
                raise RuntimeError(
                    "llama-cpp-python package is required to use the llama.cpp backend. "
                    "Install it with: pip install llama-cpp-python"
                )

        if not model.is_local:
            raise RuntimeError(
                f"Model {model.name} is not a local model, cannot load with llama.cpp"
            )

        logger.info(f"Loading model from {model.path}")

        # Get model parameters with fallbacks
        params = model.parameters
        n_ctx = params.get("n_ctx", 2048)
        n_batch = params.get("n_batch", 512)
        n_gpu_layers = params.get("n_gpu_layers", 0)

        # Load the model
        try:
            self._model = self._llama_cpp.Llama(
                model_path=model.path,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_gpu_layers=n_gpu_layers,
                use_mlock=params.get("use_mlock", False),
                seed=params.get("seed", -1),
                f16_kv=params.get("f16_kv", True),
            )

            self._current_model_info = {
                "name": model.name,
                "path": model.path,
                "parameters": {
                    "n_ctx": n_ctx,
                    "n_batch": n_batch,
                    "n_gpu_layers": n_gpu_layers,
                },
                "metadata": model.metadata,
            }

            logger.info(f"Model {model.name} loaded successfully with llama.cpp")
        except Exception as e:
            self._model = None
            logger.error(f"Failed to load model {model.name}: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text using the loaded llama.cpp model.

        Args:
            prompt: Input text prompt
            **kwargs: Additional parameters for text generation

        Returns:
            str: Generated text

        Raises:
            RuntimeError: If no model is loaded
        """
        if not self.is_model_loaded():
            raise RuntimeError("No model loaded")

        # Get generation parameters with fallbacks
        max_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k", 40)
        repeat_penalty = kwargs.get("repeat_penalty", 1.1)

        logger.debug(f"Generating with prompt: {prompt[:50]}...")

        try:
            result = self._model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=kwargs.get("stop", []),
                echo=kwargs.get("echo", False),
            )

            # Extract generated text from completion result
            generated_text = result["choices"][0]["text"]
            return generated_text

        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            raise RuntimeError(f"Text generation failed: {str(e)}")

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """
        Stream generated text using the loaded llama.cpp model.

        Args:
            prompt: Input text prompt
            **kwargs: Additional parameters for text generation

        Yields:
            str: Chunks of generated text

        Raises:
            RuntimeError: If no model is loaded
        """
        if not self.is_model_loaded():
            raise RuntimeError("No model loaded")

        # Get generation parameters with fallbacks
        max_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k", 40)
        repeat_penalty = kwargs.get("repeat_penalty", 1.1)

        logger.debug(f"Streaming generation with prompt: {prompt[:50]}...")

        try:
            for chunk in self._model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=kwargs.get("stop", []),
                echo=kwargs.get("echo", False),
                stream=True,
            ):
                # Extract generated text from completion chunk
                chunk_text = chunk["choices"][0]["text"]
                if chunk_text:
                    yield chunk_text

        except Exception as e:
            logger.error(f"Error during streaming text generation: {str(e)}")
            raise RuntimeError(f"Streaming text generation failed: {str(e)}")

    def is_model_loaded(self) -> bool:
        """
        Check if a model is loaded.

        Returns:
            bool: True if a model is loaded, False otherwise
        """
        return self._model is not None

    def unload_model(self) -> None:
        """
        Unload the current model.
        """
        if self._model is not None:
            logger.info("Unloading model")
            self._model = None
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
