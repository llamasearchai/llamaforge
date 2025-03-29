"""
Core LlamaForge implementation.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Optional, Any, Union, Type, Callable, Iterator

from .config import Config
from .model import Model
from .backends import get_backend, BaseBackend
from .backends import BACKENDS

# Import plugin registries
try:
    from .plugins.preprocessing import PREPROCESSORS
    from .plugins.postprocessing import POSTPROCESSORS
    from .plugins.tools import TOOLS
    from .plugins.commands import COMMANDS
except ImportError:
    # Plugins might not be fully imported during early initialization
    PREPROCESSORS = {}
    POSTPROCESSORS = {}
    TOOLS = {}
    COMMANDS = {}

logger = logging.getLogger(__name__)

class LlamaForge:
    """
    Main LlamaForge class for managing language models.
    
    This class provides a unified interface for working with different
    language models from various providers.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: Optional[str] = None,
        backend_name: Optional[str] = None,
    ):
        """
        Initialize LlamaForge instance.
        
        Args:
            config_path: Path to configuration file
            model_name: Name of the model to load
            backend_name: Name of the backend to use
        """
        self.config = Config(config_path)
        self.current_model: Optional[Model] = None
        self.backend: Optional[BaseBackend] = None
        self.preprocessors = []
        self.postprocessors = []
        
        # Load preprocessors and postprocessors
        self._load_plugins()
        
        # Load model if specified
        if model_name or self.config.get("default_model"):
            self.load_model(model_name or self.config.get("default_model"), backend_name)
    
    def _load_plugins(self) -> None:
        """
        Load preprocessor and postprocessor plugins from configuration.
        """
        plugins_config = self.config.get("plugins", [])
        
        for plugin_config in plugins_config:
            plugin_type = plugin_config.get("type")
            plugin_name = plugin_config.get("name")
            plugin_config_data = plugin_config.get("config", {})
            
            if not plugin_type or not plugin_name:
                logger.warning(f"Invalid plugin configuration: {plugin_config}")
                continue
            
            try:
                if plugin_type == "preprocessing" and plugin_name in PREPROCESSORS:
                    plugin_class = PREPROCESSORS[plugin_name]
                    plugin = plugin_class()
                    plugin.set_config(plugin_config_data)
                    self.preprocessors.append(plugin)
                    logger.debug(f"Loaded preprocessing plugin: {plugin_name}")
                
                elif plugin_type == "postprocessing" and plugin_name in POSTPROCESSORS:
                    plugin_class = POSTPROCESSORS[plugin_name]
                    plugin = plugin_class()
                    plugin.set_config(plugin_config_data)
                    self.postprocessors.append(plugin)
                    logger.debug(f"Loaded postprocessing plugin: {plugin_name}")
                
                else:
                    logger.warning(f"Unknown plugin type or name: {plugin_type}/{plugin_name}")
            
            except Exception as e:
                logger.error(f"Error loading plugin {plugin_name}: {str(e)}")
    
    def load_model(self, model_name: str, backend_name: Optional[str] = None) -> bool:
        """
        Load a model with the specified name.
        
        Args:
            model_name: Name of the model to load
            backend_name: Name of the backend to use
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Get model path and info
            models_info = self.config.get("models", {})
            model_info = models_info.get(model_name, {})
            
            if not model_info and not os.path.exists(model_name):
                logger.error(f"Model '{model_name}' not found in configuration or file system")
                return False
            
            # Get model path
            model_path = model_info.get("path", model_name)
            
            # Determine backend
            if backend_name is None:
                # Try to get backend from model info
                backend_name = model_info.get("backend")
                
                # If not specified, try to infer from model path or use default
                if backend_name is None:
                    if model_path.endswith((".bin", ".gguf")):
                        backend_name = "llama_cpp"
                    elif any(x in model_path for x in ["huggingface", "hf"]):
                        backend_name = "huggingface"
                    elif any(x in model_path for x in ["openai", "gpt"]):
                        backend_name = "openai_api"
                    else:
                        backend_name = self.config.get("default_backend", "llama_cpp")
            
            # Create model
            model = Model(
                name=model_name,
                path=model_path,
                config=model_info
            )
            
            # Get backend
            backend = get_backend(backend_name)
            if backend is None:
                logger.error(f"Backend '{backend_name}' not available")
                return False
            
            # Load model with backend
            try:
                backend.load_model(model)
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                return False
            
            # Update current model and backend
            self.current_model = model
            self.backend = backend
            
            logger.info(f"Loaded model '{model_name}' with {backend_name} backend")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def unload_model(self) -> bool:
        """
        Unload the currently loaded model.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.backend and self.current_model:
            try:
                self.backend.unload_model()
                self.current_model = None
                logger.info("Model unloaded")
                return True
            except Exception as e:
                logger.error(f"Error unloading model: {str(e)}")
                return False
        return True  # No model to unload
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from the specified prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text
            
        Raises:
            ValueError: If no model is loaded
        """
        if not self.backend or not self.current_model:
            raise ValueError("No model loaded")
        
        # Apply preprocessing
        processed_prompt = self._preprocess(prompt)
        
        # Generate text
        response = self.backend.generate(processed_prompt, **kwargs)
        
        # Apply postprocessing
        processed_response = self._postprocess(response)
        
        return processed_response
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Generate text from the specified prompt in streaming mode.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Yields:
            str: Generated text chunks
            
        Raises:
            ValueError: If no model is loaded
        """
        if not self.backend or not self.current_model:
            raise ValueError("No model loaded")
        
        # Apply preprocessing
        processed_prompt = self._preprocess(prompt)
        
        # Get streaming parameters
        post_streamer = None
        supports_streaming = all(
            getattr(p, "supports_streaming", False) for p in self.postprocessors
        )
        
        if not supports_streaming and self.postprocessors:
            # If postprocessors don't support streaming, buffer and process at the end
            buffer = []
            
            def post_streamer(chunk: str) -> str:
                buffer.append(chunk)
                return chunk
        
        # Generate text in streaming mode
        response_stream = self.backend.generate_stream(processed_prompt, **kwargs)
        
        # Process stream
        for chunk in response_stream:
            if post_streamer:
                # Collect chunks for later processing
                post_streamer(chunk)
                yield chunk
            elif supports_streaming:
                # Process each chunk individually
                processed_chunk = self._postprocess_stream(chunk)
                yield processed_chunk
            else:
                # No processing
                yield chunk
        
        # Apply final postprocessing if needed
        if post_streamer and buffer:
            # The final postprocessed result isn't yielded, as we've already
            # sent the raw chunks. This is just for cleanup.
            self._postprocess("".join(buffer))
    
    def _preprocess(self, text: str) -> str:
        """
        Apply preprocessing to the input text.
        
        Args:
            text: Input text
            
        Returns:
            str: Preprocessed text
        """
        processed = text
        
        for preprocessor in self.preprocessors:
            try:
                processed = preprocessor.process(processed)
            except Exception as e:
                logger.error(f"Error in preprocessor {preprocessor.name}: {str(e)}")
        
        return processed
    
    def _postprocess(self, text: str) -> str:
        """
        Apply postprocessing to the output text.
        
        Args:
            text: Output text
            
        Returns:
            str: Postprocessed text
        """
        processed = text
        
        for postprocessor in self.postprocessors:
            try:
                processed = postprocessor.process(processed)
            except Exception as e:
                logger.error(f"Error in postprocessor {postprocessor.name}: {str(e)}")
        
        return processed
    
    def _postprocess_stream(self, chunk: str) -> str:
        """
        Apply streaming postprocessing to output chunks.
        
        Args:
            chunk: Output text chunk
            
        Returns:
            str: Postprocessed chunk
        """
        processed = chunk
        
        for postprocessor in self.postprocessors:
            if getattr(postprocessor, "supports_streaming", False):
                try:
                    processed = postprocessor.process(processed)
                except Exception as e:
                    logger.error(f"Error in streaming postprocessor {postprocessor.name}: {str(e)}")
        
        return processed
    
    def list_models(self) -> List[str]:
        """
        List available models.
        
        Returns:
            List[str]: List of model names
        """
        # Get models from configuration
        models_info = self.config.get("models", {})
        model_names = list(models_info.keys())
        
        # Add models from model directories
        model_dirs = self.config.get("model_directories", [])
        for model_dir in model_dirs:
            if os.path.isdir(model_dir):
                for file in os.listdir(model_dir):
                    path = os.path.join(model_dir, file)
                    if os.path.isfile(path) and any(path.endswith(ext) for ext in [".bin", ".gguf", ".ggml"]):
                        model_names.append(file)
        
        return model_names
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Model information
        """
        models_info = self.config.get("models", {})
        return models_info.get(model_name, {"name": model_name})
    
    def use_tool(self, tool_name: str, data: Any) -> Any:
        """
        Use a tool plugin.
        
        Args:
            tool_name: Name of the tool to use
            data: Input data for the tool
            
        Returns:
            Any: Tool output
            
        Raises:
            ValueError: If tool is not found
        """
        if tool_name not in TOOLS:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        tool_class = TOOLS[tool_name]
        tool = tool_class()
        
        # Get tool configuration
        tools_config = self.config.get("tools", {})
        tool_config = tools_config.get(tool_name, {})
        tool.set_config(tool_config)
        
        # Process data
        return tool.process(data)
    
    def run_command(self, command_name: str, data: Any) -> Any:
        """
        Run a command plugin.
        
        Args:
            command_name: Name of the command to run
            data: Input data for the command
            
        Returns:
            Any: Command output
            
        Raises:
            ValueError: If command is not found
        """
        if command_name not in COMMANDS:
            raise ValueError(f"Command '{command_name}' not found")
        
        command_class = COMMANDS[command_name]
        command = command_class()
        
        # Get command configuration
        commands_config = self.config.get("commands", {})
        command_config = commands_config.get(command_name, {})
        command.set_config(command_config)
        
        # Add forge instance to data
        if isinstance(data, dict):
            data["forge"] = self
        
        # Process data
        return command.process(data) 