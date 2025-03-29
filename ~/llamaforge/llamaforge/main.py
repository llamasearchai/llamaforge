#!/usr/bin/env python3
"""
LlamaForge: Ultimate Language Model Command Interface

This module provides the main functionality for LlamaForge, including:
- Interactive chat mode
- Text generation
- Benchmarking
- Fine-tuning support
- Support for multiple backends (llama.cpp, MLX, transformers)
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
import time

# Import LlamaForge components
from .model_manager import ModelManager
from .config_wizard import ConfigWizard
from .plugin_manager import PluginManager
from .api_server import APIServer
from .version import __version__

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("llamaforge")

class LlamaForge:
    """Main LlamaForge class that provides all functionality."""
    
    def __init__(self):
        """Initialize LlamaForge with configuration."""
        self.config_path = Path.home() / ".llamaforge" / "config.json"
        self.config = self._load_config()
        
        # Initialize components
        self.model_manager = ModelManager(self.config)
        self.plugin_manager = PluginManager(self.config)
        self.api_server = None  # Initialized only when needed
        
        # Determine backend
        self.backend = self._load_backend(self.config.get("default_backend", "llama.cpp"))

    def _load_config(self):
        """Load the configuration from config.json."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            else:
                logger.warning(f"Config file not found at {self.config_path}. Using default settings.")
                # Create default configuration
                default_config = {
                    "directories": {
                        "models": os.path.expanduser("~/.llamaforge/models"),
                        "cache": os.path.expanduser("~/.llamaforge/cache"),
                        "logs": os.path.expanduser("~/.llamaforge/logs"),
                        "plugins": os.path.expanduser("~/.llamaforge/plugins")
                    },
                    "model_defaults": {
                        "context_length": 4096,
                        "max_tokens": 2048,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    },
                    "api_server": {
                        "enabled": False,
                        "host": "127.0.0.1",
                        "port": 8000
                    },
                    "advanced": {
                        "check_updates": True,
                        "telemetry": False,
                        "debug": False,
                        "backend_preference": []
                    }
                }
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _load_backend(self, backend_name):
        """Load the specified backend."""
        if backend_name == "llama.cpp":
            try:
                import llama_cpp
                logger.info("Using llama.cpp backend")
                return "llama.cpp"
            except ImportError:
                logger.warning("llama.cpp backend not available, falling back to transformers")
                backend_name = "transformers"
        
        if backend_name == "mlx":
            try:
                import mlx
                logger.info("Using MLX backend")
                return "mlx"
            except ImportError:
                logger.warning("MLX backend not available, falling back to transformers")
                backend_name = "transformers"
        
        if backend_name == "transformers":
            try:
                import transformers
                logger.info("Using transformers backend")
                return "transformers"
            except ImportError:
                logger.error("No available backends found. Please install at least one backend.")
                sys.exit(1)
        
        logger.error(f"Unknown backend '{backend_name}'. Using transformers as fallback.")
        return "transformers"

    def generate_text(self, prompt, max_tokens=None, temperature=None, top_p=None):
        """Generate text based on a prompt."""
        # Default parameters from config
        model_defaults = self.config.get("model_defaults", {})
        max_tokens = max_tokens or model_defaults.get("max_tokens", 1024)
        temperature = temperature or model_defaults.get("temperature", 0.7)
        top_p = top_p or model_defaults.get("top_p", 0.9)
        
        # Get default model if available
        model_id = self.config.get("default_model")
        if not model_id:
            models = self.model_manager.list_models()
            if models:
                model_id = models[0]["id"]
                logger.info(f"Using first available model: {model_id}")
            else:
                logger.error("No models available. Please download a model first.")
                return None
        
        logger.info(f"Generating text with model: {model_id}")
        logger.info(f"Parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
        
        # Apply preprocessor plugins
        processed_prompt = self.plugin_manager.run_preprocessors(prompt)
        
        # Placeholder for actual generation - in a real implementation, this would call the 
        # appropriate backend with the specified parameters
        # In a complete implementation, we would actually generate text here using the model
        response = f"This is a placeholder response from {model_id} using the {self.backend} backend."
        
        # Apply postprocessor plugins
        processed_response = self.plugin_manager.run_postprocessors(response, processed_prompt)
        
        return processed_response

    def chat(self):
        """Start an interactive chat session."""
        print("\nWelcome to LlamaForge Chat!")
        print(f"Using backend: {self.backend}")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'help' for commands.\n")
        
        history = []
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit']:
                    print("Exiting chat session.")
                    break
                
                if user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  exit, quit - End the chat session")
                    print("  help - Show this help message")
                    print("  clear - Clear chat history")
                    print("  models - List available models")
                    
                    # Get available plugins
                    commands = self.plugin_manager.get_available_commands()
                    if commands:
                        print("\nAvailable plugins:")
                        for name, info in commands.items():
                            print(f"  {name} - {info.get('description', 'No description')}")
                    
                    continue
                
                if user_input.lower() == 'clear':
                    history = []
                    print("Chat history cleared.")
                    continue
                
                if user_input.lower() == 'models':
                    models = self.model_manager.list_models()
                    if models:
                        print("\nAvailable models:")
                        for model in models:
                            print(f"  {model['id']} - {model.get('name', model['id'])}")
                    else:
                        print("\nNo models available. Use the model manager to download models.")
                    continue
                
                # In a real implementation, history would be properly formatted and passed to the model
                history.append({"role": "user", "content": user_input})
                
                # Generate response
                response = self.generate_text(user_input)
                print(f"\nLlamaForge: {response}")
                
                history.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                print("\nInterrupted. Exiting chat session.")
                break
            except Exception as e:
                print(f"\nError: {e}")

    def benchmark(self, model_names=None, task=None):
        """Run benchmarks on specified models."""
        print("\nRunning benchmarks...")
        models = model_names.split(',') if model_names else [self.config.get("default_model")]
        task = task or "general"
        
        print(f"Task: {task}")
        print(f"Models: {', '.join(models)}")
        
        # This would be implemented with actual benchmarking code
        print("Benchmarking functionality is a placeholder. This would run performance tests.")

    def finetune(self, model_name=None, dataset=None):
        """Fine-tune a model on a dataset."""
        model = model_name or self.config.get("default_model")
        if not model:
            logger.error("No model specified for fine-tuning.")
            return
            
        if not dataset:
            logger.error("No dataset specified for fine-tuning.")
            return
            
        print(f"\nFine-tuning model: {model}")
        print(f"Dataset: {dataset}")
        
        # This would be implemented with actual fine-tuning code
        print("Fine-tuning functionality is a placeholder. This would start the training process.")
    
    def start_api_server(self, host=None, port=None):
        """Start the API server."""
        api_config = self.config.get("api_server", {})
        host = host or api_config.get("host", "127.0.0.1")
        port = port or api_config.get("port", 8000)
        
        # Initialize API server if not already done
        if self.api_server is None:
            self.api_server = APIServer(self.config)
        
        print(f"\nStarting API server on {host}:{port}...")
        try:
            self.api_server.start(host=host, port=port)
            return True
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
            return False
    
    def run_config_wizard(self):
        """Run the configuration wizard."""
        wizard = ConfigWizard()
        return wizard.run_wizard()


def main():
    """Main entry point for LlamaForge CLI."""
    parser = argparse.ArgumentParser(description="LlamaForge - Ultimate LM CLI")
    
    # Basic options
    parser.add_argument("--version", action="store_true", help="Show version information")
    parser.add_argument("--chat", action="store_true", help="Enter interactive chat mode")
    
    # Text generation options
    generation_group = parser.add_argument_group('Text Generation')
    generation_group.add_argument("--generate", type=str, help="Generate text from a prompt")
    generation_group.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    generation_group.add_argument("--temperature", type=float, help="Sampling temperature")
    generation_group.add_argument("--top-p", type=float, help="Top-p (nucleus) sampling parameter")
    
    # Benchmarking options
    benchmark_group = parser.add_argument_group('Benchmarking')
    benchmark_group.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    benchmark_group.add_argument("--models", type=str, help="Comma-separated list of models for benchmarking")
    benchmark_group.add_argument("--task", type=str, help="Benchmark task")
    
    # Fine-tuning options
    finetune_group = parser.add_argument_group('Fine-tuning')
    finetune_group.add_argument("--finetune", action="store_true", help="Fine-tune a model")
    finetune_group.add_argument("--model", type=str, help="Model to fine-tune")
    finetune_group.add_argument("--dataset", type=str, help="Dataset for fine-tuning")
    
    # Model management options
    model_group = parser.add_argument_group('Model Management')
    model_group.add_argument("--download", type=str, help="Download a model from Hugging Face (repo_id)")
    model_group.add_argument("--list-models", action="store_true", help="List available models")
    model_group.add_argument("--import-model", type=str, help="Import a local model file or directory")
    
    # API server options
    api_group = parser.add_argument_group('API Server')
    api_group.add_argument("--api-server", action="store_true", help="Start the API server")
    api_group.add_argument("--host", type=str, help="API server host")
    api_group.add_argument("--port", type=int, help="API server port")
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument("--config-wizard", action="store_true", help="Run the configuration wizard")
    
    args = parser.parse_args()
    
    if args.version:
        print(f"LlamaForge version: {__version__}")
        sys.exit(0)
    
    # Initialize LlamaForge
    forge = LlamaForge()
    
    # Model management commands
    if args.list_models:
        models = forge.model_manager.list_models()
        if models:
            print("\nAvailable models:")
            for model in models:
                print(f"  {model['id']} - {model.get('name', model['id'])}")
        else:
            print("\nNo models available.")
        sys.exit(0)
    
    if args.download:
        print(f"Downloading model: {args.download}")
        model_id = forge.model_manager.download_model(args.download)
        if model_id:
            print(f"Successfully downloaded model: {model_id}")
        else:
            print("Failed to download model.")
        sys.exit(0)
    
    if args.import_model:
        print(f"Importing model: {args.import_model}")
        model_id = forge.model_manager.import_local_model(args.import_model)
        if model_id:
            print(f"Successfully imported model: {model_id}")
        else:
            print("Failed to import model.")
        sys.exit(0)
    
    # Configuration commands
    if args.config_wizard:
        print("Running configuration wizard...")
        forge.run_config_wizard()
        sys.exit(0)
    
    # API server commands
    if args.api_server:
        success = forge.start_api_server(host=args.host, port=args.port)
        sys.exit(0 if success else 1)
    
    # Main functionality
    if args.chat:
        forge.chat()
    elif args.generate:
        response = forge.generate_text(
            args.generate, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(f"\n{response}")
    elif args.benchmark:
        forge.benchmark(args.models, args.task)
    elif args.finetune:
        forge.finetune(args.model, args.dataset)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 