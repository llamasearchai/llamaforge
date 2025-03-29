#!/usr/bin/env python3
"""
LlamaForge: Ultimate Language Model Command Interface
This enhanced version includes interactive chat, text generation,
and modular backend support.
"""

import os
import sys
import json
import logging
import argparse
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("llamaforge")

# Try to import internal modules
try:
    from .model_manager import ModelManager
    from .config_wizard import ConfigWizard
    from .plugin_manager import PluginManager
    from .api_server import APIServer
    from .version import __version__
except ImportError:
    # When running as a script
    try:
        current_dir = Path(__file__).parent
        sys.path.append(str(current_dir.parent))
        
        from llamaforge.model_manager import ModelManager
        from llamaforge.config_wizard import ConfigWizard
        from llamaforge.plugin_manager import PluginManager
        from llamaforge.api_server import APIServer
        from llamaforge.version import __version__
    except ImportError:
        logger.error("Failed to import required modules. Please ensure LlamaForge is installed correctly.")
        __version__ = "unknown"


class LlamaForge:
    def __init__(self):
        self.config_path = Path.home() / ".llamaforge" / "config.json"
        self.config = self._load_config()
        self.model_manager = ModelManager(self.config)
        self.plugin_manager = PluginManager(self.config)
        self.api_server = None
        self.backend = self._load_backend(self.config.get("default_backend", "llama.cpp"))

    def _load_config(self) -> Dict:
        """Load the configuration from config.json."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded configuration from {self.config_path}")
                    return config
            else:
                logger.warning(f"Config file not found at {self.config_path}. Using default settings.")
                # Create default config
                wizard = ConfigWizard(self.config_path)
                return wizard.get_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _load_backend(self, backend_name: str) -> str:
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

    def generate_text(self, prompt: str, max_tokens=None, temperature=None, top_p=None, model_name=None):
        """Generate text based on a prompt."""
        # Default parameters from config
        max_tokens = max_tokens or self.config.get("default_max_tokens", 1024)
        temperature = temperature or self.config.get("default_temperature", 0.7)
        top_p = top_p or self.config.get("default_top_p", 0.9)
        model_name = model_name or self.config.get("default_model")
        
        # Get model info from registry
        model_info = None
        if model_name:
            # Try to find in registry first
            models = self.model_manager.list_models()
            for model in models:
                if model["id"] == model_name or model.get("name") == model_name:
                    model_info = model
                    break
            
            # If not found and looks like a HuggingFace model, try to download it
            if not model_info and "/" in model_name:
                logger.info(f"Model {model_name} not found in registry. Attempting to download...")
                if self.model_manager.download_from_huggingface(model_name, self.backend):
                    # Refresh models list
                    models = self.model_manager.list_models()
                    for model in models:
                        if model["id"] == model_name.replace("/", "_").lower():
                            model_info = model
                            break
        
        if not model_info:
            logger.error(f"Model {model_name} not found. Please download it first or specify a valid model.")
            return None
        
        logger.info(f"Generating text with model: {model_name}")
        logger.info(f"Parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
        
        # Apply preprocessor plugins
        context = {
            "model": model_info,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        processed_prompt = self.plugin_manager.run_preprocessors(prompt, context)
        
        # Placeholder for actual generation - in a real implementation, this would call the 
        # appropriate backend with the specified parameters
        print(f"\nGenerated text for prompt: '{processed_prompt}'")
        print("---\nThis is placeholder text from the LlamaForge text generation function.\n"
              "In a complete implementation, this would use the selected backend to generate actual text.\n---")
        
        # Dummy completion for now
        completion = "This is a placeholder response from LlamaForge. In a real implementation, this would be generated by the model."
        
        # Apply postprocessor plugins
        processed_completion = self.plugin_manager.run_postprocessors(completion, context)
        
        return processed_completion

    def chat(self):
        """Start an interactive chat session."""
        print("\nWelcome to LlamaForge Chat!")
        print(f"Using backend: {self.backend}")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'help' for commands.\n")
        
        history = []
        
        # Get available tools from plugins
        tools = self.plugin_manager.get_all_tools()
        commands = self.plugin_manager.get_all_commands()
        
        # Print available tools and commands
        if tools:
            print("\nAvailable tools:")
            for tool_name, tool_info in tools.items():
                print(f"  {tool_name}: {tool_info.get('description', '')}")
        
        if commands:
            print("\nAvailable commands:")
            for cmd_name, cmd_info in commands.items():
                print(f"  {cmd_name}: {cmd_info.get('description', '')}")
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit']:
                    print("Exiting chat session.")
                    break
                
                # Check for tool or command usage
                if user_input.startswith("/"):
                    # Command mode
                    parts = user_input[1:].split(maxsplit=1)
                    command = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    
                    if command == "help":
                        print("\nAvailable commands:")
                        print("  /exit, /quit - End the chat session")
                        print("  /help - Show this help message")
                        print("  /clear - Clear chat history")
                        print("  /models - List available models")
                        print("  /model <model_name> - Switch to a different model")
                        print("  /download <repo_id> - Download a model from HuggingFace")
                        print("  /plugins - List available plugins")
                        print("  /load <plugin_name> - Load a plugin")
                        
                        # Show custom commands from plugins
                        for cmd_name, cmd_info in commands.items():
                            print(f"  /{cmd_name} - {cmd_info.get('description', '')}")
                            
                        continue
                        
                    elif command == "clear":
                        history = []
                        print("Chat history cleared.")
                        continue
                        
                    elif command == "models":
                        models = self.model_manager.list_models()
                        if not models:
                            print("No models found in registry.")
                        else:
                            print("\nAvailable models:")
                            for model in models:
                                current = "*" if model["id"] == self.config.get("default_model") else " "
                                print(f"  [{current}] {model['id']} - {model.get('name', 'Unknown')}")
                        continue
                        
                    elif command == "model":
                        if not args:
                            print("Please specify a model name.")
                            continue
                            
                        model_name = args.strip()
                        models = self.model_manager.list_models()
                        found = False
                        for model in models:
                            if model["id"] == model_name:
                                self.config["default_model"] = model_name
                                print(f"Switched to model: {model_name}")
                                found = True
                                break
                                
                        if not found:
                            print(f"Model {model_name} not found. Use /models to see available models.")
                        continue
                        
                    elif command == "download":
                        if not args:
                            print("Please specify a model repository ID (e.g., TheBloke/Mistral-7B-Instruct-v0.1-GGUF).")
                            continue
                            
                        repo_id = args.strip()
                        print(f"Downloading model {repo_id} for {self.backend} backend...")
                        if self.model_manager.download_from_huggingface(repo_id, self.backend):
                            print(f"Model {repo_id} downloaded successfully.")
                        else:
                            print(f"Failed to download model {repo_id}.")
                        continue
                        
                    elif command == "plugins":
                        discovered = self.plugin_manager.discover_plugins()
                        loaded = self.plugin_manager.get_plugins()
                        
                        if not discovered["system"] and not discovered["user"]:
                            print("No plugins found.")
                        else:
                            print("\nSystem plugins:")
                            for plugin in discovered["system"]:
                                loaded_mark = "*" if plugin in loaded else " "
                                print(f"  [{loaded_mark}] {plugin}")
                                
                            print("\nUser plugins:")
                            for plugin in discovered["user"]:
                                loaded_mark = "*" if plugin in loaded else " "
                                print(f"  [{loaded_mark}] {plugin}")
                                
                            print("\n* = loaded")
                        continue
                        
                    elif command == "load":
                        if not args:
                            print("Please specify a plugin name.")
                            continue
                            
                        plugin_name = args.strip()
                        if self.plugin_manager.load_plugin(plugin_name):
                            print(f"Plugin {plugin_name} loaded successfully.")
                        else:
                            print(f"Failed to load plugin {plugin_name}.")
                        continue
                    
                    # Check for custom commands from plugins
                    elif command in commands:
                        cmd_args = args.split() if args else []
                        result = self.plugin_manager.execute_command(command, cmd_args, {"history": history})
                        if result:
                            print(f"\nCommand result: {result}")
                        continue
                        
                    else:
                        print(f"Unknown command: {command}. Type /help for available commands.")
                        continue
                
                # Regular chat mode - store message in history
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
                import traceback
                logger.error(traceback.format_exc())

    def benchmark(self, model_names=None, task=None):
        """Run benchmarks on specified models."""
        print("\nRunning benchmarks...")
        
        # Get models to benchmark
        models = []
        if model_names:
            # Use specified models
            for model_name in model_names.split(","):
                model_info = self.model_manager.get_model_info(model_name.strip())
                if model_info:
                    models.append(model_info)
                else:
                    logger.warning(f"Model {model_name} not found. Skipping.")
        else:
            # Use default model
            default_model = self.config.get("default_model")
            if default_model:
                model_info = self.model_manager.get_model_info(default_model)
                if model_info:
                    models.append(model_info)
        
        if not models:
            logger.error("No models found for benchmarking.")
            return
        
        task = task or "general"
        print(f"Task: {task}")
        print(f"Models: {', '.join(m['id'] for m in models)}")
        
        # Load benchmark data
        benchmark_data = []
        datasets_dir = Path(self.config.get("datasets_dir", Path.home() / ".llamaforge" / "datasets"))
        benchmark_file = datasets_dir / "sample_benchmark.json"
        
        if benchmark_file.exists():
            try:
                with open(benchmark_file, 'r') as f:
                    benchmark_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading benchmark data: {e}")
        
        if not benchmark_data:
            logger.error("No benchmark data found.")
            return
        
        # Run benchmark for each model
        for model in models:
            print(f"\nBenchmarking model: {model['id']}")
            
            correct = 0
            total = len(benchmark_data)
            
            for i, item in enumerate(benchmark_data):
                prompt = item.get("prompt", "")
                reference = item.get("reference", "")
                
                print(f"\nPrompt {i+1}/{total}: {prompt}")
                # In a real implementation, this would call the actual model
                # and evaluate the response against the reference
                print(f"Reference: {reference}")
                print("Response: <placeholder>")  # Would be actual model response
                
                # Dummy scoring for placeholder
                correct += 1
            
            accuracy = (correct / total) * 100 if total > 0 else 0
            print(f"\nResults for {model['id']}:")
            print(f"Accuracy: {accuracy:.2f}%")
        
        print("\nBenchmarking complete.")

    def finetune(self, model_name=None, dataset=None):
        """Fine-tune a model on a dataset."""
        # Get model to fine-tune
        model = model_name or self.config.get("default_model")
        if not model:
            logger.error("No model specified for fine-tuning.")
            return
            
        model_info = self.model_manager.get_model_info(model)
        if not model_info:
            logger.error(f"Model {model} not found in registry.")
            return
        
        # Verify dataset
        if not dataset:
            logger.error("No dataset specified for fine-tuning.")
            return
            
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            # Check in the datasets directory
            datasets_dir = Path(self.config.get("datasets_dir", Path.home() / ".llamaforge" / "datasets"))
            dataset_path = datasets_dir / dataset
            if not dataset_path.exists():
                logger.error(f"Dataset not found: {dataset}")
                return
        
        print(f"\nFine-tuning model: {model}")
        print(f"Dataset: {dataset_path}")
        
        # Load dataset
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
                print(f"Loaded {len(data)} training examples.")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return
        
        # In a real implementation, this would set up and run the fine-tuning process
        print("Fine-tuning functionality is a placeholder. This would start the training process.")
        print("In a complete implementation, this would use the appropriate backend to fine-tune the model.")
    
    def start_api_server(self, host=None, port=None):
        """Start the API server for LlamaForge."""
        if not self.config.get("enable_api", False):
            logger.error("API server is not enabled in the configuration.")
            return False
        
        host = host or self.config.get("api_host", "127.0.0.1")
        port = port or self.config.get("api_port", 8000)
        
        try:
            self.api_server = APIServer(self.config)
            print(f"Starting API server on http://{host}:{port}")
            print("Press Ctrl+C to stop the server.")
            self.api_server.start(host=host, port=port)
            return True
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_config_wizard(self):
        """Run the configuration wizard."""
        wizard = ConfigWizard(self.config_path)
        if wizard.run_wizard():
            # Reload configuration
            self.config = self._load_config()
            return True
        return False


def create_parser():
    """Create the argument parser for the command-line interface."""
    parser = argparse.ArgumentParser(description="LlamaForge - Ultimate LM CLI")
    
    # Basic options
    parser.add_argument("--version", action="store_true", help="Show version information")
    parser.add_argument("--chat", action="store_true", help="Enter interactive chat mode")
    
    # Text generation options
    generation_group = parser.add_argument_group('Text Generation')
    generation_group.add_argument("--generate", type=str, help="Generate text from a prompt")
    generation_group.add_argument("--model", type=str, help="Model to use for generation")
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
    finetune_group.add_argument("--dataset", type=str, help="Dataset for fine-tuning")
    
    # Model management options
    model_group = parser.add_argument_group('Model Management')
    model_group.add_argument("--download", type=str, help="Download a model from HuggingFace")
    model_group.add_argument("--list-models", action="store_true", help="List available models")
    model_group.add_argument("--import-model", type=str, help="Import a local model file or directory")
    model_group.add_argument("--import-name", type=str, help="Name for the imported model")
    model_group.add_argument("--backend", type=str, help="Backend for the imported model")
    
    # Plugin options
    plugin_group = parser.add_argument_group('Plugins')
    plugin_group.add_argument("--list-plugins", action="store_true", help="List available plugins")
    plugin_group.add_argument("--load-plugin", type=str, help="Load a plugin")
    plugin_group.add_argument("--create-plugin", type=str, help="Create a sample plugin")
    plugin_group.add_argument("--plugin-type", type=str, help="Type of plugin to create")
    
    # API server options
    server_group = parser.add_argument_group('API Server')
    server_group.add_argument("--api-server", action="store_true", help="Start the API server")
    server_group.add_argument("--api-host", type=str, help="Host for the API server")
    server_group.add_argument("--api-port", type=int, help="Port for the API server")
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument("--config-wizard", action="store_true", help="Run the configuration wizard")
    
    return parser


def main():
    """Main entry point for LlamaForge CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Show version information
    if args.version:
        print(f"LlamaForge version: {__version__}")
        sys.exit(0)
    
    # Initialize LlamaForge
    forge = LlamaForge()
    
    # Configuration wizard
    if args.config_wizard:
        forge.run_config_wizard()
        sys.exit(0)
    
    # Model management
    if args.list_models:
        models = forge.model_manager.list_models()
        if not models:
            print("No models found in registry.")
        else:
            print("\nAvailable models:")
            for model in models:
                current = "*" if model["id"] == forge.config.get("default_model") else " "
                print(f"  [{current}] {model['id']} - {model.get('name', 'Unknown')}")
                print(f"      Backend: {model.get('backend', 'Unknown')}")
                print(f"      Path: {model.get('path', 'Unknown')}")
        sys.exit(0)
    
    if args.download:
        backend = args.backend or forge.backend
        print(f"Downloading model {args.download} for {backend} backend...")
        if forge.model_manager.download_from_huggingface(args.download, backend):
            print(f"Model {args.download} downloaded successfully.")
        else:
            print(f"Failed to download model {args.download}.")
        sys.exit(0)
    
    if args.import_model:
        if not args.import_name or not args.backend:
            print("Please specify --import-name and --backend for the imported model.")
            sys.exit(1)
        
        print(f"Importing model from {args.import_model} as {args.import_name} for {args.backend} backend...")
        if forge.model_manager.import_local_model(
            Path(args.import_model), args.import_name, args.backend
        ):
            print(f"Model {args.import_name} imported successfully.")
        else:
            print(f"Failed to import model from {args.import_model}.")
        sys.exit(0)
    
    # Plugin management
    if args.list_plugins:
        discovered = forge.plugin_manager.discover_plugins()
        loaded = forge.plugin_manager.get_plugins()
        
        if not discovered["system"] and not discovered["user"]:
            print("No plugins found.")
        else:
            print("\nSystem plugins:")
            for plugin in discovered["system"]:
                loaded_mark = "*" if plugin in loaded else " "
                print(f"  [{loaded_mark}] {plugin}")
                
            print("\nUser plugins:")
            for plugin in discovered["user"]:
                loaded_mark = "*" if plugin in loaded else " "
                print(f"  [{loaded_mark}] {plugin}")
                
            print("\n* = loaded")
        sys.exit(0)
    
    if args.load_plugin:
        if forge.plugin_manager.load_plugin(args.load_plugin):
            print(f"Plugin {args.load_plugin} loaded successfully.")
        else:
            print(f"Failed to load plugin {args.load_plugin}.")
        sys.exit(0)
    
    if args.create_plugin:
        if not args.plugin_type:
            print("Please specify --plugin-type for the plugin.")
            sys.exit(1)
        
        if forge.plugin_manager.create_sample_plugin(args.create_plugin, args.plugin_type):
            print(f"Sample {args.plugin_type} plugin '{args.create_plugin}' created successfully.")
        else:
            print(f"Failed to create sample plugin.")
        sys.exit(0)
    
    # API server
    if args.api_server:
        forge.start_api_server(args.api_host, args.api_port)
        sys.exit(0)
    
    # Main functionality
    if args.chat:
        forge.chat()
    elif args.generate:
        result = forge.generate_text(
            args.generate, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature,
            top_p=args.top_p,
            model_name=args.model
        )
        if result:
            print(result)
    elif args.benchmark:
        forge.benchmark(args.models, args.task)
    elif args.finetune:
        forge.finetune(args.model, args.dataset)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 