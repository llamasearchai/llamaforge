#!/usr/bin/env python3
"""
LlamaForge: Ultimate Language Model Command Interface
This enhanced version includes interactive chat, text generation,
and modular backend support.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class LlamaForge:
    def __init__(self):
        self.config_path = Path.home() / ".llamaforge" / "config.json"
        self.config = self._load_config()
        self.backend = self._load_backend(
            self.config.get("default_backend", "llama.cpp")
        )

    def _load_config(self):
        """Load the configuration from config.json."""
        try:
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    return json.load(f)
            else:
                logging.warning(
                    f"Config file not found at {self.config_path}. Using default settings."
                )
                return {}
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return {}

    def _load_backend(self, backend_name):
        """Load the specified backend."""
        if backend_name == "llama.cpp":
            try:
                import llama_cpp

                logging.info("Using llama.cpp backend")
                return "llama.cpp"
            except ImportError:
                logging.warning(
                    "llama.cpp backend not available, falling back to transformers"
                )
                backend_name = "transformers"

        if backend_name == "mlx":
            try:
                import mlx

                logging.info("Using MLX backend")
                return "mlx"
            except ImportError:
                logging.warning(
                    "MLX backend not available, falling back to transformers"
                )
                backend_name = "transformers"

        if backend_name == "transformers":
            try:
                import transformers

                logging.info("Using transformers backend")
                return "transformers"
            except ImportError:
                logging.error(
                    "No available backends found. Please install at least one backend."
                )
                sys.exit(1)

        logging.error(
            f"Unknown backend '{backend_name}'. Using transformers as fallback."
        )
        return "transformers"

    def generate_text(self, prompt, max_tokens=None, temperature=None, top_p=None):
        """Generate text based on a prompt."""
        # Default parameters from config
        max_tokens = max_tokens or self.config.get("default_max_tokens", 1024)
        temperature = temperature or self.config.get("default_temperature", 0.7)
        top_p = top_p or self.config.get("default_top_p", 0.9)

        model_path = self.config.get("default_model")
        if not model_path:
            logging.error("No default model specified in configuration.")
            return None

        logging.info(f"Generating text with model: {model_path}")
        logging.info(
            f"Parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}"
        )

        # Placeholder for actual generation - in a real implementation, this would call the
        # appropriate backend with the specified parameters
        print(f"\nGenerated text for prompt: '{prompt}'")
        print(
            "---\nThis is placeholder text from the LlamaForge text generation function.\n"
            "In a complete implementation, this would use the selected backend to generate actual text.\n---"
        )

        return "Placeholder generated text"

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
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting chat session.")
                    break

                if user_input.lower() == "help":
                    print("\nAvailable commands:")
                    print("  exit, quit - End the chat session")
                    print("  help - Show this help message")
                    print("  clear - Clear chat history")
                    print("  models - List available models")
                    continue

                if user_input.lower() == "clear":
                    history = []
                    print("Chat history cleared.")
                    continue

                if user_input.lower() == "models":
                    print("\nAvailable models:")
                    print(
                        f"  Current: {self.config.get('default_model', 'No default model set')}"
                    )
                    print("  To use a different model, update your config.json file.")
                    continue

                # In a real implementation, history would be properly formatted and passed to the model
                history.append({"role": "user", "content": user_input})

                # Generate response (placeholder)
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
        models = (
            model_names.split(",")
            if model_names
            else [self.config.get("default_model")]
        )
        task = task or "general"

        print(f"Task: {task}")
        print(f"Models: {', '.join(models)}")

        # This would be implemented with actual benchmarking code
        print(
            "Benchmarking functionality is a placeholder. This would run performance tests."
        )

    def finetune(self, model_name=None, dataset=None):
        """Fine-tune a model on a dataset."""
        model = model_name or self.config.get("default_model")
        if not model:
            logging.error("No model specified for fine-tuning.")
            return

        if not dataset:
            logging.error("No dataset specified for fine-tuning.")
            return

        print(f"\nFine-tuning model: {model}")
        print(f"Dataset: {dataset}")

        # This would be implemented with actual fine-tuning code
        print(
            "Fine-tuning functionality is a placeholder. This would start the training process."
        )


def main():
    """Main entry point for LlamaForge CLI."""
    parser = argparse.ArgumentParser(description="LlamaForge - Ultimate LM CLI")

    # Basic options
    parser.add_argument(
        "--version", action="store_true", help="Show version information"
    )
    parser.add_argument(
        "--chat", action="store_true", help="Enter interactive chat mode"
    )

    # Text generation options
    generation_group = parser.add_argument_group("Text Generation")
    generation_group.add_argument(
        "--generate", type=str, help="Generate text from a prompt"
    )
    generation_group.add_argument(
        "--max-tokens", type=int, help="Maximum tokens to generate"
    )
    generation_group.add_argument(
        "--temperature", type=float, help="Sampling temperature"
    )
    generation_group.add_argument(
        "--top-p", type=float, help="Top-p (nucleus) sampling parameter"
    )

    # Benchmarking options
    benchmark_group = parser.add_argument_group("Benchmarking")
    benchmark_group.add_argument(
        "--benchmark", action="store_true", help="Run benchmarks"
    )
    benchmark_group.add_argument(
        "--models", type=str, help="Comma-separated list of models for benchmarking"
    )
    benchmark_group.add_argument("--task", type=str, help="Benchmark task")

    # Fine-tuning options
    finetune_group = parser.add_argument_group("Fine-tuning")
    finetune_group.add_argument(
        "--finetune", action="store_true", help="Fine-tune a model"
    )
    finetune_group.add_argument("--model", type=str, help="Model to fine-tune")
    finetune_group.add_argument("--dataset", type=str, help="Dataset for fine-tuning")

    args = parser.parse_args()

    if args.version:
        try:
            __version__ = "0.1.0"  # Hardcoded version as fallback
            print(f"LlamaForge version: {__version__}")
        except Exception:
            print("Version information not available")
            sys.exit(1)
        sys.exit(0)

    # Initialize LlamaForge
    forge = LlamaForge()

    if args.chat:
        forge.chat()
    elif args.generate:
        forge.generate_text(
            args.generate,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif args.benchmark:
        forge.benchmark(args.models, args.task)
    elif args.finetune:
        forge.finetune(args.model, args.dataset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
