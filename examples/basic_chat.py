#!/usr/bin/env python
"""
Basic chat example for LlamaForge.

This example demonstrates how to use LlamaForge for a simple chat interface
with a local model.
"""

import argparse
import logging
import os
import sys

# Add parent directory to path to allow importing llamaforge
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llamaforge import LlamaForge

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main():
    parser = argparse.ArgumentParser(description="Basic chat example for LlamaForge")
    parser.add_argument("--model", help="Model to use for chat")
    parser.add_argument("--backend", help="Backend to use for model")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for generation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Maximum tokens to generate"
    )
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()

    print("Initializing LlamaForge...")
    forge = LlamaForge(
        config_path=args.config, model_name=args.model, backend_name=args.backend
    )

    # Ensure model is loaded
    if not forge.current_model:
        print("No model loaded. Available models:")
        models = forge.list_models()
        if models:
            for model in models:
                print(f"  - {model}")
            print("\nPlease select a model with --model")
        else:
            print("No models found. Please configure a model first.")
        return

    print(f"Chat with {forge.current_model.name}. Type 'exit' to quit.")

    # Configuration
    generation_params = {"temperature": args.temperature, "max_tokens": args.max_tokens}

    # Chat history
    history = []

    # Chat loop
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break

            # Add to history and create prompt
            history.append(f"User: {user_input}")
            full_prompt = "\n".join(history) + "\nAssistant: "

            print("\nAssistant:", end="", flush=True)

            # Stream the response
            response_text = ""
            for chunk in forge.generate_stream(full_prompt, **generation_params):
                print(chunk, end="", flush=True)
                response_text += chunk

            # Add to history
            history.append(f"Assistant: {response_text}")

    except KeyboardInterrupt:
        print("\nExiting chat...")

    print("\nChat ended.")


if __name__ == "__main__":
    main()
