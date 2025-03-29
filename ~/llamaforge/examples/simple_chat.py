#!/usr/bin/env python3
"""
Simple example of using LlamaForge for chat.

This example demonstrates how to use LlamaForge as a Python library
to chat with a language model.
"""

import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("llamaforge_example")

# Import LlamaForge classes
from llamaforge import LlamaForge

def main():
    """Run the simple chat example."""
    # Initialize LlamaForge
    logger.info("Initializing LlamaForge...")
    llamaforge = LlamaForge()
    
    # Get available models
    models = llamaforge.model_manager.list_models()
    
    if not models:
        logger.error("No models found. Please download a model first:")
        logger.error("  llamaforge model add --name TheBloke/Llama-2-7B-Chat-GGUF")
        return
    
    # Select first model
    model_name = list(models.keys())[0]
    logger.info(f"Using model: {model_name}")
    
    # Welcome message
    print("\n📝 Simple LlamaForge Chat Example")
    print("--------------------------------")
    print(f"Using model: {model_name}")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("--------------------------------\n")
    
    # Chat loop
    conversation_history = []
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if the user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Add to conversation history
        conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Generate response
            print("Model: ", end="", flush=True)
            response = ""
            
            for token in llamaforge.chat(
                model=model_name,
                messages=conversation_history,
                stream=True
            ):
                print(token, end="", flush=True)
                response += token
            
            print()  # Add newline after response
            
            # Add to conversation history
            conversation_history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            print("Sorry, there was an error generating a response.")

if __name__ == "__main__":
    main() 