#!/usr/bin/env python3
"""
Example of creating a custom plugin for LlamaForge.

This example demonstrates how to create a simple preprocessor plugin
that enhances prompts by adding relevant context from a knowledge base.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("llamaforge_plugin_example")

# Import LlamaForge plugin base classes
try:
    from llamaforge.plugin_manager import PreprocessorPlugin, PluginBase
except ImportError:
    # Add parent directory to path for when running directly
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    try:
        from llamaforge.plugin_manager import PreprocessorPlugin, PluginBase
    except ImportError:
        logger.error("Could not import LlamaForge plugin classes.")
        logger.error("Make sure LlamaForge is installed or this script is run from the LlamaForge directory.")
        sys.exit(1)

class KnowledgeEnhancerPlugin(PreprocessorPlugin):
    """
    Preprocessor plugin that enhances prompts with relevant information 
    from a knowledge base.
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """Initialize the plugin with an optional path to a knowledge base file."""
        super().__init__(
            name="knowledge_enhancer",
            description="Enhances prompts with relevant information from a knowledge base"
        )
        self.knowledge_base = []
        
        # Default knowledge base path
        if knowledge_base_path is None:
            knowledge_base_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "knowledge_base.json"
            )
        
        # Load knowledge base if it exists
        if os.path.exists(knowledge_base_path):
            try:
                with open(knowledge_base_path, "r") as f:
                    self.knowledge_base = json.load(f)
                logger.info(f"Loaded knowledge base with {len(self.knowledge_base)} entries")
            except Exception as e:
                logger.error(f"Failed to load knowledge base: {e}")
        else:
            # Create a sample knowledge base
            self.knowledge_base = [
                {"keyword": "python", "context": "Python is a high-level, interpreted programming language known for its readability."},
                {"keyword": "llama", "context": "Llama is a family of large language models developed by Meta AI."},
                {"keyword": "transformer", "context": "Transformer is a deep learning architecture introduced in the paper 'Attention Is All You Need'."},
                {"keyword": "api", "context": "API (Application Programming Interface) is a set of rules that allow programs to communicate with each other."}
            ]
            
            # Save the sample knowledge base
            try:
                with open(knowledge_base_path, "w") as f:
                    json.dump(self.knowledge_base, f, indent=4)
                logger.info(f"Created sample knowledge base at {knowledge_base_path}")
            except Exception as e:
                logger.error(f"Failed to create sample knowledge base: {e}")
    
    def process(self, prompt: str) -> str:
        """
        Process the prompt by adding relevant knowledge from the knowledge base.
        
        Args:
            prompt: The input prompt to enhance
            
        Returns:
            The enhanced prompt with added context
        """
        # Skip empty prompts
        if not prompt or not self.knowledge_base:
            return prompt
        
        # Find relevant entries in the knowledge base
        relevant_contexts = []
        for entry in self.knowledge_base:
            if entry["keyword"].lower() in prompt.lower():
                relevant_contexts.append(entry["context"])
        
        # If no relevant context found, return the original prompt
        if not relevant_contexts:
            return prompt
        
        # Add the context to the prompt
        enhanced_prompt = "I have the following relevant information:\n"
        enhanced_prompt += "\n".join([f"- {context}" for context in relevant_contexts])
        enhanced_prompt += f"\n\nWith this context in mind, please respond to: {prompt}"
        
        return enhanced_prompt
    
    def cleanup(self) -> None:
        """Clean up resources when the plugin is unloaded."""
        logger.info("Cleaning up Knowledge Enhancer plugin")
        self.knowledge_base = []

# Example of how to use the plugin
def main():
    """Run the plugin example."""
    # Create the plugin
    plugin = KnowledgeEnhancerPlugin()
    
    # Information about the plugin
    print("\n🔌 LlamaForge Custom Plugin Example")
    print("-----------------------------------")
    print(f"Plugin Name: {plugin.name}")
    print(f"Plugin Description: {plugin.description}")
    print(f"Knowledge Base Entries: {len(plugin.knowledge_base)}")
    print("-----------------------------------\n")
    
    # Test the plugin with some example prompts
    example_prompts = [
        "What is Python used for in data science?",
        "Explain how Llama models work.",
        "Why are transformer architectures important in NLP?",
        "How can I use an API to integrate different services?",
        "What's the weather like today?" # No relevant knowledge
    ]
    
    for i, prompt in enumerate(example_prompts, 1):
        print(f"Example {i}:")
        print(f"Original Prompt: {prompt}")
        enhanced_prompt = plugin.process(prompt)
        print(f"Enhanced Prompt: {enhanced_prompt}")
        print()
    
    # Instructions for installing the plugin
    print("To install this plugin:")
    print("1. Save this file in the ~/.llamaforge/plugins/ directory")
    print("2. The plugin will be automatically loaded when LlamaForge starts")
    print("3. You can also load it programmatically:")
    print("   from llamaforge.plugin_manager import PluginManager")
    print("   plugin_manager = PluginManager()")
    print("   plugin_manager.load_plugin('/path/to/this/file')")

if __name__ == "__main__":
    main() 