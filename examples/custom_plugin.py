#!/usr/bin/env python3
"""
Example of creating custom plugins for LlamaForge.

This example demonstrates how to create different types of plugins:
- Preprocessor: Modify prompts before sending to the model
- Postprocessor: Modify completions before returning to the user
- Formatter: Format model outputs in a specific way
- Command: Add custom commands to the CLI
- Tool: Add custom tools for chat mode
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llamaforge.plugin_manager import (
    CommandPlugin,
    FormatterPlugin,
    PluginBase,
    PostprocessorPlugin,
    PreprocessorPlugin,
    ToolPlugin,
)


class PromptEnhancerPlugin(PreprocessorPlugin):
    """Example preprocessor plugin that enhances prompts with additional context."""

    def __init__(self):
        super().__init__(
            name="prompt_enhancer",
            description="Enhances prompts with additional context and instructions",
        )
        self.enhancement_template = "Please provide a detailed, accurate, and helpful response to the following: {prompt}"

    def process(self, prompt: str) -> str:
        """Enhance the prompt with additional context."""
        return self.enhancement_template.format(prompt=prompt)


class MarkdownFormatterPlugin(PostprocessorPlugin):
    """Example postprocessor plugin that formats code blocks in markdown."""

    def __init__(self):
        super().__init__(
            name="markdown_formatter",
            description="Formats code blocks in markdown syntax",
        )
        self.code_pattern = re.compile(r"```(\w+)?\n(.*?)\n```", re.DOTALL)

    def process(self, completion: str) -> str:
        """Format code blocks in the completion."""

        def format_code_block(match):
            lang = match.group(1) or ""
            code = match.group(2)
            return f"```{lang}\n{code}\n```"

        return self.code_pattern.sub(format_code_block, completion)


class JSONFormatterPlugin(FormatterPlugin):
    """Example formatter plugin that formats output as JSON."""

    def __init__(self):
        super().__init__(name="json_formatter", description="Formats output as JSON")

    def format(self, text: str, format_args: Optional[Dict[str, Any]] = None) -> str:
        """Format the text as JSON."""
        try:
            # Try to parse as JSON first
            data = json.loads(text)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError:
            # If not valid JSON, create a simple JSON object
            return json.dumps({"text": text}, indent=2)


class WeatherCommandPlugin(CommandPlugin):
    """Example command plugin that simulates a weather lookup."""

    def __init__(self):
        super().__init__(
            name="weather",
            description="Get the weather for a location",
            usage="weather <location>",
        )

    def execute(self, args: List[str]) -> str:
        """Execute the weather command."""
        if not args:
            return "Error: Please provide a location. Usage: weather <location>"

        location = " ".join(args)

        # In a real plugin, this would call a weather API
        # Here we just return a simulated response
        return f"Weather for {location}: Sunny, 72°F (22°C), Wind: 5 mph"


class CalculatorToolPlugin(ToolPlugin):
    """Example tool plugin that provides a calculator in chat mode."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform calculations",
            usage="calculator <expression>",
        )

    def execute(self, args: List[str]) -> str:
        """Execute the calculator tool."""
        if not args:
            return "Error: Please provide an expression. Usage: calculator <expression>"

        expression = " ".join(args)

        try:
            # Use eval to calculate the result (in a real plugin, you'd want to use a safer method)
            # This is just for demonstration purposes
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"


def register_plugins():
    """Register all example plugins."""
    plugins = [
        PromptEnhancerPlugin(),
        MarkdownFormatterPlugin(),
        JSONFormatterPlugin(),
        WeatherCommandPlugin(),
        CalculatorToolPlugin(),
    ]

    # In a real plugin, you would save these to the plugins directory
    # Here we just print information about them
    print("Example Plugins:")
    for plugin in plugins:
        print(f"- {plugin.name}: {plugin.description}")

    return plugins


def main():
    """Run the example plugin demonstration."""
    print("LlamaForge Custom Plugin Example")
    print("================================")

    plugins = register_plugins()

    # Demonstrate each plugin
    print("\nDemonstrating plugins:")

    # Preprocessor
    prompt_enhancer = next(p for p in plugins if p.name == "prompt_enhancer")
    original_prompt = "What is the capital of France?"
    enhanced_prompt = prompt_enhancer.process(original_prompt)
    print(f"\n1. Prompt Enhancer Plugin:")
    print(f"   Original: {original_prompt}")
    print(f"   Enhanced: {enhanced_prompt}")

    # Postprocessor
    markdown_formatter = next(p for p in plugins if p.name == "markdown_formatter")
    original_completion = "Here's some code:\n```python\nprint('Hello, world!')\n```"
    formatted_completion = markdown_formatter.process(original_completion)
    print(f"\n2. Markdown Formatter Plugin:")
    print(f"   Original: {original_completion}")
    print(f"   Formatted: {formatted_completion}")

    # Formatter
    json_formatter = next(p for p in plugins if p.name == "json_formatter")
    text = "This is a simple text response"
    json_output = json_formatter.format(text)
    print(f"\n3. JSON Formatter Plugin:")
    print(f"   Original: {text}")
    print(f"   JSON: {json_output}")

    # Command
    weather_command = next(p for p in plugins if p.name == "weather")
    weather_result = weather_command.execute(["New York"])
    print(f"\n4. Weather Command Plugin:")
    print(f"   Command: weather New York")
    print(f"   Result: {weather_result}")

    # Tool
    calculator_tool = next(p for p in plugins if p.name == "calculator")
    calc_result = calculator_tool.execute(["2 + 2 * 10"])
    print(f"\n5. Calculator Tool Plugin:")
    print(f"   Command: calculator 2 + 2 * 10")
    print(f"   Result: {calc_result}")

    print("\nTo use these plugins in LlamaForge:")
    print("1. Save them to ~/.llamaforge/plugins/")
    print("2. Enable plugins in your config")
    print("3. Use them in chat mode or with the CLI")


if __name__ == "__main__":
    main()
