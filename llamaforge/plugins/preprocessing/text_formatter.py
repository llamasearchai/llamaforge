"""
Text formatter preprocessing plugin.
"""

import re
from typing import Any, Dict, Optional

from ..base import BasePlugin


class TextFormatterPlugin(BasePlugin):
    """
    Plugin for formatting text input.

    This plugin can:
    - Trim whitespace
    - Add system instructions
    - Format as a chat prompt
    - Apply templates
    """

    @property
    def name(self) -> str:
        """
        Get the name of the plugin.

        Returns:
            str: Plugin name
        """
        return "text_formatter"

    @property
    def description(self) -> str:
        """
        Get the description of the plugin.

        Returns:
            str: Plugin description
        """
        return "Formats text input for model processing"

    def process(self, text: str) -> str:
        """
        Format the input text.

        Args:
            text: Input text

        Returns:
            str: Formatted text
        """
        # Get configuration options
        trim_whitespace = self.get_config("trim_whitespace", True)
        add_system_instruction = self.get_config("add_system_instruction", False)
        system_instruction = self.get_config("system_instruction", "")
        format_as_chat = self.get_config("format_as_chat", False)
        template = self.get_config("template", "")

        # Process the text
        processed_text = text

        # Trim whitespace
        if trim_whitespace:
            processed_text = processed_text.strip()

        # Apply template
        if template:
            processed_text = template.replace("{text}", processed_text)

        # Format as chat
        if format_as_chat:
            processed_text = self._format_as_chat(processed_text)

        # Add system instruction
        if add_system_instruction and system_instruction:
            if format_as_chat:
                processed_text = f"<s>[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n{processed_text}"
            else:
                processed_text = f"{system_instruction}\n\n{processed_text}"

        return processed_text

    def _format_as_chat(self, text: str) -> str:
        """
        Format text as a chat prompt.

        Args:
            text: Input text

        Returns:
            str: Chat-formatted text
        """
        chat_format = self.get_config("chat_format", "llama2")

        if chat_format == "llama2":
            # Format for Llama 2 chat models
            return f"<s>[INST] {text} [/INST]"
        elif chat_format == "chatml":
            # Format for ChatML
            return f"<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n"
        elif chat_format == "alpaca":
            # Format for Alpaca instruction format
            return f"### Instruction:\n{text}\n\n### Response:\n"
        else:
            # Default to no special formatting
            return text
