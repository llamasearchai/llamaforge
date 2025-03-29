"""
Text cleaner postprocessing plugin.
"""

import re
from typing import Any, Dict, Optional

from ..base import BasePlugin

class TextCleanerPlugin(BasePlugin):
    """
    Plugin for cleaning text output.
    
    This plugin can:
    - Remove special tokens
    - Trim whitespace
    - Remove incomplete sentences
    - Clean up formatting
    """
    
    @property
    def name(self) -> str:
        """
        Get the name of the plugin.
        
        Returns:
            str: Plugin name
        """
        return "text_cleaner"
    
    @property
    def description(self) -> str:
        """
        Get the description of the plugin.
        
        Returns:
            str: Plugin description
        """
        return "Cleans and formats model output text"
    
    @property
    def supports_streaming(self) -> bool:
        """
        Check if the plugin supports streaming.
        
        Returns:
            bool: True if supports streaming, False otherwise
        """
        return True
    
    def process(self, text: str) -> str:
        """
        Clean the output text.
        
        Args:
            text: Output text
            
        Returns:
            str: Cleaned text
        """
        # Get configuration options
        trim_whitespace = self.get_config("trim_whitespace", True)
        remove_special_tokens = self.get_config("remove_special_tokens", True)
        remove_incomplete_sentences = self.get_config("remove_incomplete_sentences", False)
        
        # Process the text
        processed_text = text
        
        # Remove special tokens
        if remove_special_tokens:
            processed_text = self._remove_special_tokens(processed_text)
        
        # Trim whitespace
        if trim_whitespace:
            processed_text = processed_text.strip()
        
        # Remove incomplete sentences
        if remove_incomplete_sentences and not self.get_config("streaming", False):
            processed_text = self._remove_incomplete_sentences(processed_text)
        
        return processed_text
    
    def _remove_special_tokens(self, text: str) -> str:
        """
        Remove special tokens from text.
        
        Args:
            text: Input text
            
        Returns:
            str: Text without special tokens
        """
        # Common special tokens to remove
        patterns = [
            r"<\|im_end\|>.*",
            r"<\|im_start\|>assistant\s*",
            r"\[/INST\]",
            r"<s>",
            r"</s>",
            r"<pad>",
        ]
        
        result = text
        for pattern in patterns:
            result = re.sub(pattern, "", result)
        
        # Custom patterns from config
        custom_patterns = self.get_config("custom_token_patterns", [])
        for pattern in custom_patterns:
            result = re.sub(pattern, "", result)
        
        return result
    
    def _remove_incomplete_sentences(self, text: str) -> str:
        """
        Remove incomplete sentences at the end of text.
        
        Args:
            text: Input text
            
        Returns:
            str: Text without incomplete sentences
        """
        # Find the last complete sentence
        sentence_endings = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
        
        # Keep the text up to the last sentence ending
        last_ending_pos = -1
        for ending in sentence_endings:
            pos = text.rfind(ending)
            if pos > last_ending_pos:
                last_ending_pos = pos
        
        if last_ending_pos >= 0:
            return text[:last_ending_pos + 2].rstrip()
        
        return text 