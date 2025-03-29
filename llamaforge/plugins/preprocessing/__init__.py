"""
Preprocessing plugins for LlamaForge.
"""

from typing import Dict, Type

from ..base import BasePlugin
from .text_formatter import TextFormatterPlugin

# Registry of available preprocessors
PREPROCESSORS: Dict[str, Type[BasePlugin]] = {
    "text_formatter": TextFormatterPlugin,
}

__all__ = ["PREPROCESSORS", "TextFormatterPlugin"] 