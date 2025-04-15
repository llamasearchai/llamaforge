"""
Postprocessing plugins for LlamaForge.
"""

from typing import Dict, Type

from ..base import BasePlugin
from .text_cleaner import TextCleanerPlugin

# Registry of available postprocessors
POSTPROCESSORS: Dict[str, Type[BasePlugin]] = {
    "text_cleaner": TextCleanerPlugin,
}

__all__ = ["POSTPROCESSORS", "TextCleanerPlugin"]
