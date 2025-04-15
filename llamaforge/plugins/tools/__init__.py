"""
Tools plugins for LlamaForge.
"""

from typing import Dict, Type

from ..base import BasePlugin
from .calculator import CalculatorPlugin

# Registry of available tools
TOOLS: Dict[str, Type[BasePlugin]] = {
    "calculator": CalculatorPlugin,
}

__all__ = ["TOOLS", "CalculatorPlugin"]
