"""
Commands plugins for LlamaForge.
"""

from typing import Dict, Type

from ..base import BasePlugin
from .benchmark import BenchmarkPlugin

# Registry of available commands
COMMANDS: Dict[str, Type[BasePlugin]] = {
    "benchmark": BenchmarkPlugin,
}

__all__ = ["COMMANDS", "BenchmarkPlugin"] 