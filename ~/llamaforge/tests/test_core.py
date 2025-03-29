#!/usr/bin/env python3
"""
Core tests for LlamaForge.
"""

import unittest
import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llamaforge import __version__, LlamaForge
from llamaforge.model_manager import ModelManager
from llamaforge.config_wizard import ConfigWizard
from llamaforge.plugin_manager import PluginManager
from llamaforge.api_server import APIServer


class TestLlamaForgeCore(unittest.TestCase):
    """Test core functionality of LlamaForge."""

    def test_version(self):
        """Test that version is a string."""
        self.assertIsInstance(__version__, str)
        self.assertTrue(len(__version__) > 0)

    def test_imports(self):
        """Test that all modules can be imported."""
        self.assertIsNotNone(LlamaForge)
        self.assertIsNotNone(ModelManager)
        self.assertIsNotNone(ConfigWizard)
        self.assertIsNotNone(PluginManager)
        self.assertIsNotNone(APIServer)

    def test_llamaforge_init(self):
        """Test that LlamaForge can be initialized."""
        # Use a temporary config for testing
        test_config = {
            "dirs": {
                "models": os.path.expanduser("~/.llamaforge_test/models"),
                "plugins": os.path.expanduser("~/.llamaforge_test/plugins"),
                "cache": os.path.expanduser("~/.llamaforge_test/cache"),
                "logs": os.path.expanduser("~/.llamaforge_test/logs")
            },
            "model_defaults": {
                "backend": "llama.cpp",
                "context_length": 2048
            },
            "api_server": {
                "enabled": False,
                "host": "127.0.0.1",
                "port": 8000
            },
            "advanced": {
                "log_level": "INFO",
                "plugins_enabled": True
            }
        }
        
        # Mock the configuration loading
        LlamaForge._load_config = lambda self: test_config
        
        # Initialize LlamaForge
        llamaforge = LlamaForge()
        
        # Check that components are initialized
        self.assertIsNotNone(llamaforge.config)
        self.assertIsInstance(llamaforge.model_manager, ModelManager)
        self.assertIsInstance(llamaforge.plugin_manager, PluginManager)


if __name__ == "__main__":
    unittest.main() 