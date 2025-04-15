"""
Tests for the Config class.
"""

import json
import os
import tempfile

import pytest

from llamaforge.config import Config


class TestConfig:
    """
    Tests for the Config class.
    """

    def test_init_default(self):
        """Test initialization with default values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch the default config dir
            original_default = Config.DEFAULT_CONFIG_DIR
            Config.DEFAULT_CONFIG_DIR = temp_dir

            try:
                config = Config()

                # Check that the config path is correct
                assert config.config_path == os.path.join(
                    temp_dir, Config.DEFAULT_CONFIG_FILE
                )

                # Check that the config file was created
                assert os.path.exists(config.config_path)

                # Check that the config data has default values
                assert config.config_data["default_backend"] == "llama_cpp"
                assert config.config_data["default_model"] is None
                assert "models" in config.config_data
                assert "plugins" in config.config_data
            finally:
                # Restore the default config dir
                Config.DEFAULT_CONFIG_DIR = original_default

    def test_init_custom_path(self):
        """Test initialization with a custom path."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            try:
                config = Config(config_path=temp_file.name)

                # Check that the config path is correct
                assert config.config_path == temp_file.name

                # Check that the config file was created
                assert os.path.exists(config.config_path)
            finally:
                # Clean up
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)

    def test_get_set(self):
        """Test get and set methods."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            try:
                config = Config(config_path=temp_file.name)

                # Test basic get/set
                config.set("test_key", "test_value")
                assert config.get("test_key") == "test_value"

                # Test get with default
                assert config.get("nonexistent_key", "default") == "default"

                # Test nested keys
                config.set("nested.key", "nested_value")
                assert config.get("nested.key") == "nested_value"

                # Check that the nested structure was created
                assert "nested" in config.config_data
                assert "key" in config.config_data["nested"]
                assert config.config_data["nested"]["key"] == "nested_value"
            finally:
                # Clean up
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)

    def test_save_load(self):
        """Test save and load methods."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            try:
                # Create and save a config
                config1 = Config(config_path=temp_file.name)
                config1.set("test_key", "test_value")
                config1.save()

                # Create a new config that loads from the same file
                config2 = Config(config_path=temp_file.name)

                # Check that the value was loaded
                assert config2.get("test_key") == "test_value"
            finally:
                # Clean up
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)

    def test_add_remove_model(self):
        """Test add_model and remove_model methods."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            try:
                config = Config(config_path=temp_file.name)

                # Test add_model
                config.add_model(
                    name="test_model",
                    path="/path/to/model",
                    backend="test_backend",
                    params={"param1": "value1"},
                )

                # Check that the model was added
                assert "test_model" in config.config_data["models"]
                assert (
                    config.config_data["models"]["test_model"]["path"]
                    == "/path/to/model"
                )
                assert (
                    config.config_data["models"]["test_model"]["backend"]
                    == "test_backend"
                )
                assert config.config_data["models"]["test_model"]["param1"] == "value1"

                # Test remove_model
                result = config.remove_model("test_model")
                assert result is True
                assert "test_model" not in config.config_data["models"]

                # Test removing a nonexistent model
                result = config.remove_model("nonexistent_model")
                assert result is False
            finally:
                # Clean up
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
