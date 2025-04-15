"""
Tests for the Model class.
"""

import os

import pytest

from llamaforge.model import Model


class TestModel:
    """
    Tests for the Model class.
    """
    
    def test_init(self):
        """Test basic initialization."""
        model = Model(
            name="test_model",
            path="/path/to/model",
            config={"param1": "value1"}
        )
        
        assert model.name == "test_model"
        assert model.path == "/path/to/model"
        assert model.config == {"param1": "value1"}
    
    def test_is_local(self):
        """Test is_local method."""
        # Local file
        model1 = Model(
            name="local_model",
            path="/path/to/model.gguf",
            config={}
        )
        assert model1.is_local() is True
        
        # Remote URL
        model2 = Model(
            name="remote_model",
            path="https://example.com/model.gguf",
            config={}
        )
        assert model2.is_local() is False
        
        # Model name (not a path)
        model3 = Model(
            name="huggingface_model",
            path="huggingface/llama-2-7b",
            config={}
        )
        assert model3.is_local() is False
    
    def test_get_parameters(self):
        """Test get_parameters method."""
        model = Model(
            name="test_model",
            path="/path/to/model",
            config={
                "parameters": {
                    "param1": "value1",
                    "param2": 2
                }
            }
        )
        
        # Get all parameters
        assert model.get_parameters() == {"param1": "value1", "param2": 2}
        
        # Get specific parameter
        assert model.get_parameters("param1") == "value1"
        
        # Get nonexistent parameter with default
        assert model.get_parameters("param3", "default") == "default"
    
    def test_get_param(self):
        """Test get_param method."""
        model = Model(
            name="test_model",
            path="/path/to/model",
            config={
                "param1": "value1",
                "nested": {
                    "param2": "value2"
                }
            }
        )
        
        # Get top-level parameter
        assert model.get_param("param1") == "value1"
        
        # Get nested parameter with dot notation
        assert model.get_param("nested.param2") == "value2"
        
        # Get nonexistent parameter with default
        assert model.get_param("param3", "default") == "default"
        assert model.get_param("nested.param3", "default") == "default"
    
    def test_metadata(self):
        """Test metadata-related methods."""
        model = Model(
            name="test_model",
            path="/path/to/model",
            config={
                "metadata": {
                    "author": "Test Author",
                    "version": "1.0.0",
                    "license": "MIT"
                }
            }
        )
        
        # Check all metadata
        assert model.metadata == {"author": "Test Author: Nik Jois
        
        # Check specific metadata
        assert model.get_metadata("author") == "Test Author"
        assert model.get_metadata("version") == "1.0.0"
        
        # Check nonexistent metadata with default
        assert model.get_metadata("description", "No description") == "No description" 