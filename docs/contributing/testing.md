# Testing Guide

This guide provides instructions for testing the LlamaForge project, covering unit tests, integration tests, and benchmarks.

## Testing Philosophy

The LlamaForge project embraces test-driven development and aims for high test coverage. We believe that:

1. All core functionality should be tested
2. Tests should be fast and independent
3. Tests should be readable and maintainable
4. Tests should help document the expected behavior

## Test Setup

### Prerequisites

Before running tests, make sure you have the development environment set up:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Or with poetry
poetry install --with dev
```

### Test Configuration

The pytest configuration is in `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = --verbose --cov=llamaforge --cov-report=term --cov-report=xml
```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=llamaforge

# Generate a coverage report
pytest --cov=llamaforge --cov-report=html
```

### Running Specific Tests

```bash
# Run tests in a specific file
pytest tests/test_config.py

# Run a specific test
pytest tests/test_config.py::TestConfig::test_add_model

# Run tests matching a pattern
pytest -k "config or model"

# Run tests by marker
pytest -m "slow"
```

## Writing Tests

### Test Structure

Tests are organized in the `tests/` directory, mirroring the structure of the `llamaforge/` package:

```
tests/
├── __init__.py
├── test_config.py
├── test_model.py
├── backends/
│   ├── test_base.py
│   ├── test_llama_cpp.py
│   └── ...
├── plugins/
│   ├── preprocessing/
│   │   ├── test_text_formatter.py
│   │   └── ...
│   └── ...
└── ...
```

### Test Classes and Functions

We use class-based tests for related test cases:

```python
import pytest
from llamaforge.config import Config

class TestConfig:
    """Tests for the Config class."""
    
    def setup_method(self):
        """Setup before each test method."""
        self.config = Config(config_path=":memory:")  # Use in-memory config for tests
    
    def test_init(self):
        """Test initialization with default values."""
        assert self.config.config_path == ":memory:"
        assert isinstance(self.config.config, dict)
    
    def test_get_set(self):
        """Test get and set methods."""
        self.config.set("test_key", "test_value")
        assert self.config.get("test_key") == "test_value"
        
        # Test nested keys
        self.config.set("nested.key", "nested_value")
        assert self.config.get("nested.key") == "nested_value"
        
        # Test default value
        assert self.config.get("nonexistent", default="default") == "default"
```

### Test Fixtures

Use pytest fixtures for common test setup and resources:

```python
import pytest
from llamaforge.config import Config
from llamaforge.model import Model

@pytest.fixture
def config():
    """Create a test configuration."""
    config = Config(config_path=":memory:")
    config.set("models", {})
    config.set("default_model", "")
    return config

@pytest.fixture
def model(config):
    """Create a test model."""
    return Model(
        name="test-model",
        path="/path/to/test-model",
        config={"backend": "test_backend"}
    )

def test_model_properties(model):
    """Test model properties."""
    assert model.name == "test-model"
    assert model.path == "/path/to/test-model"
    assert model.backend == "test_backend"
```

### Parameterized Tests

Use parameterization to test multiple inputs:

```python
import pytest
from llamaforge.plugins.preprocessing.text_formatter import TextFormatterPlugin

@pytest.mark.parametrize("input_text,trim_whitespace,expected", [
    ("  hello  ", True, "hello"),
    ("  hello  ", False, "  hello  "),
    ("hello\nworld", True, "hello\nworld"),
])
def test_trim_whitespace(input_text, trim_whitespace, expected):
    """Test whitespace trimming."""
    formatter = TextFormatterPlugin()
    result = formatter.process(input_text, trim_whitespace=trim_whitespace)
    assert result == expected
```

### Mocking

Use mocking to isolate unit tests from external dependencies:

```python
import pytest
from unittest.mock import patch, MagicMock
from llamaforge.backends.llama_cpp import LlamaCppBackend

def test_generate():
    """Test text generation."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.generate.return_value = "Generated text"
    
    # Patch the llama_cpp.Llama class
    with patch("llama_cpp.Llama") as mock_llama:
        mock_llama.return_value = mock_model
        
        # Create the backend and test generation
        backend = LlamaCppBackend()
        backend.load_model("/path/to/model")
        result = backend.generate("Test prompt")
        
        # Assertions
        assert result == "Generated text"
        mock_model.generate.assert_called_once()
        args, kwargs = mock_model.generate.call_args
        assert "Test prompt" in kwargs.get("prompt", "")
```

### Testing Exceptions

Test that exceptions are raised when expected:

```python
import pytest
from llamaforge.exceptions import ModelNotFoundError
from llamaforge import LlamaForge

def test_load_nonexistent_model():
    """Test loading a non-existent model."""
    forge = LlamaForge()
    with pytest.raises(ModelNotFoundError) as excinfo:
        forge.load_model("nonexistent-model")
    assert "Model not found" in str(excinfo.value)
```

## Test Categories

### Unit Tests

Unit tests focus on testing individual components in isolation:

- Tests for the `Config` class
- Tests for the `Model` class
- Tests for each backend
- Tests for each plugin

### Integration Tests

Integration tests verify that components work together correctly:

```python
def test_end_to_end_generation():
    """Test end-to-end text generation workflow."""
    forge = LlamaForge()
    
    # Add and load a model
    forge.add_model("test-model", "/path/to/test/model")
    forge.load_model("test-model")
    
    # Enable plugins
    forge.enable_plugin("text_formatter")
    forge.enable_plugin("text_cleaner")
    
    # Generate text
    result = forge.generate("Test prompt")
    
    # Check the result (specific assertions would depend on the test setup)
    assert isinstance(result, str)
    assert len(result) > 0
```

### Performance Tests

Use benchmarks to test performance:

```python
import pytest
import time
from llamaforge import LlamaForge

@pytest.mark.benchmark
def test_generation_performance():
    """Benchmark text generation performance."""
    forge = LlamaForge()
    forge.add_model("test-model", "/path/to/test/model")
    forge.load_model("test-model")
    
    prompt = "Generate a response to this prompt."
    
    # Warmup
    forge.generate(prompt)
    
    # Benchmark
    start_time = time.time()
    for _ in range(10):
        forge.generate(prompt)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    print(f"Average generation time: {avg_time:.4f} seconds")
    
    # This is a benchmark, not a strict test, so we don't have assertions
    # In a CI environment, you might want to add assertions for regression testing
```

## Test Data

### Sample Models

For testing, we use smaller test models:

```python
@pytest.fixture
def tiny_model_path():
    """Path to a tiny test model."""
    return "tests/data/tiny-model.gguf"
```

### Sample Prompts

Create fixtures for sample prompts:

```python
@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "Hello, how are you?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about AI."
    ]
```

## Test Utilities

Create helper functions for common test operations:

```python
# tests/utils.py
import tempfile
import json
import os

def create_temp_config(config_data):
    """Create a temporary config file for testing."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, 'w') as f:
        json.dump(config_data, f)
    return path

def get_test_model_config():
    """Get a test model configuration."""
    return {
        "name": "test-model",
        "path": "/path/to/test-model",
        "backend": "llama.cpp",
        "parameters": {
            "n_gpu_layers": 0,
            "n_ctx": 512
        }
    }
```

## Continuous Integration

Tests are automatically run in CI when you push changes:

1. All pull requests must pass the test suite
2. Test coverage is tracked and reported
3. New features should maintain or improve coverage

## Test Coverage

We aim for high test coverage, but focus on testing the most critical parts:

```bash
# Check coverage
pytest --cov=llamaforge

# Generate a coverage report
pytest --cov=llamaforge --cov-report=html
```

Coverage reports help identify untested code. Open `htmlcov/index.html` to view detailed coverage information.

## Troubleshooting Tests

### Common Issues

#### Tests Failing Due to Missing Dependencies

```bash
pip install -e ".[dev]"
```

#### Tests Failing Due to Path Issues

Make sure you're running tests from the root directory of the project.

#### Tests Failing Due to Configuration Issues

Use in-memory or temporary configurations for tests to avoid affecting user configurations.

#### Tests Hanging or Taking Too Long

Add timeouts to tests:

```python
@pytest.mark.timeout(5)
def test_potentially_slow_function():
    """Test with a 5-second timeout."""
    result = potentially_slow_function()
    assert result == expected_result
```

## Best Practices

1. **Isolation**: Tests should not depend on each other or external state
2. **Speed**: Tests should run quickly to enable fast feedback
3. **Readability**: Test names and assertions should clearly indicate what's being tested
4. **Coverage**: Aim to test all code paths, especially error handling
5. **Maintainability**: Keep tests simple and DRY (Don't Repeat Yourself)

## Advanced Testing Techniques

### Property-Based Testing

Use hypothesis for property-based testing:

```python
import pytest
from hypothesis import given, strategies as st
from llamaforge.plugins.preprocessing.text_formatter import TextFormatterPlugin

@given(
    text=st.text(),
    trim_whitespace=st.booleans(),
    add_system_instruction=st.booleans(),
    system_instruction=st.text()
)
def test_text_formatter_properties(text, trim_whitespace, add_system_instruction, system_instruction):
    """Property-based test for TextFormatterPlugin."""
    formatter = TextFormatterPlugin()
    result = formatter.process(
        text, 
        trim_whitespace=trim_whitespace,
        add_system_instruction=add_system_instruction,
        system_instruction=system_instruction
    )
    
    # Properties that should always hold:
    if trim_whitespace:
        assert not (result.startswith(" ") or result.endswith(" "))
    
    if add_system_instruction and system_instruction:
        assert system_instruction in result
```

### Snapshot Testing

Test that output matches expected snapshots:

```python
import pytest
from llamaforge.plugins.preprocessing.text_formatter import TextFormatterPlugin

def test_format_as_chat_llama2(snapshot):
    """Test formatting text as Llama2 chat."""
    formatter = TextFormatterPlugin()
    input_text = "Tell me about quantum computing"
    result = formatter.process(
        input_text,
        format_as_chat=True,
        chat_format="llama2",
        system_instruction="You are a helpful assistant."
    )
    
    # Compare with snapshot
    snapshot.assert_match(result, "llama2_chat_format.txt")
```

### Testing with Real Models

For integration testing with real models:

```python
import pytest

# Mark tests that require real models
pytestmark = pytest.mark.real_models

@pytest.mark.skipif(
    not os.path.exists("/path/to/real/model.gguf"), 
    reason="Real model not available"
)
def test_with_real_model():
    """Test with a real model."""
    forge = LlamaForge()
    forge.add_model("real-model", "/path/to/real/model.gguf")
    forge.load_model("real-model")
    
    result = forge.generate("This is a test prompt")
    assert isinstance(result, str)
    assert len(result) > 0
```

## Conclusion

Testing is a crucial part of the LlamaForge development process. Well-written tests help ensure that the library functions correctly, provides a safety net for refactoring, and serves as executable documentation for how components should work.

By following the guidelines in this testing guide, you'll contribute to keeping LlamaForge reliable and maintainable. 