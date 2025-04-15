# Contributing Guidelines

Thank you for your interest in contributing to LlamaForge! This document provides guidelines for contributing to the project.

## Code of Conduct

We expect all contributors to adhere to a standard of respect and inclusivity. Be kind to one another, accept constructive feedback, and focus on what is best for the community.

## Ways to Contribute

There are many ways to contribute to LlamaForge:

1. **Report bugs**: Submit issues for any bugs you find
2. **Suggest features**: Submit issues for feature requests
3. **Improve documentation**: Fix typos, clarify explanations, add examples
4. **Write code**: Submit pull requests for bug fixes or new features
5. **Review code**: Review pull requests from other contributors

## Development Process

1. **Fork the repository**: Create your own fork of the repository
2. **Create a branch**: Create a branch for your changes
3. **Make your changes**: Implement your changes
4. **Run tests**: Make sure your changes pass the tests
5. **Submit a pull request**: Open a pull request to propose your changes

## Coding Standards

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style. In particular:

- Use 4 spaces for indentation
- Use snake_case for variable and function names
- Use CamelCase for class names
- Maximum line length of 88 characters (we use Black for formatting)

### Code Formatting

We use the following tools to maintain code quality:

- [Black](https://black.readthedocs.io/): Code formatter
- [isort](https://pycqa.github.io/isort/): Import sorter
- [mypy](http://mypy-lang.org/): Static type checker

To format your code:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black llamaforge tests
isort llamaforge tests
```

### Type Annotations

Use type annotations for all function arguments and return values:

```python
def add_numbers(a: int, b: int) -> int:
    return a + b
```

### Docstrings

Use Google-style docstrings:

```python
def add_model(
    self, 
    name: str, 
    path: str, 
    backend: str, 
    parameters: Optional[Dict[str, Any]] = None
) -> None:
    """Adds a new model to the configuration.
    
    Args:
        name: Name of the model
        path: Path to the model file or model identifier
        backend: Backend to use (llama.cpp, huggingface, or openai)
        parameters: Optional parameters for the model
        
    Raises:
        ValueError: If a model with the given name already exists
        
    Examples:
        >>> forge.add_model(
        ...     name="llama-2-7b",
        ...     path="/path/to/model.gguf",
        ...     backend="llama.cpp",
        ...     parameters={"temperature": 0.7}
        ... )
    """
```

## Testing

All code should be tested. We use [pytest](https://docs.pytest.org/) for testing:

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=llamaforge tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files starting with `test_`
- Name test functions starting with `test_`
- Test both success and failure cases

Example:

```python
def test_add_model():
    # Setup
    config = Config()
    
    # Test
    config.add_model("test-model", "path/to/model", "llama.cpp")
    
    # Verify
    assert "test-model" in config.get_models()
    
    # Test failure case
    with pytest.raises(ValueError):
        config.add_model("test-model", "path/to/model", "llama.cpp")
```

## Documentation

Documentation is a crucial part of LlamaForge. We use [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

To build and preview the documentation:

```bash
# Install mkdocs and theme
pip install mkdocs mkdocs-material

# Build and serve documentation
mkdocs serve
```

### Documentation Structure

- `docs/index.md`: Main page
- `docs/getting-started/`: Getting started guides
- `docs/guides/`: How-to guides for specific features
- `docs/api/`: API reference
- `docs/contributing/`: Contributing guidelines

## Pull Request Process

1. **Update tests**: Add or update tests for your changes
2. **Update documentation**: Update the documentation to reflect your changes
3. **Run CI checks**: Make sure your changes pass all CI checks
4. **Request review**: Request a review from a maintainer
5. **Address feedback**: Address any feedback from reviewers

## Issue Guidelines

When submitting an issue, please include:

### Bug Reports

- A clear description of the bug
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Your environment (operating system, Python version, etc.)

### Feature Requests

- A clear description of the feature
- Use cases for the feature
- Any relevant context or constraints

## Licensing

By contributing to LlamaForge, you agree that your contributions will be licensed under the project's MIT License. 