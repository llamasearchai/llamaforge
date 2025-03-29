# Development Guide

This guide provides instructions for developers who want to contribute to the LlamaForge project.

## Setting Up Your Development Environment

### Requirements

- Python 3.8 or higher
- Git
- pip
- poetry (optional but recommended)

### Clone the Repository

```bash
git clone https://github.com/llamasearch/llamaforge.git
cd llamaforge
```

### Install Development Dependencies

Using pip:

```bash
pip install -e ".[dev]"
```

Using poetry:

```bash
poetry install --with dev
```

This installs LlamaForge in development mode along with all development dependencies.

## Project Structure

The LlamaForge project is organized as follows:

```
llamaforge/
├── llamaforge/            # Main package
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration management
│   ├── model.py           # Model class
│   ├── exceptions.py      # Custom exceptions
│   ├── server.py          # API server
│   ├── chat.py            # Chat interface 
│   ├── commands/          # CLI commands
│   ├── backends/          # Model backends
│   │   ├── __init__.py
│   │   ├── base.py        # Base backend class
│   │   ├── llama_cpp.py   # llama.cpp backend
│   │   ├── huggingface.py # Hugging Face backend
│   │   └── openai_api.py  # OpenAI API backend
│   └── plugins/           # Plugin system
│       ├── __init__.py
│       ├── base.py        # Base plugin class
│       ├── preprocessing/ # Preprocessing plugins
│       ├── postprocessing/# Postprocessing plugins
│       ├── tools/         # Tool plugins
│       └── commands/      # Command plugins
├── tests/                 # Unit tests
├── examples/              # Example scripts and notebooks
├── docs/                  # Documentation
├── .github/               # GitHub workflows
├── .gitignore             # Git ignore file
├── pyproject.toml         # Project metadata and dependencies
├── setup.py               # Setup script
├── pytest.ini             # Pytest configuration
└── README.md              # Project readme
```

## Development Workflow

### Branching Strategy

We use a feature branch workflow:

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. Make your changes and commit them with meaningful commit messages.

3. Push your branch to GitHub:
   ```bash
   git push -u origin feature/your-feature-name
   ```

4. Create a pull request in GitHub.

### Code Style

We follow the [Black](https://black.readthedocs.io/) code style for Python, with a line length of 88 characters. We also use [isort](https://pycqa.github.io/isort/) for import sorting and [mypy](http://mypy-lang.org/) for static type checking.

To format your code:

```bash
# Format code with Black
black llamaforge tests examples

# Sort imports
isort llamaforge tests examples

# Check types with mypy
mypy llamaforge
```

### Commits

Please follow these guidelines for commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Testing

We use [pytest](https://docs.pytest.org/) for unit testing. Tests are located in the `tests/` directory.

### Running Tests

Run all tests:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=llamaforge
```

Generate a coverage report:

```bash
pytest --cov=llamaforge --cov-report=html
```

### Writing Tests

When adding new features, please include tests. Here's a basic example:

```python
# tests/test_feature.py
import pytest
from llamaforge import YourFeature

def test_your_feature():
    feature = YourFeature()
    assert feature.does_something() == expected_result
    
@pytest.mark.parametrize("input,expected", [
    ("input1", "expected1"),
    ("input2", "expected2"),
])
def test_parameterized(input, expected):
    feature = YourFeature()
    assert feature.process(input) == expected
```

## Documentation

We use [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/) for documentation.

### Building Documentation

Build and preview the documentation:

```bash
mkdocs serve
```

Then visit `http://localhost:8000` in your browser.

### Writing Documentation

Documentation files are in Markdown format and located in the `docs/` directory. The structure is defined in `mkdocs.yml`.

When adding new features, please update the relevant documentation files. For API documentation, we use [mkdocstrings](https://mkdocstrings.github.io/) to generate documentation from docstrings.

Example docstring format:

```python
def function_name(param1: str, param2: int = 42) -> bool:
    """Function description.
    
    A longer description that can span
    multiple lines.
    
    Args:
        param1: Description of param1
        param2: Description of param2, defaults to 42
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        
    Examples:
        >>> function_name("example", 10)
        True
    """
```

## Adding New Features

### Adding a New Backend

To add a new backend:

1. Create a new file in `llamaforge/backends/` (e.g., `llamaforge/backends/custom_backend.py`).
2. Implement the `BaseBackend` interface:

```python
from llamaforge.backends.base import BaseBackend

class CustomBackend(BaseBackend):
    def __init__(self):
        super().__init__()
        self.model = None
        
    def load_model(self, model_path, parameters=None):
        """Load a model from the given path with the given parameters."""
        parameters = parameters or {}
        # Implement model loading logic
        
    def generate(self, prompt, parameters=None):
        """Generate text from the given prompt with the given parameters."""
        parameters = parameters or {}
        # Implement text generation logic
        
    def generate_stream(self, prompt, parameters=None):
        """Stream generated text from the given prompt with the given parameters."""
        parameters = parameters or {}
        # Implement streaming text generation logic
        
    def is_model_loaded(self):
        """Check if a model is loaded."""
        return self.model is not None
```

3. Register the backend in `llamaforge/backends/__init__.py`:

```python
from llamaforge.backends.custom_backend import CustomBackend

BACKENDS["custom"] = CustomBackend
```

4. Add tests for your backend in `tests/backends/test_custom_backend.py`.

5. Update the documentation to include your new backend.

### Adding a New Plugin

To add a new plugin:

1. Determine the type of plugin (preprocessing, postprocessing, tool, or command).

2. Create a new file in the appropriate subdirectory, e.g., `llamaforge/plugins/preprocessing/my_plugin.py`:

```python
from llamaforge.plugins.base import BasePlugin

class MyPlugin(BasePlugin):
    @property
    def name(self):
        return "my_plugin"
        
    @property
    def description(self):
        return "Description of my plugin"
        
    def process(self, text, **kwargs):
        # Implement processing logic
        return modified_text
```

3. Register the plugin in the appropriate `__init__.py` file:

```python
from llamaforge.plugins.preprocessing.my_plugin import MyPlugin

PREPROCESSORS["my_plugin"] = MyPlugin
```

4. Add tests for your plugin in `tests/plugins/preprocessing/test_my_plugin.py`.

5. Update the documentation to include your new plugin.

## Common Development Tasks

### Adding Dependencies

To add new dependencies, update the `pyproject.toml` file:

```toml
[tool.poetry.dependencies]
new-dependency = "^1.0.0"
```

If using pip, also update the `setup.py` file.

### Updating CI Workflow

The continuous integration workflow is defined in `.github/workflows/ci.yml`. If you need to make changes to the CI process, edit this file.

### Publishing a Release

1. Update the version number in `llamaforge/__init__.py` and `pyproject.toml`.

2. Update the changelog with your changes.

3. Create a tag for the new version:
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```

4. The CI workflow will automatically build and publish the package to PyPI.

## Troubleshooting

### Common Issues

#### Import Errors

If you're seeing import errors, make sure your environment is properly set up:

```bash
pip install -e .
```

#### Test Failures

If tests are failing, try:

1. Make sure all dependencies are installed: `pip install -e ".[dev]"`
2. Check if the failing tests have any specific requirements
3. Run specific failing tests with more verbosity: `pytest tests/test_specific.py -v`

#### Documentation Build Issues

If you're having issues building the documentation:

1. Make sure you have all dependencies installed: `pip install -e ".[docs]"`
2. Check for syntax errors in your Markdown files
3. Verify that the `mkdocs.yml` file is correctly configured

## Getting Help

If you need help, you can:

- Check the existing issues on GitHub
- Create a new issue for questions, bug reports, or feature requests
- Reach out to the maintainers directly

## Coding Standards

### Python Version Compatibility

LlamaForge supports Python 3.8 and higher. Please ensure your code is compatible with all supported Python versions.

### Type Annotations

We use type annotations throughout the codebase. Please add type annotations to all function signatures and use mypy to check for type errors.

### Error Handling

Use custom exceptions from `llamaforge.exceptions` where appropriate. Always include informative error messages.

### Logging

Use the `logging` module for debug and info messages:

```python
import logging

logger = logging.getLogger(__name__)

def function():
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
```

### Performance Considerations

Be mindful of performance, especially in code that's called frequently:

- Use profiling tools to identify bottlenecks
- Consider memory usage for large models
- Use generators and lazy loading where appropriate
- Add caching for expensive operations

Thank you for contributing to LlamaForge! 