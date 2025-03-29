# Contributing to LlamaForge

Thank you for your interest in contributing to LlamaForge! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to foster an open and welcoming environment.

## How to Contribute

There are many ways to contribute to LlamaForge:

1. Reporting bugs
2. Suggesting features
3. Improving documentation
4. Writing code
5. Reviewing pull requests
6. Creating examples and tutorials

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/yourusername/llamaforge.git
cd llamaforge
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

5. Install the package in development mode:
```bash
pip install -e .
```

## Development Workflow

1. Create a new branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes
3. Run tests to ensure your changes don't break existing functionality:
```bash
pytest
```

4. Format your code with `black`:
```bash
black .
```

5. Run linting with `flake8`:
```bash
flake8 .
```

6. Commit your changes:
```bash
git commit -am "Add your descriptive commit message"
```

7. Push your changes to your fork:
```bash
git push origin feature/your-feature-name
```

8. Create a pull request on GitHub

## Pull Request Guidelines

- Follow the [Python Style Guide](https://peps.python.org/pep-0008/)
- Include tests for new features
- Update documentation as needed
- Keep your pull request focused on a single topic
- Add a clear description of the changes
- Link related issues in the pull request description

## Project Structure

```
llamaforge/
├── llamaforge/          # Main package code
│   ├── __init__.py      # Package initialization
│   ├── main.py          # Main entry point
│   ├── api_server.py    # API server implementation
│   ├── config_wizard.py # Configuration wizard
│   ├── model_manager.py # Model management
│   ├── plugin_manager.py # Plugin system
│   └── version.py       # Version information
├── scripts/             # Installation and utility scripts
├── tests/               # Test files
├── docs/                # Documentation
├── examples/            # Example code
└── LICENSE              # License file
```

## Writing Tests

- Place tests in the `tests/` directory
- Use `pytest` for testing
- Organize tests to mirror the structure of the `llamaforge` package

Example test file:
```python
# tests/test_model_manager.py
import pytest
from llamaforge.model_manager import ModelManager

def test_list_models():
    manager = ModelManager({"directories": {"models": "/tmp/models"}})
    models = manager.list_models()
    assert isinstance(models, dict)
```

## Documentation Guidelines

- Use docstrings for all public methods and functions
- Follow the [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings
- Keep the README.md updated with examples and features
- Update documentation when adding or changing features

## Creating Examples

If you have created an interesting example of using LlamaForge, consider adding it to the `examples/` directory:

1. Create a new Python file in the `examples/` directory
2. Include clear documentation in the script
3. Make sure it runs with the current version of LlamaForge
4. Add a brief description to the README.md file

## Reporting Bugs

When reporting bugs, please include:

1. The steps to reproduce the bug
2. The expected behavior
3. The actual behavior
4. Your environment information (OS, Python version, etc.)
5. Any relevant logs or error messages

## Feature Requests

When suggesting features, please include:

1. A clear description of the feature
2. The motivation for adding this feature
3. Examples of how the feature would be used
4. If possible, thoughts on how to implement the feature

## License

By contributing to LlamaForge, you agree that your contributions will be licensed under the project's [MIT License](LICENSE). 