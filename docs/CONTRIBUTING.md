# Contributing to LlamaForge

Thank you for your interest in contributing to LlamaForge! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to foster an inclusive and respectful community.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/llamaforge.git
   cd llamaforge
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   pip install -e ".[dev]"  # Installs development dependencies
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. Make your changes to the codebase.
2. Run the tests to ensure your changes don't break existing functionality:
   ```bash
   python tests/run_tests.py
   ```
3. Format your code using the provided tools:
   ```bash
   black llamaforge tests examples
   isort llamaforge tests examples
   ```
4. Run linting to check for code quality issues:
   ```bash
   flake8 llamaforge tests examples
   mypy llamaforge
   ```
5. Commit your changes with a descriptive commit message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```
6. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Create a Pull Request on GitHub.

## Pull Request Guidelines

- Fill in the required template.
- Include tests for new features or bug fixes.
- Update documentation if necessary.
- Ensure all tests pass.
- Keep your PR focused on a single topic.

## Coding Standards

- Follow PEP 8 style guidelines.
- Use type hints for function parameters and return values.
- Write docstrings for all functions, classes, and modules.
- Keep functions and methods small and focused.
- Use meaningful variable and function names.

## Adding New Features

When adding new features, please follow these guidelines:

1. Discuss the feature in an issue before implementing it.
2. Design the feature to be modular and extensible.
3. Add appropriate tests for the feature.
4. Update documentation to reflect the new feature.
5. Add an example demonstrating the feature if applicable.

## Plugin Development

LlamaForge supports plugins for extending functionality. To develop a plugin:

1. Create a new Python file in the `~/.llamaforge/plugins` directory.
2. Inherit from the appropriate plugin base class:
   - `PreprocessorPlugin`: Modify prompts before sending to the model.
   - `PostprocessorPlugin`: Modify completions before returning to the user.
   - `FormatterPlugin`: Format model outputs in a specific way.
   - `CommandPlugin`: Add custom commands to the CLI.
   - `ToolPlugin`: Add custom tools for chat mode.
   - `AdapterPlugin`: Adapt different model formats and APIs.
3. Implement the required methods for your plugin type.
4. Test your plugin with LlamaForge.

Example plugin:

```python
from llamaforge.plugin_manager import PreprocessorPlugin

class MyPlugin(PreprocessorPlugin):
    def __init__(self):
        super().__init__(
            name="my_plugin",
            description="A sample plugin that modifies the prompt"
        )
    
    def process(self, prompt):
        return f"Enhanced prompt: {prompt}"
```

## Documentation

- Update the README.md file with any new features or changes.
- Add or update API documentation in the docs directory.
- Include examples for new features.

## Testing

- Write unit tests for all new functionality.
- Ensure existing tests pass with your changes.
- Run the test suite before submitting a PR.

## Releasing

Only project maintainers can release new versions. The release process is:

1. Update version in `llamaforge/version.py`.
2. Update CHANGELOG.md with the changes in the new version.
3. Create a new release on GitHub with release notes.
4. Build and upload the package to PyPI:
   ```bash
   python setup.py sdist bdist_wheel
   twine upload dist/*
   ```

## Getting Help

If you need help with contributing, please:

- Check the documentation in the docs directory.
- Open an issue on GitHub with your question.
- Reach out to the maintainers via email or GitHub discussions.

Thank you for contributing to LlamaForge! 