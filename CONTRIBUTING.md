# Contributing to LlamaForge

Thank you for your interest in contributing to LlamaForge! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## Getting Started

### Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/llamaforge.git
   cd llamaforge
   ```

3. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with clear, descriptive commit messages:
   ```bash
   git commit -m "Add feature: description of your feature"
   ```

3. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a pull request against the main repository.

## Pull Request Process

1. Ensure your code follows the project's style guidelines.
2. Include tests for any new functionality.
3. Update documentation to reflect any changes.
4. Make sure all tests pass and the build is successful.
5. The pull request will be reviewed by maintainers, who may suggest changes.
6. Once approved, a maintainer will merge your pull request.

## Coding Standards

- Follow PEP 8 style guidelines for Python code.
- Use type hints where appropriate.
- Write docstrings for all functions, classes, and modules.
- Keep functions small and focused on a single task.

## Testing

- Write tests for all new features and bug fixes.
- Run the test suite before submitting a pull request:
  ```bash
  pytest
  ```

## Documentation

- Update documentation for any changed functionality.
- Documentation should be clear and accessible to users of all experience levels.
- Example code should be tested and functional.

## Reporting Bugs

When reporting bugs, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment information (OS, Python version, LlamaForge version)

## Feature Requests

We welcome feature requests! When submitting a feature request, please:

- Provide a clear description of the feature
- Explain why this feature would be useful
- Consider how it could be implemented
- Indicate if you're willing to help implement it

## Git Commit Guidelines

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Additional Resources

- [Python Documentation Style Guide](https://docs.python.org/3/documenting/index.html)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [GitHub Pull Request Documentation](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)

Thank you for contributing to LlamaForge! 