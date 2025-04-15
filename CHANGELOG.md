# Changelog

All notable changes to LlamaForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2023-03-15

### Added
- Model Manager for downloading and managing models from various sources
- Plugin System for extending functionality
- API Server compatible with OpenAI API
- Configuration Wizard for easy setup
- Enhanced Chat mode with plugin support
- Support for multiple backend engines (llama.cpp, Hugging Face, OpenAI API)
- Streaming output for text generation
- Documentation and examples

### Changed
- Complete architecture refactor for better extensibility
- Improved performance with caching and optimized model loading
- Enhanced command-line interface with rich formatting

### Fixed
- Memory leak in long-running chat sessions
- Handling of special tokens in model outputs
- Path resolution issues on Windows

## [0.1.0] - 2023-02-10

### Added
- Initial release with basic functionality
- Support for llama.cpp models
- Simple command-line interface
- Basic text generation capabilities 