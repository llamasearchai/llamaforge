#!/usr/bin/env python3
"""
Enhanced LlamaForge Installer and Setup Script
==============================================

This script installs and configures the enhanced version of LlamaForge - a comprehensive
command-line interface for language models with these new features:

- Model Manager: Download and manage models from Hugging Face and local sources
- Plugin System: Extend functionality with preprocessing, postprocessing, tools, and commands
- API Server: Compatible with the OpenAI API for integration with existing applications  
- Configuration Wizard: Interactive setup for easy configuration
- Enhanced Chat: Chat mode with plugin support and commands

Run this script to install the latest version of LlamaForge with all enhancements.

Usage Examples:
    python3 install_llamaforge.py --dir ~/.llamaforge --backends all
    python3 install_llamaforge.py --no-path --no-sample-data

Version: 0.2.0
"""

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("llamaforge_installer")

# Global constants
DEFAULT_INSTALL_DIR = Path.home() / ".llamaforge"

CORE_DEPENDENCIES = [
    "fastapi>=0.70.0",
    "uvicorn>=0.15.0",
    "httpx>=0.20.0",
    "numpy>=1.21.0",
    "pydantic>=1.8.0",
    "requests>=2.25.0",
    "tqdm>=4.60.0",
    "huggingface_hub>=0.10.0",
]

BACKEND_DEPENDENCIES = {
    "mlx": ["mlx>=0.2.0", "mlx-lm>=0.0.3"],
    "llama.cpp": ["llama-cpp-python>=0.1.86"],
    "transformers": ["transformers>=4.18.0", "torch>=2.0.0", "accelerate>=0.16.0", "sentencepiece>=0.1.96"],
}

# ASCII art banner
BANNER = r"""
  _      _                        ______                    
 | |    | |                      |  ____|                   
 | |    | | __ _ _ __ ___   __ _ | |__ ___  _ __ __ _  ___ 
 | |    | |/ _` | '_ ` _ \ / _` ||  __/ _ \| '__/ _` |/ _ \
 | |____| | (_| | | | | | | (_| || | | (_) | | | (_| |  __/
 |______|_|\__,_|_| |_| |_|\__,_||_|  \___/|_|  \__, |\___|
                  v0.2.0 Installer               __/ |     
                                                |___/      
"""

SOURCE_FILES = [
    "main.py",
    "version.py",
    "model_manager.py",
    "config_wizard.py",
    "plugin_manager.py",
    "api_server.py",
]

def check_python_version():
    """Ensure Python 3.8 or higher is used."""
    logger.info("Checking Python version...")
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required. Current version: {}.{}".format(
            sys.version_info.major, sys.version_info.minor))
        sys.exit(1)
    logger.info(f"Python version {sys.version_info.major}.{sys.version_info.minor} is compatible.")

def check_platform_compatibility():
    """Detect the operating system and CPU to determine backend support."""
    logger.info("Checking platform compatibility...")
    system = platform.system()
    processor = platform.processor()
    logger.info(f"Detected system: {system}, processor: {processor}")
    
    mlx_supported = False
    if system == "Darwin" and ("arm" in processor.lower() or "Apple" in processor):
        mlx_supported = True
        logger.info("Apple Silicon detected: MLX backend is available.")
    else:
        logger.info("MLX backend is not available on this system.")
    
    cuda_supported = False
    if system in ["Linux", "Windows"]:
        # Check for NVIDIA GPU (very simplified check)
        try:
            if system == "Windows":
                result = subprocess.run(["wmic", "path", "win32_VideoController", "get", "name"], 
                                        capture_output=True, text=True)
                if "NVIDIA" in result.stdout:
                    cuda_supported = True
            else:  # Linux
                result = subprocess.run(["nvidia-smi"], capture_output=True)
                if result.returncode == 0:
                    cuda_supported = True
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
        
        if cuda_supported:
            logger.info("NVIDIA GPU detected: CUDA backend is available.")
        else:
            logger.info("No NVIDIA GPU detected.")
    
    return {
        "system": system, 
        "processor": processor, 
        "mlx_supported": mlx_supported,
        "cuda_supported": cuda_supported
    }

def create_directory_structure(install_dir: Path):
    """Create the complete directory structure required by LlamaForge."""
    logger.info(f"Creating installation directories at {install_dir}...")
    directories = {
        "models_dir": install_dir / "models",
        "cache_dir": install_dir / "cache",
        "logs_dir": install_dir / "logs",
        "config_dir": install_dir,
        "datasets_dir": install_dir / "datasets",
        "plugins_dir": install_dir / "plugins",
    }
    
    # Create subdirectories for different model backends
    for backend in ["llama.cpp", "mlx", "transformers"]:
        directories[f"{backend}_dir"] = directories["models_dir"] / backend
    
    # Create plugin directories
    directories["system_plugins_dir"] = directories["plugins_dir"] / "system"
    directories["user_plugins_dir"] = directories["plugins_dir"] / "user"
    
    # Create all directories
    for name, dir_path in directories.items():
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            sys.exit(1)
    
    return directories

def install_dependencies(backends: list):
    """Install core and backend-specific dependencies using pip."""
    logger.info("Installing core dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + CORE_DEPENDENCIES)
    except subprocess.CalledProcessError as e:
        logger.error(f"Core dependencies installation failed: {e}")
        sys.exit(1)
    
    if backends:
        for backend in backends:
            deps = BACKEND_DEPENDENCIES.get(backend, [])
            if deps:
                logger.info(f"Installing dependencies for {backend} backend...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + deps)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Installation for backend {backend} failed: {e}. Continuing installation.")
    
    logger.info("Dependency installation complete.")

def create_config_file(dirs: dict, platform_info: dict, config_file: Path):
    """Create the JSON configuration file with default settings."""
    logger.info("Creating configuration file...")
    
    # Determine default backend based on platform
    if platform_info["mlx_supported"]:
        default_backend = "mlx"
    elif platform_info["cuda_supported"]:
        default_backend = "llama.cpp"  # Better CUDA support in llama.cpp generally
    else:
        default_backend = "llama.cpp"  # CPU fallback
    
    # Create default configuration
    config = {
        "models_dir": str(dirs["models_dir"]),
        "cache_dir": str(dirs["cache_dir"]),
        "logs_dir": str(dirs["logs_dir"]),
        "datasets_dir": str(dirs["datasets_dir"]),
        "plugins_dir": str(dirs["plugins_dir"]),
        "system_plugins_dir": str(dirs["system_plugins_dir"]),
        "user_plugins_dir": str(dirs["user_plugins_dir"]),
        "default_backend": default_backend,
        "default_model": None,  # Will be set by model manager when model is downloaded
        "default_context_length": 4096,
        "default_max_tokens": 1024,
        "default_temperature": 0.7,
        "default_top_p": 0.9,
        "enable_api": True,
        "api_host": "127.0.0.1",
        "api_port": 8000,
        "auto_update_check": True,
        "telemetry": False,
        "chat_templates": {
            "llama": "<s>[INST] {prompt} [/INST]",
            "mistral": "<s>[INST] {prompt} [/INST]",
            "mixtral": "<s>[INST] {prompt} [/INST]",
            "phi": "<|user|>{prompt}<|assistant|>",
            "gemma": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
            "default": "{prompt}"
        },
    }
    
    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration file written to {config_file}")
    except Exception as e:
        logger.error(f"Failed to write configuration file: {e}")
        sys.exit(1)
    
    return config

def get_source_file(file_name: str, current_dir: Path):
    """Find a source file in the current directory or from the user's modified version."""
    # First check if it exists in the current directory (for development)
    local_file = current_dir / file_name
    if local_file.exists():
        return local_file
    
    # Then check if it might be in the ~/.llamaforge directory (for updating)
    llamaforge_file = Path.home() / ".llamaforge" / "llamaforge" / file_name
    if llamaforge_file.exists():
        return llamaforge_file
    
    # Finally, look for it in the current directory's llamaforge subfolder
    module_file = current_dir / "llamaforge" / file_name
    if module_file.exists():
        return module_file
    
    logger.warning(f"Source file {file_name} not found. Installation may be incomplete.")
    return None

def copy_source_files(install_dir: Path, current_dir: Path):
    """Copy all source files to the installation directory."""
    logger.info("Copying source files...")
    module_dir = install_dir / "llamaforge"
    module_dir.mkdir(exist_ok=True)
    
    # Create __init__.py to expose main function
    init_file = module_dir / "__init__.py"
    init_content = """# LlamaForge package initialization
from .main import main
from .version import __version__

__all__ = ["main", "__version__"]
"""
    try:
        with open(init_file, "w") as f:
            f.write(init_content)
        logger.info(f"Created {init_file}")
    except Exception as e:
        logger.error(f"Error creating __init__.py: {e}")
    
    # Copy all source files
    for file_name in SOURCE_FILES:
        source_file = get_source_file(file_name, current_dir)
        if source_file:
            dest_file = module_dir / file_name
            try:
                shutil.copy2(source_file, dest_file)
                logger.info(f"Copied {file_name} to {dest_file}")
            except Exception as e:
                logger.error(f"Error copying {file_name}: {e}")

def create_launcher_script(install_dir: Path):
    """Create the CLI launcher script."""
    logger.info("Creating launcher script...")
    launcher_file = install_dir / "llamaforge_cli"
    launcher_content = '''#!/usr/bin/env python3
"""
LlamaForge CLI Launcher
This script launches the LlamaForge command-line interface.
"""

import os
import sys
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Add the parent directory to the Python path
sys.path.insert(0, str(script_dir))

# If ~/.llamaforge is in the Python path, execute main.py
try:
    from llamaforge.main import main
    main()
except ImportError as e:
    print(f"Error importing LlamaForge: {e}")
    print("Please ensure LlamaForge is installed correctly.")
    sys.exit(1)
except KeyboardInterrupt:
    print("\\nLlamaForge terminated by user.")
    sys.exit(0)
except Exception as e:
    print(f"\\nError in LlamaForge: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    try:
        with open(launcher_file, "w") as f:
            f.write(launcher_content)
        launcher_file.chmod(0o755)  # Make executable
        logger.info(f"Created launcher script at {launcher_file}")
    except Exception as e:
        logger.error(f"Error creating launcher script: {e}")

def add_to_system_path(install_dir: Path, launcher_file: Path):
    """Add LlamaForge to system PATH."""
    logger.info("Adding LlamaForge to system PATH...")
    system = platform.system()
    
    if system in ["Linux", "Darwin"]:  # Unix-like systems
        # Create symlink in /usr/local/bin
        symlink_path = Path("/usr/local/bin/llamaforge")
        try:
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()  # Remove existing symlink
            symlink_path.symlink_to(launcher_file)
            logger.info(f"Symlink created at {symlink_path}")
        except PermissionError:
            logger.warning(f"Permission denied creating symlink at {symlink_path}")
            logger.info(f"To add LlamaForge to PATH, run: sudo ln -s {launcher_file} {symlink_path}")
        
        # Also try to add to .bashrc or .zshrc if possible
        zshrc_path = Path.home() / ".zshrc"
        bashrc_path = Path.home() / ".bashrc"
        
        shell_config = None
        if zshrc_path.exists():
            shell_config = zshrc_path
        elif bashrc_path.exists():
            shell_config = bashrc_path
        
        if shell_config:
            try:
                with open(shell_config, "a") as f:
                    f.write(f"\n# Added by LlamaForge installer\nexport PATH=\"$PATH:{install_dir}\"\n")
                logger.info(f"Added LlamaForge to PATH in {shell_config}")
            except Exception as e:
                logger.warning(f"Failed to update shell config: {e}")
    
    elif system == "Windows":
        # Add to user PATH on Windows
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS) as key:
                current_path, _ = winreg.QueryValueEx(key, "PATH")
                install_path = str(install_dir)
                if install_path not in current_path:
                    new_path = current_path + ";" + install_path
                    winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
                    logger.info(f"Added {install_path} to user PATH")
        except Exception as e:
            logger.warning(f"Failed to add to Windows PATH: {e}")
            logger.info(f"To add LlamaForge to PATH, add {install_dir} to your PATH environment variable.")

def download_sample_data(install_dir: Path):
    """Download sample datasets for benchmarks and fine-tuning."""
    logger.info("Setting up sample data...")
    datasets_dir = install_dir / "datasets"
    
    # Create sample benchmark dataset
    sample_benchmark = [
        {
            "prompt": "What is the capital of France?",
            "reference": "The capital of France is Paris."
        },
        {
            "prompt": "List three renewable energy sources.",
            "reference": "Solar power, wind power, and hydroelectric power are renewable energy sources."
        },
        {
            "prompt": "Explain the concept of machine learning.",
            "reference": "Machine learning is a subset of AI that enables systems to learn from data rather than explicit programming."
        }
    ]
    
    sample_benchmark_path = datasets_dir / "sample_benchmark.json"
    try:
        with open(sample_benchmark_path, "w") as f:
            json.dump(sample_benchmark, f, indent=2)
        logger.info(f"Created sample benchmark at {sample_benchmark_path}")
    except Exception as e:
        logger.error(f"Error creating sample benchmark: {e}")
    
    # Create sample fine-tuning dataset
    sample_finetune = [
        {
            "instruction": "Explain how photosynthesis works",
            "input": "",
            "output": "Photosynthesis is the process where plants convert sunlight, water, and carbon dioxide into oxygen and glucose. The plant uses chlorophyll in its cells to absorb energy from sunlight, then uses that energy to convert CO2 and water into glucose and oxygen. The glucose provides energy for the plant, while oxygen is released as a byproduct."
        },
        {
            "instruction": "Write a short poem about the moon",
            "input": "",
            "output": "Silver orb in darkened skies,\nAncient guardian of the night,\nReflecting sun's forgotten light,\nGuiding dreams as shadows rise."
        },
        {
            "instruction": "Explain the difference between RAM and hard drive storage",
            "input": "",
            "output": "RAM (Random Access Memory) and hard drive storage differ in several key ways. RAM is temporary, volatile memory used by the computer for active tasks. It's much faster than a hard drive but loses its data when power is turned off. Hard drives provide permanent, non-volatile storage for files and programs, with much larger capacity but slower access speeds."
        }
    ]
    
    sample_finetune_path = datasets_dir / "sample_finetune.json"
    try:
        with open(sample_finetune_path, "w") as f:
            json.dump(sample_finetune, f, indent=2)
        logger.info(f"Created sample fine-tuning dataset at {sample_finetune_path}")
    except Exception as e:
        logger.error(f"Error creating sample fine-tuning dataset: {e}")

def create_readme(install_dir: Path):
    """Create a README file with usage instructions."""
    logger.info("Creating README file...")
    readme_file = install_dir / "README.md"
    readme_content = f"""# LlamaForge v0.2.0

LlamaForge is a comprehensive command-line interface for language models with extensive features and customization options.

## Features

- **Model Management**: Download and manage models from HuggingFace or local files
- **Plugin System**: Extend functionality with custom plugins
- **API Server**: OpenAI-compatible API for integrating with existing applications
- **Multiple Backends**: Support for llama.cpp, MLX (Apple Silicon), and Hugging Face Transformers
- **Interactive Chat**: Rich chat experience with commands and context management
- **Text Generation**: Generate text with flexible parameter controls
- **Benchmarking**: Evaluate model performance across different tasks
- **Fine-tuning**: Customize models with your own data

## Usage

### Basic Commands

```bash
# Show help
llamaforge --help

# Start interactive chat
llamaforge --chat

# Generate text from a prompt
llamaforge --generate "Write a poem about artificial intelligence"

# Show available models
llamaforge --list-models

# Download a model
llamaforge --download TheBloke/Mistral-7B-Instruct-v0.1-GGUF
```

### Advanced Features

```bash
# Run the API server
llamaforge --api-server

# Fine-tune a model
llamaforge --finetune --model your_model --dataset path/to/dataset.json

# Run benchmarks
llamaforge --benchmark --models model1,model2 --task general

# Manage plugins
llamaforge --list-plugins
llamaforge --load-plugin plugin_name
```

## Configuration

The configuration file is located at: `{install_dir / "config.json"}`

You can run the configuration wizard with:

```bash
llamaforge --config-wizard
```

For more details, visit the [GitHub repository](https://github.com/your-username/llamaforge).
"""
    
    try:
        with open(readme_file, "w") as f:
            f.write(readme_content)
        logger.info(f"Created README at {readme_file}")
    except Exception as e:
        logger.error(f"Error creating README: {e}")

def main():
    """Main installation function."""
    print(BANNER)
    logger.info("Starting LlamaForge v0.2.0 installation...")
    
    parser = argparse.ArgumentParser(description="LlamaForge Installer")
    parser.add_argument("--dir", help="Installation directory", default=str(DEFAULT_INSTALL_DIR))
    parser.add_argument("--backends", help="Comma-separated list of backends to install (mlx,llama.cpp,transformers,all)", default="all")
    parser.add_argument("--no-path", help="Do not add LlamaForge to system PATH", action="store_true")
    parser.add_argument("--no-sample-data", help="Do not download sample data", action="store_true")
    args = parser.parse_args()
    
    # Validate Python version and platform
    check_python_version()
    platform_info = check_platform_compatibility()
    
    # Set up installation directory
    install_dir = Path(args.dir).resolve()
    directories = create_directory_structure(install_dir)
    
    # Determine which backends to install
    if args.backends.lower() == "all":
        backends = ["llama.cpp", "transformers"]
        if platform_info["mlx_supported"]:
            backends.append("mlx")
    else:
        backends = [b.strip() for b in args.backends.split(",")]
        # Verify MLX is only installed on supported platforms
        if "mlx" in backends and not platform_info["mlx_supported"]:
            logger.warning("MLX backend is not supported on this platform. Skipping MLX installation.")
            backends.remove("mlx")
    
    # Install dependencies
    install_dependencies(backends)
    
    # Create configuration
    config_file = install_dir / "config.json"
    config = create_config_file(directories, platform_info, config_file)
    
    # Copy source files
    copy_source_files(install_dir, Path.cwd())
    
    # Create launcher script
    launcher_file = install_dir / "llamaforge_cli"
    create_launcher_script(install_dir)
    
    # Add to system PATH if requested
    if not args.no_path:
        add_to_system_path(install_dir, launcher_file)
    
    # Download sample data if requested
    if not args.no_sample_data:
        download_sample_data(install_dir)
    
    # Create README
    create_readme(install_dir)
    
    # Installation complete
    logger.info("LlamaForge installation complete!")
    print("\n" + "=" * 60)
    print("LlamaForge v0.2.0 has been successfully installed!")
    print(f"Installation directory: {install_dir}")
    
    # Print usage instructions
    print("\nTo run LlamaForge, use:")
    if args.no_path:
        print(f"  {launcher_file}")
    else:
        print("  llamaforge")
    
    print("\nExample commands:")
    print("  llamaforge --help            # Show help")
    print("  llamaforge --chat            # Start interactive chat")
    print("  llamaforge --config-wizard   # Configure settings")
    print("  llamaforge --download TheBloke/Mistral-7B-Instruct-v0.1-GGUF  # Download a model")
    
    print("\nFor more information, see the README file in the installation directory.")
    print("=" * 60)

if __name__ == "__main__":
    main() 