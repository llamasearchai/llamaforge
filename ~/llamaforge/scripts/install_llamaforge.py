#!/usr/bin/env python3
"""
LlamaForge Installation Script

This script installs LlamaForge to the user's home directory and sets up the CLI launcher.
It performs system checks, installs dependencies, and creates the necessary directory structure.
"""

import os
import sys
import platform
import subprocess
import shutil
import json
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Union
import pkg_resources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("llamaforge_installer")

# Constants
DEFAULT_CONFIG = {
    "version": "0.2.0",
    "directories": {
        "models": "~/.llamaforge/models",
        "cache": "~/.llamaforge/cache",
        "plugins": "~/.llamaforge/plugins",
        "logs": "~/.llamaforge/logs"
    },
    "model_defaults": {
        "context_length": 4096,
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    },
    "api_server": {
        "enabled": False,
        "host": "localhost",
        "port": 8000
    },
    "advanced": {
        "check_updates": True,
        "telemetry": False
    }
}

REQUIRED_PACKAGES = [
    "requests>=2.25.0",
    "tqdm>=4.50.0",
    "numpy>=1.19.0",
    "huggingface_hub>=0.10.0",
    "fastapi>=0.70.0",
    "uvicorn>=0.15.0",
    "pydantic>=1.9.0",
    "rich>=10.0.0"
]

OPTIONAL_PACKAGES = {
    "llama": ["llama-cpp-python>=0.1.0"],
    "mlx": ["mlx>=0.0.3"],
    "transformers": ["transformers>=4.20.0", "torch>=1.10.0"]
}

def check_python_version() -> bool:
    """Check if the Python version is compatible with LlamaForge."""
    logger.info(f"Checking Python version...")
    major, minor = sys.version_info.major, sys.version_info.minor
    if major < 3 or (major == 3 and minor < 8):
        logger.error(f"Python {major}.{minor} is not supported. LlamaForge requires Python 3.8 or higher.")
        return False
    logger.info(f"Python {major}.{minor} is compatible with LlamaForge.")
    return True

def check_system_compatibility() -> bool:
    """Check if the system is compatible with LlamaForge."""
    logger.info(f"Checking system compatibility...")
    system = platform.system().lower()
    
    if system not in ["linux", "darwin", "windows"]:
        logger.error(f"Unsupported operating system: {system}")
        return False
    
    if system == "darwin":
        # Check if macOS has Apple Silicon
        processor = platform.processor()
        if processor == "arm":
            logger.info("Detected Apple Silicon (M1/M2/M3) Mac.")
        else:
            logger.info("Detected Intel Mac.")
    
    logger.info(f"System {system} is compatible with LlamaForge.")
    return True

def install_dependencies(install_all: bool = False) -> bool:
    """Install the required dependencies for LlamaForge."""
    logger.info("Installing required dependencies...")
    
    # Install required packages
    try:
        for package in REQUIRED_PACKAGES:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        if install_all:
            # Install all optional dependencies
            for backend, packages in OPTIONAL_PACKAGES.items():
                logger.info(f"Installing {backend} dependencies...")
                for package in packages:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            # Ask user which optional dependencies to install
            for backend, packages in OPTIONAL_PACKAGES.items():
                if input(f"Install {backend} backend dependencies? (y/n): ").lower() == "y":
                    logger.info(f"Installing {backend} dependencies...")
                    for package in packages:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        logger.info("All dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def create_directory_structure() -> Tuple[Path, bool]:
    """Create the directory structure for LlamaForge."""
    logger.info("Creating directory structure...")
    install_dir = Path.home() / ".llamaforge"
    
    try:
        # Create main directories
        install_dir.mkdir(exist_ok=True)
        (install_dir / "llamaforge").mkdir(exist_ok=True)
        (install_dir / "models").mkdir(exist_ok=True)
        (install_dir / "cache").mkdir(exist_ok=True)
        (install_dir / "plugins").mkdir(exist_ok=True)
        (install_dir / "logs").mkdir(exist_ok=True)
        
        logger.info(f"Directory structure created at {install_dir}")
        return install_dir, True
    except Exception as e:
        logger.error(f"Failed to create directory structure: {e}")
        return install_dir, False

def copy_files(source_dir: Path, install_dir: Path) -> bool:
    """Copy LlamaForge files to the installation directory."""
    logger.info("Copying LlamaForge files...")
    
    try:
        # Copy package files
        source_package_dir = source_dir / "llamaforge"
        target_package_dir = install_dir / "llamaforge"
        
        for file in source_package_dir.glob("*.py"):
            shutil.copy2(file, target_package_dir)
            logger.info(f"Copied {file.name} to {target_package_dir}")
        
        # Make executable
        for file in target_package_dir.glob("*.py"):
            os.chmod(file, 0o755)
        
        # Copy CLI launcher
        cli_script = source_dir / "scripts" / "llamaforge_cli"
        if cli_script.exists():
            target_cli = install_dir / "llamaforge_cli"
            shutil.copy2(cli_script, target_cli)
            os.chmod(target_cli, 0o755)
            logger.info(f"Copied CLI launcher to {target_cli}")
        
        logger.info("All files copied successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to copy files: {e}")
        return False

def create_config_file(install_dir: Path) -> bool:
    """Create the default configuration file."""
    logger.info("Creating configuration file...")
    config_path = install_dir / "config.json"
    
    try:
        with open(config_path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        
        logger.info(f"Configuration file created at {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create configuration file: {e}")
        return False

def setup_cli_launcher() -> bool:
    """Set up the CLI launcher in the user's PATH."""
    logger.info("Setting up CLI launcher...")
    
    try:
        home_dir = Path.home()
        install_dir = home_dir / ".llamaforge"
        
        # Create symlink in ~/.local/bin (Linux/macOS)
        if platform.system().lower() in ["linux", "darwin"]:
            bin_dir = home_dir / ".local" / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            
            symlink_path = bin_dir / "llamaforge"
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            
            os.symlink(install_dir / "llamaforge_cli", symlink_path)
            logger.info(f"Created symlink at {symlink_path}")
            
            # Check if ~/.local/bin is in PATH
            if str(bin_dir) not in os.environ.get("PATH", ""):
                logger.warning(f"{bin_dir} is not in your PATH. Add it to your shell's profile.")
                if platform.system().lower() == "darwin":
                    logger.info("For zsh, add this line to ~/.zshrc: export PATH=$HOME/.local/bin:$PATH")
                else:
                    logger.info("For bash, add this line to ~/.bashrc: export PATH=$HOME/.local/bin:$PATH")
        
        # For Windows, add a batch file
        elif platform.system().lower() == "windows":
            # Create Scripts directory if it doesn't exist
            scripts_dir = Path(sys.prefix) / "Scripts"
            scripts_dir.mkdir(exist_ok=True)
            
            # Create batch file
            batch_path = scripts_dir / "llamaforge.bat"
            with open(batch_path, "w") as f:
                f.write(f'@echo off\r\npython "{install_dir / "llamaforge_cli"}" %*\r\n')
            
            logger.info(f"Created batch file at {batch_path}")
        
        logger.info("CLI launcher setup complete.")
        return True
    except Exception as e:
        logger.error(f"Failed to set up CLI launcher: {e}")
        return False

def main() -> int:
    """Main function for the installer."""
    parser = argparse.ArgumentParser(description="Install LlamaForge")
    parser.add_argument("--all-deps", action="store_true", help="Install all optional dependencies")
    parser.add_argument("--no-deps", action="store_true", help="Skip dependency installation")
    args = parser.parse_args()
    
    logger.info("Starting LlamaForge installation...")
    
    # Perform system checks
    if not check_python_version() or not check_system_compatibility():
        logger.error("System checks failed. Installation aborted.")
        return 1
    
    # Install dependencies
    if not args.no_deps:
        if not install_dependencies(args.all_deps):
            logger.error("Dependency installation failed. Installation aborted.")
            return 1
    else:
        logger.info("Skipping dependency installation.")
    
    # Create directory structure
    install_dir, success = create_directory_structure()
    if not success:
        logger.error("Failed to create directory structure. Installation aborted.")
        return 1
    
    # Copy files
    source_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    if not copy_files(source_dir, install_dir):
        logger.error("Failed to copy files. Installation aborted.")
        return 1
    
    # Create config file
    if not create_config_file(install_dir):
        logger.error("Failed to create configuration file. Installation aborted.")
        return 1
    
    # Set up CLI launcher
    if not setup_cli_launcher():
        logger.error("Failed to set up CLI launcher. Installation aborted.")
        return 1
    
    logger.info("""
===============================
LlamaForge installation complete!
===============================

You can now run LlamaForge by typing:

    llamaforge

If this command is not found, you may need to:
1. Add ~/.local/bin to your PATH (Linux/macOS)
2. Restart your terminal or open a new one

To configure LlamaForge, run:

    llamaforge config

Enjoy using LlamaForge!
    """)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 