#!/usr/bin/env python3
"""
LlamaForge Configuration Wizard

This module provides an interactive configuration wizard for setting up LlamaForge.
It helps users configure directories, model defaults, and other settings.
"""

import os
import sys
import json
import logging
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

logger = logging.getLogger("llamaforge.config_wizard")

class ConfigWizard:
    """Interactive configuration wizard for LlamaForge."""
    
    def __init__(self, config_path: str):
        """Initialize the configuration wizard.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.system_info = self._detect_system()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration file or create a default one if it doesn't exist.
        
        Returns:
            Configuration dictionary
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse configuration file {self.config_path}, using defaults")
        
        # Default configuration
        config = {
            "version": "0.1.0",
            "dirs": {
                "models": os.path.expanduser("~/.llamaforge/models"),
                "plugins": os.path.expanduser("~/.llamaforge/plugins"),
                "cache": os.path.expanduser("~/.llamaforge/cache"),
                "logs": os.path.expanduser("~/.llamaforge/logs")
            },
            "model_defaults": {
                "backend": "llama.cpp",
                "context_length": 4096,
                "default_model": None
            },
            "api_server": {
                "enabled": False,
                "host": "127.0.0.1",
                "port": 8000
            },
            "advanced": {
                "log_level": "INFO",
                "plugins_enabled": True,
                "max_cache_size_gb": 1
            }
        }
        
        logger.info(f"Created default configuration")
        return config
    
    def _detect_system(self) -> Dict[str, Any]:
        """Detect system information.
        
        Returns:
            Dictionary with system information
        """
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
            "memory_gb": None
        }
        
        # Try to detect memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            info["memory_gb"] = round(memory.total / (1024**3), 1)
        except (ImportError, AttributeError):
            # If psutil is not available, use platform-specific methods
            if info["os"] == "Linux":
                try:
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if 'MemTotal' in line:
                                memory_kb = int(line.split()[1])
                                info["memory_gb"] = round(memory_kb / (1024**2), 1)
                                break
                except Exception:
                    pass
            elif info["os"] == "Darwin":  # macOS
                try:
                    import subprocess
                    result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
                    memory_bytes = int(result.stdout.strip())
                    info["memory_gb"] = round(memory_bytes / (1024**3), 1)
                except Exception:
                    pass
            elif info["os"] == "Windows":
                try:
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    c_ulonglong = ctypes.c_ulonglong
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ('dwLength', ctypes.c_ulong),
                            ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', c_ulonglong),
                            ('ullAvailPhys', c_ulonglong),
                            ('ullTotalPageFile', c_ulonglong),
                            ('ullAvailPageFile', c_ulonglong),
                            ('ullTotalVirtual', c_ulonglong),
                            ('ullAvailVirtual', c_ulonglong),
                            ('ullExtendedVirtual', c_ulonglong),
                        ]

                    memory_status = MEMORYSTATUSEX()
                    memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
                    info["memory_gb"] = round(memory_status.ullTotalPhys / (1024**3), 1)
                except Exception:
                    pass
        
        # Detect GPU capabilities
        info["has_gpu"] = False
        info["gpu_name"] = None
        
        # Check for CUDA
        try:
            import torch
            info["has_gpu"] = torch.cuda.is_available()
            if info["has_gpu"]:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        except ImportError:
            pass
        
        # Check for Metal on macOS
        if info["os"] == "Darwin" and not info["has_gpu"]:
            try:
                info["has_metal"] = platform.processor() in ["arm", "arm64"]
                if info["has_metal"]:
                    info["gpu_name"] = "Apple Silicon"
            except Exception:
                pass
        
        logger.info(f"Detected system information: {info}")
        return info
    
    def _save_config(self) -> None:
        """Save the configuration to the configuration file."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Write to a temporary file first
            with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
                json.dump(self.config, tmp, indent=2)
            
            # Move the temporary file to the actual config file
            shutil.move(tmp.name, self.config_path)
            
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    
    def _print_header(self, text: str) -> None:
        """Print a formatted header.
        
        Args:
            text: Header text
        """
        print("\n" + "=" * 60)
        print(text.center(60))
        print("=" * 60 + "\n")
    
    def _print_section(self, text: str) -> None:
        """Print a formatted section header.
        
        Args:
            text: Section header text
        """
        print("\n" + "-" * 60)
        print(text)
        print("-" * 60 + "\n")
    
    def _get_input(self, prompt: str, default: Optional[str] = None) -> str:
        """Get input from the user with a default value.
        
        Args:
            prompt: Prompt text
            default: Default value
            
        Returns:
            User input or default value
        """
        if default is not None:
            prompt = f"{prompt} [{default}]: "
        else:
            prompt = f"{prompt}: "
        
        value = input(prompt)
        if not value and default is not None:
            return default
        return value
    
    def _get_bool_input(self, prompt: str, default: bool = True) -> bool:
        """Get a boolean input from the user.
        
        Args:
            prompt: Prompt text
            default: Default value
            
        Returns:
            Boolean user input
        """
        default_str = "Y/n" if default else "y/N"
        prompt = f"{prompt} [{default_str}]: "
        
        value = input(prompt).strip().lower()
        if not value:
            return default
        
        return value in ["y", "yes", "true", "1"]
    
    def _get_choice_input(self, prompt: str, choices: List[str], default: Optional[int] = None) -> str:
        """Get a choice input from the user.
        
        Args:
            prompt: Prompt text
            choices: List of choices
            default: Default choice index
            
        Returns:
            Selected choice
        """
        print(f"{prompt}")
        for i, choice in enumerate(choices):
            default_marker = " (default)" if default is not None and i == default else ""
            print(f"{i+1}. {choice}{default_marker}")
        
        while True:
            if default is not None:
                value = input(f"Enter your choice [1-{len(choices)}, default={default+1}]: ")
                if not value:
                    return choices[default]
            else:
                value = input(f"Enter your choice [1-{len(choices)}]: ")
            
            try:
                index = int(value) - 1
                if 0 <= index < len(choices):
                    return choices[index]
                print(f"Invalid choice. Please enter a number between 1 and {len(choices)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def run_wizard(self) -> Dict[str, Any]:
        """Run the configuration wizard.
        
        Returns:
            Updated configuration dictionary
        """
        self._print_header("LlamaForge Configuration Wizard")
        print("This wizard will help you set up LlamaForge.")
        print("Press Ctrl+C at any time to cancel.")
        
        try:
            # Display system information
            self._print_section("System Information")
            print(f"OS: {self.system_info['os']} {self.system_info['os_version']}")
            print(f"Architecture: {self.system_info['architecture']}")
            print(f"CPU Count: {self.system_info['cpu_count']}")
            print(f"Python Version: {self.system_info['python_version']}")
            
            if self.system_info.get("memory_gb"):
                print(f"Memory: {self.system_info['memory_gb']} GB")
            
            if self.system_info.get("has_gpu"):
                print(f"GPU: {self.system_info['gpu_name']}")
                if self.system_info.get("gpu_memory_gb"):
                    print(f"GPU Memory: {self.system_info['gpu_memory_gb']} GB")
            elif self.system_info.get("has_metal"):
                print("GPU: Apple Silicon (Metal)")
            else:
                print("GPU: None detected")
            
            # Configure directories
            self._print_section("Directories")
            print("Configure the directories where LlamaForge will store models, plugins, and cache.")
            
            self.config["dirs"]["models"] = self._get_input(
                "Models directory", self.config["dirs"]["models"]
            )
            self.config["dirs"]["plugins"] = self._get_input(
                "Plugins directory", self.config["dirs"]["plugins"]
            )
            self.config["dirs"]["cache"] = self._get_input(
                "Cache directory", self.config["dirs"]["cache"]
            )
            self.config["dirs"]["logs"] = self._get_input(
                "Logs directory", self.config["dirs"]["logs"]
            )
            
            # Create directories
            for dir_name, dir_path in self.config["dirs"].items():
                os.makedirs(os.path.expanduser(dir_path), exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            
            # Configure model defaults
            self._print_section("Model Defaults")
            print("Configure default settings for models.")
            
            # Choose backend
            available_backends = ["llama.cpp"]
            if self.system_info.get("has_gpu") or self.system_info.get("has_metal"):
                available_backends.append("mlx")
            available_backends.append("transformers")
            
            default_backend_index = available_backends.index(self.config["model_defaults"]["backend"]) if self.config["model_defaults"]["backend"] in available_backends else 0
            backend = self._get_choice_input(
                "Select the default model backend:",
                available_backends,
                default_backend_index
            )
            self.config["model_defaults"]["backend"] = backend
            
            # Context length
            context_length = self._get_input(
                "Default context length (in tokens)",
                str(self.config["model_defaults"]["context_length"])
            )
            self.config["model_defaults"]["context_length"] = int(context_length)
            
            # API server
            self._print_section("API Server")
            print("Configure the API server compatible with OpenAI's API.")
            
            api_enabled = self._get_bool_input(
                "Enable API server",
                self.config["api_server"]["enabled"]
            )
            self.config["api_server"]["enabled"] = api_enabled
            
            if api_enabled:
                self.config["api_server"]["host"] = self._get_input(
                    "API server host",
                    self.config["api_server"]["host"]
                )
                
                port = self._get_input(
                    "API server port",
                    str(self.config["api_server"]["port"])
                )
                self.config["api_server"]["port"] = int(port)
            
            # Advanced options
            self._print_section("Advanced Options")
            print("Configure advanced options for LlamaForge.")
            
            log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
            default_log_level_index = log_levels.index(self.config["advanced"]["log_level"]) if self.config["advanced"]["log_level"] in log_levels else 1
            log_level = self._get_choice_input(
                "Select the default log level:",
                log_levels,
                default_log_level_index
            )
            self.config["advanced"]["log_level"] = log_level
            
            plugins_enabled = self._get_bool_input(
                "Enable plugins",
                self.config["advanced"]["plugins_enabled"]
            )
            self.config["advanced"]["plugins_enabled"] = plugins_enabled
            
            max_cache_size = self._get_input(
                "Maximum cache size (in GB)",
                str(self.config["advanced"]["max_cache_size_gb"])
            )
            self.config["advanced"]["max_cache_size_gb"] = float(max_cache_size)
            
            # Save configuration
            self._print_section("Configuration Complete")
            print("Configuration complete! Saving...")
            self._save_config()
            
            print(f"\nConfiguration saved to: {self.config_path}")
            return self.config
            
        except KeyboardInterrupt:
            print("\n\nConfiguration cancelled.")
            return self.config
        
        except Exception as e:
            logger.error(f"Failed to run configuration wizard: {e}")
            print(f"\n\nAn error occurred: {e}")
            return self.config


def main():
    """Run the configuration wizard as a standalone script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Default configuration path
    default_config_path = os.path.expanduser("~/.llamaforge/config.json")
    
    # Allow specifying a different configuration path
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = default_config_path
    
    wizard = ConfigWizard(config_path)
    wizard.run_wizard()


if __name__ == "__main__":
    main() 