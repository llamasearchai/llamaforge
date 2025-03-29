#!/usr/bin/env python3
"""
LlamaForge Configuration Wizard
This module provides an interactive configuration wizard for setting up LlamaForge.
"""

import os
import sys
import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Set up logging
logger = logging.getLogger("llamaforge.config_wizard")


class ConfigWizard:
    def __init__(self, config_path: str = "~/.llamaforge/config.json"):
        """Initialize the configuration wizard."""
        self.config_path = os.path.expanduser(config_path)
        self.config = self._load_config()
        self.detected_info = self._detect_system()
    
    def _load_config(self) -> Dict:
        """Load existing configuration or create a new one with defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded existing configuration from {self.config_path}")
                    return config
            except Exception as e:
                logger.warning(f"Error loading configuration: {e}. Creating new configuration.")
        
        # Default configuration
        config = {
            "directories": {
                "models": os.path.expanduser("~/.llamaforge/models"),
                "cache": os.path.expanduser("~/.llamaforge/cache"),
                "logs": os.path.expanduser("~/.llamaforge/logs"),
                "plugins": os.path.expanduser("~/.llamaforge/plugins")
            },
            "model_defaults": {
                "context_length": 4096,
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            },
            "api_server": {
                "enabled": False,
                "host": "127.0.0.1",
                "port": 8000
            },
            "advanced": {
                "check_updates": True,
                "telemetry": False,
                "debug": False,
                "backend_preference": []  # Will be populated based on system detection
            }
        }
        
        return config
    
    def _detect_system(self) -> Dict:
        """Detect system information to help with configuration."""
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "available_backends": ["cpu"],  # CPU is always available
            "gpu_info": []
        }
        
        # Detect CUDA
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                # Parse GPU information
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            gpu_name = parts[0].strip()
                            gpu_memory = parts[1].strip()
                            info["gpu_info"].append({"name": gpu_name, "memory": gpu_memory})
                
                if info["gpu_info"]:
                    info["available_backends"].append("cuda")
                    # Set cuda as preferred backend if available
                    self.config["advanced"]["backend_preference"] = ["cuda", "cpu"]
        except Exception as e:
            logger.debug(f"Error detecting NVIDIA GPUs: {e}")
        
        # Detect Metal (macOS)
        if info["os"] == "Darwin":
            # Check for Apple Silicon
            if info["architecture"] == "arm64":
                info["available_backends"].append("metal")
                # Set metal as preferred backend on Apple Silicon
                self.config["advanced"]["backend_preference"] = ["metal", "cpu"]
                info["gpu_info"].append({"name": "Apple Silicon", "memory": "Shared"})
        
        return info
    
    def _save_config(self):
        """Save the configuration to file."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def print_header(self, text: str):
        """Print a header with decoration."""
        print("\n" + "=" * 60)
        print(f" {text}")
        print("=" * 60)
    
    def print_section(self, text: str):
        """Print a section header."""
        print("\n" + "-" * 40)
        print(f" {text}")
        print("-" * 40)
    
    def get_input(self, prompt: str, default: Any = None) -> str:
        """Get input from the user with a default value."""
        if default is not None:
            prompt = f"{prompt} [{default}]: "
        else:
            prompt = f"{prompt}: "
        
        value = input(prompt).strip()
        if not value and default is not None:
            return str(default)
        return value
    
    def get_boolean_input(self, prompt: str, default: bool = None) -> bool:
        """Get a boolean input from the user."""
        if default is not None:
            default_str = "Y" if default else "N"
            prompt = f"{prompt} [Y/N] [{default_str}]: "
        else:
            prompt = f"{prompt} [Y/N]: "
        
        while True:
            value = input(prompt).strip().lower()
            if not value and default is not None:
                return default
            
            if value in ['y', 'yes', 'true', '1']:
                return True
            elif value in ['n', 'no', 'false', '0']:
                return False
            else:
                print("Please enter Y or N.")
    
    def get_choice_input(self, prompt: str, choices: List[str], default: str = None) -> str:
        """Get a choice from a list of options."""
        print(f"\n{prompt}")
        for i, choice in enumerate(choices, 1):
            if choice == default:
                print(f"{i}. {choice} (default)")
            else:
                print(f"{i}. {choice}")
        
        default_index = choices.index(default) + 1 if default in choices else None
        default_str = str(default_index) if default_index else ""
        
        while True:
            value = input(f"Enter your choice [1-{len(choices)}] [{default_str}]: ").strip()
            
            if not value and default_index:
                return choices[default_index - 1]
            
            try:
                index = int(value)
                if 1 <= index <= len(choices):
                    return choices[index - 1]
                else:
                    print(f"Please enter a number between 1 and {len(choices)}.")
            except ValueError:
                print("Please enter a valid number.")
    
    def run_wizard(self):
        """Run the interactive configuration wizard."""
        self.print_header("LlamaForge Configuration Wizard")
        print("This wizard will help you set up LlamaForge on your system.")
        print("Press Enter to accept default values shown in [brackets].")
        
        # Display system information
        self.print_section("System Information")
        print(f"Operating System: {self.detected_info['os']} {self.detected_info['os_version']}")
        print(f"Architecture: {self.detected_info['architecture']}")
        print(f"Python Version: {self.detected_info['python_version']}")
        
        print("\nDetected Hardware:")
        if self.detected_info["gpu_info"]:
            for gpu in self.detected_info["gpu_info"]:
                print(f"- {gpu['name']} ({gpu['memory']})")
        else:
            print("- No GPU detected, using CPU only")
        
        print("\nAvailable Backends:")
        for backend in self.detected_info["available_backends"]:
            print(f"- {backend}")
        
        # Configure directories
        self.print_section("Directory Configuration")
        directories = self.config["directories"]
        
        for key, value in directories.items():
            new_value = self.get_input(f"{key.capitalize()} directory", value)
            directories[key] = os.path.expanduser(new_value)
            
            # Create directory if it doesn't exist
            os.makedirs(directories[key], exist_ok=True)
        
        # Configure model defaults
        self.print_section("Model Defaults")
        model_defaults = self.config["model_defaults"]
        
        model_defaults["context_length"] = int(self.get_input(
            "Default context length", model_defaults["context_length"]))
        model_defaults["max_tokens"] = int(self.get_input(
            "Default maximum tokens to generate", model_defaults["max_tokens"]))
        model_defaults["temperature"] = float(self.get_input(
            "Default temperature (0.0-1.0)", model_defaults["temperature"]))
        model_defaults["top_p"] = float(self.get_input(
            "Default top_p (0.0-1.0)", model_defaults["top_p"]))
        model_defaults["repeat_penalty"] = float(self.get_input(
            "Default repeat penalty (1.0+)", model_defaults["repeat_penalty"]))
        
        # Configure API server
        self.print_section("API Server Configuration")
        api_server = self.config["api_server"]
        
        api_server["enabled"] = self.get_boolean_input(
            "Enable the API server (OpenAI-compatible)", api_server["enabled"])
        
        if api_server["enabled"]:
            api_server["host"] = self.get_input("API server host", api_server["host"])
            api_server["port"] = int(self.get_input("API server port", api_server["port"]))
        
        # Configure advanced options
        self.print_section("Advanced Options")
        advanced = self.config["advanced"]
        
        advanced["check_updates"] = self.get_boolean_input(
            "Check for updates automatically", advanced["check_updates"])
        advanced["telemetry"] = self.get_boolean_input(
            "Send anonymous usage data", advanced["telemetry"])
        advanced["debug"] = self.get_boolean_input(
            "Enable debug logging", advanced["debug"])
        
        # Configure backend preference
        if len(self.detected_info["available_backends"]) > 1:
            self.print_section("Backend Preference")
            print("Choose your preferred backend order:")
            
            remaining_backends = self.detected_info["available_backends"].copy()
            backend_preference = []
            
            while remaining_backends:
                backend = self.get_choice_input(
                    "Select next preferred backend:",
                    remaining_backends,
                    remaining_backends[0] if remaining_backends else None
                )
                backend_preference.append(backend)
                remaining_backends.remove(backend)
            
            advanced["backend_preference"] = backend_preference
        
        # Save configuration
        self.print_section("Save Configuration")
        if self.get_boolean_input("Save the configuration", True):
            if self._save_config():
                print("\nConfiguration has been saved successfully!")
                print(f"Configuration file: {self.config_path}")
            else:
                print("\nFailed to save the configuration. Please check permissions and try again.")
        else:
            print("\nConfiguration was not saved.")
        
        return self.config


def main():
    """Run the configuration wizard when called directly."""
    wizard = ConfigWizard()
    wizard.run_wizard()


if __name__ == "__main__":
    main() 