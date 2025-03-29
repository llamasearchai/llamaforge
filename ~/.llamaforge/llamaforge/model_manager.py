#!/usr/bin/env python3
"""
LlamaForge Model Manager
This module provides functionality for managing models, including downloading,
listing, and loading models for inference.
"""

import os
import sys
import json
import shutil
import hashlib
import logging
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Set up logging
logger = logging.getLogger("llamaforge.model_manager")


class ModelManager:
    """Manager for downloading, verifying, and loading models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models_dir = os.path.expanduser(config.get("directories", {}).get("models", "~/.llamaforge/models"))
        self.cache_dir = os.path.expanduser(config.get("directories", {}).get("cache", "~/.llamaforge/cache"))
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load the model registry
        self.registry = self.load_registry()
    
    def load_registry(self) -> Dict[str, Any]:
        """
        Load the model registry from disk.
        
        Returns:
            Dict containing the model registry
        """
        registry_path = os.path.join(self.models_dir, "registry.json")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                logger.info(f"Loaded model registry from {registry_path}")
                return registry
            except Exception as e:
                logger.error(f"Error loading model registry: {e}")
        
        # Create a new registry if one doesn't exist
        registry = {
            "models": {}
        }
        self.save_registry(registry)
        return registry
    
    def save_registry(self, registry: Dict[str, Any] = None) -> bool:
        """
        Save the model registry to disk.
        
        Args:
            registry: Registry to save (if None, use the current one)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if registry is None:
            registry = self.registry
        
        registry_path = os.path.join(self.models_dir, "registry.json")
        try:
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            logger.info(f"Saved model registry to {registry_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        for model_id, model_info in self.registry.get("models", {}).items():
            models.append({
                "id": model_id,
                **model_info
            })
        return models
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model_id: ID of the model to get information for
            
        Returns:
            Dict containing model information, or None if not found
        """
        model_info = self.registry.get("models", {}).get(model_id)
        if model_info:
            return {
                "id": model_id,
                **model_info
            }
        return None
    
    def add_model(self, model_id: str, model_info: Dict[str, Any]) -> bool:
        """
        Add a model to the registry.
        
        Args:
            model_id: ID to use for the model
            model_info: Information about the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        if model_id in self.registry.get("models", {}):
            logger.warning(f"Model {model_id} already exists in registry")
            return False
        
        self.registry.setdefault("models", {})[model_id] = model_info
        return self.save_registry()
    
    def remove_model(self, model_id: str, delete_files: bool = False) -> bool:
        """
        Remove a model from the registry.
        
        Args:
            model_id: ID of the model to remove
            delete_files: Whether to delete the model files
            
        Returns:
            bool: True if successful, False otherwise
        """
        if model_id not in self.registry.get("models", {}):
            logger.warning(f"Model {model_id} not found in registry")
            return False
        
        model_info = self.registry["models"][model_id]
        
        # Delete model files if requested
        if delete_files and "path" in model_info:
            model_path = os.path.join(self.models_dir, model_info["path"])
            try:
                if os.path.isdir(model_path):
                    shutil.rmtree(model_path)
                else:
                    os.remove(model_path)
                logger.info(f"Deleted model files at {model_path}")
            except Exception as e:
                logger.error(f"Error deleting model files: {e}")
                return False
        
        # Remove from registry
        del self.registry["models"][model_id]
        return self.save_registry()
    
    def download_model(self, repo_id: str, filename: str = None, backend: str = None) -> Optional[str]:
        """
        Download a model from Hugging Face.
        
        Args:
            repo_id: Hugging Face repository ID (e.g., 'TheBloke/Llama-2-7B-GGUF')
            filename: Specific filename to download
            backend: Backend to use for the model ('llamacpp', 'mlx', 'transformers')
            
        Returns:
            str: Model ID if successful, None otherwise
        """
        if backend is None:
            # Auto-detect appropriate backend
            if filename and filename.endswith('.gguf'):
                backend = 'llamacpp'
            elif self._is_mac_silicon():
                backend = 'mlx'
            else:
                backend = 'transformers'
        
        try:
            if backend == 'llamacpp':
                return self._download_gguf_model(repo_id, filename)
            elif backend == 'mlx':
                return self._download_mlx_model(repo_id)
            elif backend == 'transformers':
                return self._download_transformers_model(repo_id)
            else:
                logger.error(f"Unknown backend: {backend}")
                return None
        except Exception as e:
            logger.error(f"Error downloading model {repo_id}: {e}")
            return None
    
    def _download_gguf_model(self, repo_id: str, filename: str = None) -> Optional[str]:
        """
        Download a GGUF model for use with llama.cpp backend.
        
        Args:
            repo_id: Hugging Face repository ID
            filename: Specific filename to download (if None, download the first GGUF file)
            
        Returns:
            str: Model ID if successful, None otherwise
        """
        try:
            # Get model info from Hugging Face
            model_files = self._get_hf_model_files(repo_id)
            
            # Filter for GGUF files
            gguf_files = [f for f in model_files if f.endswith('.gguf')]
            if not gguf_files:
                logger.error(f"No GGUF files found in {repo_id}")
                return None
            
            # Select the file to download
            if filename:
                if filename not in gguf_files:
                    logger.error(f"File {filename} not found in {repo_id}")
                    return None
                target_file = filename
            else:
                # Try to find a reasonable default (prefer Q4_K_M if available)
                q4_files = [f for f in gguf_files if 'q4_k_m' in f.lower()]
                if q4_files:
                    target_file = q4_files[0]
                else:
                    # Just use the first GGUF file
                    target_file = gguf_files[0]
            
            # Create model directory
            model_id = f"{repo_id.split('/')[-1]}-{target_file.replace('.gguf', '')}"
            model_dir = os.path.join(self.models_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Download file
            model_path = os.path.join(model_dir, target_file)
            download_url = f"https://huggingface.co/{repo_id}/resolve/main/{target_file}"
            
            logger.info(f"Downloading {download_url} to {model_path}")
            self._download_file(download_url, model_path)
            
            # Add to registry
            model_info = {
                "name": model_id,
                "source": repo_id,
                "path": os.path.join(model_id, target_file),
                "backend": "llamacpp",
                "format": "gguf",
                "parameters": {},
                "metadata": {
                    "huggingface": repo_id,
                    "filename": target_file
                }
            }
            
            self.add_model(model_id, model_info)
            logger.info(f"Downloaded and registered model {model_id}")
            
            return model_id
            
        except Exception as e:
            logger.error(f"Error downloading GGUF model {repo_id}: {e}")
            return None
    
    def _download_mlx_model(self, repo_id: str) -> Optional[str]:
        """
        Download a model for use with MLX backend.
        
        Args:
            repo_id: Hugging Face repository ID
            
        Returns:
            str: Model ID if successful, None otherwise
        """
        try:
            # Check if we're on Mac with Apple Silicon
            if not self._is_mac_silicon():
                logger.error("MLX backend is only supported on Mac with Apple Silicon")
                return None
            
            # Create model directory
            model_id = repo_id.split('/')[-1]
            model_dir = os.path.join(self.models_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Use huggingface_hub to download the model
            try:
                import huggingface_hub
                logger.info(f"Downloading {repo_id} to {model_dir} using huggingface_hub")
                huggingface_hub.snapshot_download(
                    repo_id=repo_id,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
            except ImportError:
                # Fall back to git clone
                logger.info(f"huggingface_hub not available, using git clone for {repo_id}")
                subprocess.run(
                    ["git", "clone", f"https://huggingface.co/{repo_id}", model_dir],
                    check=True
                )
            
            # Add to registry
            model_info = {
                "name": model_id,
                "source": repo_id,
                "path": model_id,
                "backend": "mlx",
                "format": "safetensors",
                "parameters": {},
                "metadata": {
                    "huggingface": repo_id
                }
            }
            
            self.add_model(model_id, model_info)
            logger.info(f"Downloaded and registered model {model_id}")
            
            return model_id
            
        except Exception as e:
            logger.error(f"Error downloading MLX model {repo_id}: {e}")
            return None
    
    def _download_transformers_model(self, repo_id: str) -> Optional[str]:
        """
        Download a model for use with Transformers backend.
        
        Args:
            repo_id: Hugging Face repository ID
            
        Returns:
            str: Model ID if successful, None otherwise
        """
        try:
            # Create model directory
            model_id = repo_id.split('/')[-1]
            model_dir = os.path.join(self.models_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Use huggingface_hub to download the model
            try:
                import huggingface_hub
                logger.info(f"Downloading {repo_id} to {model_dir} using huggingface_hub")
                huggingface_hub.snapshot_download(
                    repo_id=repo_id,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
            except ImportError:
                # Fall back to git clone
                logger.info(f"huggingface_hub not available, using git clone for {repo_id}")
                subprocess.run(
                    ["git", "clone", f"https://huggingface.co/{repo_id}", model_dir],
                    check=True
                )
            
            # Add to registry
            model_info = {
                "name": model_id,
                "source": repo_id,
                "path": model_id,
                "backend": "transformers",
                "format": "transformers",
                "parameters": {},
                "metadata": {
                    "huggingface": repo_id
                }
            }
            
            self.add_model(model_id, model_info)
            logger.info(f"Downloaded and registered model {model_id}")
            
            return model_id
            
        except Exception as e:
            logger.error(f"Error downloading Transformers model {repo_id}: {e}")
            return None
    
    def import_local_model(self, model_path: str, model_id: str = None, backend: str = None) -> Optional[str]:
        """
        Import a local model file or directory.
        
        Args:
            model_path: Path to the model file or directory
            model_id: ID to use for the model (if None, derived from filename)
            backend: Backend to use ('llamacpp', 'mlx', 'transformers')
            
        Returns:
            str: Model ID if successful, None otherwise
        """
        try:
            # Validate path
            if not os.path.exists(model_path):
                logger.error(f"Model path {model_path} does not exist")
                return None
            
            # Determine model ID
            if model_id is None:
                if os.path.isdir(model_path):
                    model_id = os.path.basename(model_path)
                else:
                    model_id = os.path.splitext(os.path.basename(model_path))[0]
            
            # Auto-detect backend if not specified
            if backend is None:
                if model_path.endswith('.gguf'):
                    backend = 'llamacpp'
                elif os.path.isdir(model_path) and self._is_mac_silicon():
                    backend = 'mlx'
                else:
                    backend = 'transformers'
            
            # Create target directory/file in models directory
            target_path = os.path.join(self.models_dir, model_id)
            
            if os.path.isdir(model_path):
                # Directory-based model
                if os.path.exists(target_path):
                    logger.warning(f"Target directory {target_path} already exists, will not overwrite")
                    return None
                
                # Copy directory
                shutil.copytree(model_path, target_path)
                relative_path = model_id
                
                # Determine format
                if backend == 'transformers':
                    format_type = 'transformers'
                elif backend == 'mlx':
                    format_type = 'safetensors'
                else:
                    format_type = 'directory'
                
            else:
                # File-based model
                os.makedirs(target_path, exist_ok=True)
                target_file = os.path.join(target_path, os.path.basename(model_path))
                
                if os.path.exists(target_file):
                    logger.warning(f"Target file {target_file} already exists, will not overwrite")
                    return None
                
                # Copy file
                shutil.copy2(model_path, target_file)
                relative_path = os.path.join(model_id, os.path.basename(model_path))
                
                # Determine format
                if model_path.endswith('.gguf'):
                    format_type = 'gguf'
                elif model_path.endswith('.bin'):
                    format_type = 'bin'
                else:
                    format_type = 'file'
            
            # Add to registry
            model_info = {
                "name": model_id,
                "source": "local",
                "path": relative_path,
                "backend": backend,
                "format": format_type,
                "parameters": {},
                "metadata": {
                    "original_path": model_path
                }
            }
            
            self.add_model(model_id, model_info)
            logger.info(f"Imported and registered model {model_id}")
            
            return model_id
            
        except Exception as e:
            logger.error(f"Error importing local model {model_path}: {e}")
            return None
    
    def verify_model(self, model_id: str) -> bool:
        """
        Verify that a model exists and has all required files.
        
        Args:
            model_id: ID of the model to verify
            
        Returns:
            bool: True if the model is valid, False otherwise
        """
        model_info = self.get_model_info(model_id)
        if not model_info:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        # Check that model path exists
        model_path = os.path.join(self.models_dir, model_info["path"])
        if not os.path.exists(model_path):
            logger.error(f"Model path {model_path} does not exist")
            return False
        
        # Check backend-specific requirements
        backend = model_info.get("backend")
        
        if backend == "llamacpp":
            # For llama.cpp, just need the GGUF file
            if not model_path.endswith('.gguf'):
                logger.error(f"Model {model_id} is not a GGUF file")
                return False
            
        elif backend == "mlx" or backend == "transformers":
            # For MLX and Transformers, need a config.json file
            if os.path.isdir(model_path):
                config_path = os.path.join(model_path, "config.json")
                if not os.path.exists(config_path):
                    logger.error(f"Model {model_id} is missing config.json")
                    return False
            else:
                logger.error(f"Model {model_id} is not a directory")
                return False
        
        logger.info(f"Verified model {model_id}")
        return True
    
    def _get_hf_model_files(self, repo_id: str) -> List[str]:
        """
        Get a list of files in a Hugging Face repository.
        
        Args:
            repo_id: Hugging Face repository ID
            
        Returns:
            List of filenames
        """
        try:
            # Try using huggingface_hub if available
            import huggingface_hub
            repo_info = huggingface_hub.repo_info(repo_id)
            siblings = repo_info.siblings
            return [s.rfilename for s in siblings]
        except ImportError:
            # Fall back to API request
            url = f"https://huggingface.co/api/models/{repo_id}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            siblings = data.get("siblings", [])
            return [s.get("rfilename") for s in siblings if "rfilename" in s]
    
    def _download_file(self, url: str, target_path: str):
        """
        Download a file with progress reporting.
        
        Args:
            url: URL to download from
            target_path: Path to save the file to
        """
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        
        with open(target_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                    logger.info(f"Downloaded {downloaded / (1024 * 1024):.1f}MB / {total_size / (1024 * 1024):.1f}MB ({progress:.1f}%)")
    
    def _is_mac_silicon(self) -> bool:
        """
        Check if we're running on a Mac with Apple Silicon.
        
        Returns:
            bool: True if running on Mac with Apple Silicon, False otherwise
        """
        import platform
        return platform.system() == "Darwin" and platform.processor() == "arm" 