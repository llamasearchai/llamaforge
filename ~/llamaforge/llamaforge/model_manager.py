#!/usr/bin/env python3
"""
LlamaForge Model Manager

This module provides functionality for managing language models:
- Downloading models from Hugging Face
- Managing a registry of models
- Verifying model integrity
- Loading models for inference
"""

import os
import sys
import json
import shutil
import logging
import hashlib
import requests
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger("llamaforge.model_manager")

class ModelManager:
    """Manages model downloading, verification, and loading."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model manager with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models_dir = Path(os.path.expanduser(config.get("dirs", {}).get("models", "~/.llamaforge/models")))
        self.cache_dir = Path(os.path.expanduser(config.get("dirs", {}).get("cache", "~/.llamaforge/cache")))
        self.registry_path = self.models_dir / "registry.json"
        self.registry = self._load_registry()
        
        # Create necessary directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model manager initialized with models directory: {self.models_dir}")
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the model registry from disk.
        
        Returns:
            Dictionary mapping model IDs to model metadata
        """
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    registry = json.load(f)
                logger.info(f"Loaded model registry from {self.registry_path}")
                return registry
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse registry file {self.registry_path}, creating new registry")
        
        logger.info("Creating new model registry")
        return {}
    
    def _save_registry(self) -> None:
        """Save the model registry to disk."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
            logger.info(f"Saved model registry to {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models.
        
        Returns:
            Dictionary mapping model IDs to model metadata
        """
        return self.registry.copy()
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model metadata, or None if the model is not found
        """
        return self.registry.get(model_id)
    
    def add_model(self, model_id: str, model_info: Dict[str, Any]) -> bool:
        """Add a model to the registry.
        
        Args:
            model_id: ID of the model
            model_info: Model metadata
            
        Returns:
            True if the model was added successfully
        """
        if model_id in self.registry:
            logger.warning(f"Model {model_id} already exists in registry, updating")
        
        self.registry[model_id] = model_info
        self._save_registry()
        logger.info(f"Added model {model_id} to registry")
        return True
    
    def remove_model(self, model_id: str, delete_files: bool = False) -> bool:
        """Remove a model from the registry.
        
        Args:
            model_id: ID of the model
            delete_files: Whether to delete model files
            
        Returns:
            True if the model was removed successfully
        """
        if model_id not in self.registry:
            logger.warning(f"Model {model_id} not found in registry")
            return False
        
        if delete_files:
            model_path = Path(os.path.expanduser(self.registry[model_id].get("path", "")))
            if model_path.exists():
                try:
                    if model_path.is_dir():
                        shutil.rmtree(model_path)
                    else:
                        model_path.unlink()
                    logger.info(f"Deleted model files at {model_path}")
                except Exception as e:
                    logger.error(f"Failed to delete model files at {model_path}: {e}")
                    return False
        
        del self.registry[model_id]
        self._save_registry()
        logger.info(f"Removed model {model_id} from registry")
        return True
    
    def download_model(self, repo_id: str, model_id: Optional[str] = None, backend: Optional[str] = None, **kwargs) -> Optional[str]:
        """Download a model from Hugging Face.
        
        Args:
            repo_id: Hugging Face repository ID
            model_id: Local ID for the model (defaults to repo_id)
            backend: Backend to use for the model (llama.cpp, mlx, transformers)
            **kwargs: Additional arguments for the downloader
            
        Returns:
            Model ID if download was successful, None otherwise
        """
        # Default model_id to repo_id
        if model_id is None:
            model_id = repo_id.split("/")[-1]
        
        # Default backend to config setting or llama.cpp
        if backend is None:
            backend = self.config.get("model_defaults", {}).get("backend", "llama.cpp")
        
        logger.info(f"Downloading model {repo_id} as {model_id} for backend {backend}")
        
        try:
            # Call appropriate download method based on backend
            if backend == "llama.cpp":
                success = self._download_llamacpp_model(repo_id, model_id, **kwargs)
            elif backend == "mlx":
                success = self._download_mlx_model(repo_id, model_id, **kwargs)
            elif backend == "transformers":
                success = self._download_transformers_model(repo_id, model_id, **kwargs)
            else:
                logger.error(f"Unsupported backend: {backend}")
                return None
            
            if success:
                logger.info(f"Successfully downloaded model {model_id}")
                return model_id
            else:
                logger.error(f"Failed to download model {model_id}")
                return None
        
        except Exception as e:
            logger.error(f"Error downloading model {repo_id}: {e}")
            return None
    
    def _download_llamacpp_model(self, repo_id: str, model_id: str, **kwargs) -> bool:
        """Download a model for llama.cpp backend.
        
        Args:
            repo_id: Hugging Face repository ID
            model_id: Local ID for the model
            **kwargs: Additional arguments for the downloader
            
        Returns:
            True if download was successful
        """
        logger.info(f"Downloading llama.cpp model from {repo_id}")
        
        # Determine model file to download (usually GGUF)
        filename = kwargs.get("filename")
        if not filename:
            # Try to find a GGUF file in the repo
            try:
                from huggingface_hub import list_repo_files
                files = list_repo_files(repo_id)
                gguf_files = [f for f in files if f.endswith(".gguf")]
                
                if not gguf_files:
                    logger.error(f"No GGUF files found in repository {repo_id}")
                    return False
                
                # Sort by size or q-level if that information is available
                # For now, just take the last one which is often the best quality
                filename = gguf_files[-1]
                logger.info(f"Selected GGUF file: {filename}")
            
            except Exception as e:
                logger.error(f"Error finding GGUF files in repository: {e}")
                return False
        
        # Create model directory
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the model
        try:
            from huggingface_hub import hf_hub_download
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.cache_dir),
                local_dir=str(model_dir),
                resume_download=True,
                force_download=kwargs.get("force", False)
            )
            model_path = Path(file_path)
            logger.info(f"Downloaded model file to {model_path}")
            
            # Add to registry
            self.registry[model_id] = {
                "name": model_id,
                "backend": "llama.cpp",
                "repo_id": repo_id,
                "path": str(model_path),
                "filename": filename,
                "format": "gguf",
                "parameters": kwargs.get("parameters", {})
            }
            self._save_registry()
            
            return True
        
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return False
    
    def _download_mlx_model(self, repo_id: str, model_id: str, **kwargs) -> bool:
        """Download a model for MLX backend.
        
        Args:
            repo_id: Hugging Face repository ID
            model_id: Local ID for the model
            **kwargs: Additional arguments for the downloader
            
        Returns:
            True if download was successful
        """
        logger.info(f"Downloading MLX model from {repo_id}")
        
        # Create model directory
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use mlx_lm convert functionality if available
            convert_cmd = [
                sys.executable, "-m", "mlx_lm.convert", 
                "--hf-path", repo_id,
                "--mlx-path", str(model_dir)
            ]
            
            logger.info(f"Running conversion command: {' '.join(convert_cmd)}")
            result = subprocess.run(convert_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Conversion failed: {result.stderr}")
                return False
            
            logger.info(f"Conversion output: {result.stdout}")
            
            # Add to registry
            self.registry[model_id] = {
                "name": model_id,
                "backend": "mlx",
                "repo_id": repo_id,
                "path": str(model_dir),
                "format": "mlx",
                "parameters": kwargs.get("parameters", {})
            }
            self._save_registry()
            
            return True
        
        except Exception as e:
            logger.error(f"Error downloading MLX model: {e}")
            return False
    
    def _download_transformers_model(self, repo_id: str, model_id: str, **kwargs) -> bool:
        """Download a model for transformers backend.
        
        Args:
            repo_id: Hugging Face repository ID
            model_id: Local ID for the model
            **kwargs: Additional arguments for the downloader
            
        Returns:
            True if download was successful
        """
        logger.info(f"Downloading transformers model from {repo_id}")
        
        # Create model directory
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use transformers to download the model
            from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
            
            # First try to download the config to see if the model exists
            config = AutoConfig.from_pretrained(repo_id)
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir=str(self.cache_dir))
            tokenizer.save_pretrained(str(model_dir))
            
            # Only download model weights if requested
            if kwargs.get("download_weights", True):
                model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    cache_dir=str(self.cache_dir),
                    torch_dtype="auto",
                    device_map="auto" if kwargs.get("use_gpu", False) else None
                )
                model.save_pretrained(str(model_dir))
            
            # Add to registry
            self.registry[model_id] = {
                "name": model_id,
                "backend": "transformers",
                "repo_id": repo_id,
                "path": str(model_dir),
                "format": "transformers",
                "parameters": kwargs.get("parameters", {})
            }
            self._save_registry()
            
            return True
        
        except Exception as e:
            logger.error(f"Error downloading transformers model: {e}")
            return False
    
    def import_local_model(self, path: str, model_id: str, backend: str, **kwargs) -> Optional[str]:
        """Import a local model.
        
        Args:
            path: Path to the model file or directory
            model_id: ID for the model
            backend: Backend to use for the model
            **kwargs: Additional metadata for the model
            
        Returns:
            Model ID if import was successful, None otherwise
        """
        model_path = Path(os.path.expanduser(path))
        if not model_path.exists():
            logger.error(f"Model path {model_path} does not exist")
            return None
        
        logger.info(f"Importing local model from {model_path} as {model_id}")
        
        # Verify the model
        valid, format_type = self._verify_model(model_path, backend)
        if not valid:
            logger.error(f"Failed to verify model at {model_path}")
            return None
        
        # Add to registry
        self.registry[model_id] = {
            "name": model_id,
            "backend": backend,
            "path": str(model_path),
            "format": format_type,
            "parameters": kwargs.get("parameters", {})
        }
        self._save_registry()
        
        logger.info(f"Imported model {model_id}")
        return model_id
    
    def _verify_model(self, path: Path, backend: str) -> Tuple[bool, str]:
        """Verify a model's integrity.
        
        Args:
            path: Path to the model file or directory
            backend: Backend for the model
            
        Returns:
            Tuple of (is_valid, format_type)
        """
        logger.info(f"Verifying model at {path} for backend {backend}")
        
        if backend == "llama.cpp":
            # Check if it's a GGUF file
            if path.is_file() and path.suffix == ".gguf":
                return True, "gguf"
            else:
                logger.error(f"Expected a .gguf file for llama.cpp backend")
                return False, ""
        
        elif backend == "mlx":
            # Check for MLX model files
            if path.is_dir():
                required_files = ["config.json", "tokenizer.json", "weights.safetensors"]
                missing_files = [f for f in required_files if not (path / f).exists()]
                if missing_files:
                    logger.error(f"Missing required files for MLX model: {missing_files}")
                    return False, ""
                return True, "mlx"
            else:
                logger.error(f"Expected a directory for MLX backend")
                return False, ""
        
        elif backend == "transformers":
            # Check for transformers model files
            if path.is_dir():
                # At minimum we need a config.json and tokenizer
                if not (path / "config.json").exists():
                    logger.error(f"Missing config.json for transformers model")
                    return False, ""
                
                if not any((path / f).exists() for f in ["tokenizer.json", "tokenizer_config.json"]):
                    logger.error(f"Missing tokenizer files for transformers model")
                    return False, ""
                
                # Check if this is a safetensors model
                if list(path.glob("*.safetensors")):
                    return True, "safetensors"
                
                # Check if this is a pytorch model
                if list(path.glob("*.bin")):
                    return True, "pytorch"
                
                # Allow tokenizer-only models
                return True, "transformers"
            else:
                logger.error(f"Expected a directory for transformers backend")
                return False, ""
        
        else:
            logger.error(f"Unsupported backend: {backend}")
            return False, ""
    
    def load_model(self, model_id: str, **kwargs) -> Optional[Any]:
        """Load a model for inference.
        
        Args:
            model_id: ID of the model to load
            **kwargs: Additional parameters for loading
            
        Returns:
            Loaded model instance, or None if loading failed
        """
        if model_id not in self.registry:
            logger.error(f"Model {model_id} not found in registry")
            return None
        
        model_info = self.registry[model_id]
        backend = model_info.get("backend")
        model_path = Path(os.path.expanduser(model_info.get("path", "")))
        
        logger.info(f"Loading model {model_id} with backend {backend}")
        
        try:
            # Load model based on backend
            if backend == "llama.cpp":
                return self._load_llamacpp_model(model_path, model_info, **kwargs)
            elif backend == "mlx":
                return self._load_mlx_model(model_path, model_info, **kwargs)
            elif backend == "transformers":
                return self._load_transformers_model(model_path, model_info, **kwargs)
            else:
                logger.error(f"Unsupported backend: {backend}")
                return None
        
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None
    
    def _load_llamacpp_model(self, model_path: Path, model_info: Dict[str, Any], **kwargs) -> Optional[Any]:
        """Load a model for llama.cpp backend.
        
        Args:
            model_path: Path to the model file
            model_info: Model metadata
            **kwargs: Additional parameters for loading
            
        Returns:
            Loaded model instance, or None if loading failed
        """
        try:
            from llama_cpp import Llama
            
            # Get parameters from model_info, with override from kwargs
            params = {
                "n_ctx": kwargs.get("context_length", 
                         model_info.get("parameters", {}).get("context_length", 
                         self.config.get("model_defaults", {}).get("context_length", 4096))),
                "n_batch": kwargs.get("batch_size", 512),
                "n_gpu_layers": kwargs.get("n_gpu_layers", -1) if kwargs.get("use_gpu", True) else 0,
                "verbose": kwargs.get("verbose", False)
            }
            
            logger.info(f"Loading llama.cpp model from {model_path} with params: {params}")
            
            model = Llama(
                model_path=str(model_path),
                **params
            )
            
            return model
        
        except Exception as e:
            logger.error(f"Error loading llama.cpp model: {e}")
            return None
    
    def _load_mlx_model(self, model_path: Path, model_info: Dict[str, Any], **kwargs) -> Optional[Any]:
        """Load a model for MLX backend.
        
        Args:
            model_path: Path to the model directory
            model_info: Model metadata
            **kwargs: Additional parameters for loading
            
        Returns:
            Loaded model instance, or None if loading failed
        """
        try:
            # Import mlx modules
            import mlx.core as mx
            from mlx_lm import load, generate
            
            logger.info(f"Loading MLX model from {model_path}")
            
            # Load the model
            model, tokenizer = load(str(model_path))
            
            # Create a generator wrapper
            max_tokens = kwargs.get("max_tokens", 
                         model_info.get("parameters", {}).get("max_tokens", 
                         self.config.get("model_defaults", {}).get("max_tokens", 2048)))
            
            temperature = kwargs.get("temperature", 
                          model_info.get("parameters", {}).get("temperature", 
                          self.config.get("model_defaults", {}).get("temperature", 0.7)))
            
            generator = generate.Generator(
                model=model,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return generator
        
        except Exception as e:
            logger.error(f"Error loading MLX model: {e}")
            return None
    
    def _load_transformers_model(self, model_path: Path, model_info: Dict[str, Any], **kwargs) -> Optional[Any]:
        """Load a model for transformers backend.
        
        Args:
            model_path: Path to the model directory
            model_info: Model metadata
            **kwargs: Additional parameters for loading
            
        Returns:
            Loaded model instance, or None if loading failed
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            logger.info(f"Loading transformers model from {model_path}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            
            # Determine device map
            device_map = "auto" if kwargs.get("use_gpu", True) and torch.cuda.is_available() else None
            
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map=device_map,
                torch_dtype=torch.float16 if kwargs.get("use_fp16", True) else torch.float32
            )
            
            # Create text generation pipeline
            text_generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=kwargs.get("max_tokens", 
                            model_info.get("parameters", {}).get("max_tokens", 
                            self.config.get("model_defaults", {}).get("max_tokens", 2048))),
                temperature=kwargs.get("temperature", 
                             model_info.get("parameters", {}).get("temperature", 
                             self.config.get("model_defaults", {}).get("temperature", 0.7))),
                top_k=kwargs.get("top_k", 50),
                top_p=kwargs.get("top_p", 
                         model_info.get("parameters", {}).get("top_p", 
                         self.config.get("model_defaults", {}).get("top_p", 0.9))),
                do_sample=kwargs.get("do_sample", True)
            )
            
            return text_generator
        
        except Exception as e:
            logger.error(f"Error loading transformers model: {e}")
            return None
    
    def get_model_path(self, model_id: str) -> Optional[str]:
        """Get the path for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Path to the model, or None if the model is not found
        """
        if model_id not in self.registry:
            logger.error(f"Model {model_id} not found in registry")
            return None
        
        return self.registry[model_id].get("path")
    
    def set_model_parameter(self, model_id: str, parameter: str, value: Any) -> bool:
        """Set a parameter for a model.
        
        Args:
            model_id: ID of the model
            parameter: Parameter name
            value: Parameter value
            
        Returns:
            True if the parameter was set successfully
        """
        if model_id not in self.registry:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        if "parameters" not in self.registry[model_id]:
            self.registry[model_id]["parameters"] = {}
        
        self.registry[model_id]["parameters"][parameter] = value
        self._save_registry()
        logger.info(f"Set parameter {parameter}={value} for model {model_id}")
        return True
    
    def clean_cache(self, max_size_gb: Optional[float] = None) -> bool:
        """Clean the model cache.
        
        Args:
            max_size_gb: Maximum cache size in GB
            
        Returns:
            True if cleaning was successful
        """
        if max_size_gb is None:
            max_size_gb = self.config.get("advanced", {}).get("max_cache_size_gb", 1.0)
        
        logger.info(f"Cleaning cache, maximum size: {max_size_gb} GB")
        
        # Calculate current cache size
        cache_size = 0
        for path in self.cache_dir.glob("**/*"):
            if path.is_file():
                cache_size += path.stat().st_size
        
        cache_size_gb = cache_size / (1024 ** 3)
        logger.info(f"Current cache size: {cache_size_gb:.2f} GB")
        
        if cache_size_gb <= max_size_gb:
            logger.info("Cache size is within limits, no cleaning needed")
            return True
        
        # Clean cache
        try:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleaned successfully")
            return True
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
            return False 