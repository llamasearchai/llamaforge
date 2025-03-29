#!/usr/bin/env python
"""
API server demo for LlamaForge.

This example demonstrates how to start the API server and make
requests to it using the OpenAI client.
"""

import sys
import os
import argparse
import logging
import time
import subprocess
import requests
import json
from typing import Dict, Any, List

# Add parent directory to path to allow importing llamaforge
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def check_server_ready(host: str, port: int, max_retries: int = 10, retry_delay: float = 1.0) -> bool:
    """
    Check if the server is ready to accept connections.
    
    Args:
        host: Server hostname
        port: Server port
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        bool: True if server is ready, False otherwise
    """
    for i in range(max_retries):
        try:
            response = requests.get(f"http://{host}:{port}/v1/models")
            if response.status_code == 200:
                logger.info("Server is ready")
                return True
        except requests.RequestException:
            pass
        
        logger.info(f"Waiting for server to start (attempt {i+1}/{max_retries})...")
        time.sleep(retry_delay)
    
    logger.error("Server failed to start")
    return False

def make_completion_request(host: str, port: int, model: str, prompt: str) -> Dict[str, Any]:
    """
    Make a completion request to the server.
    
    Args:
        host: Server hostname
        port: Server port
        model: Model name
        prompt: Input prompt
        
    Returns:
        Dict[str, Any]: Response data
    """
    url = f"http://{host}:{port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def make_chat_request(host: str, port: int, model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Make a chat completion request to the server.
    
    Args:
        host: Server hostname
        port: Server port
        model: Model name
        messages: Chat messages
        
    Returns:
        Dict[str, Any]: Response data
    """
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.7,
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def demo_openai_client(host: str, port: int, model: str) -> None:
    """
    Demonstrate using the OpenAI client with the LlamaForge API server.
    
    Args:
        host: Server hostname
        port: Server port
        model: Model name
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("OpenAI Python client not installed. Try: pip install openai")
        return
    
    client = OpenAI(
        api_key = "REDACTED",  # Not used but required
        base_url=f"http://{host}:{port}/v1"
    )
    
    # Chat completion
    logger.info("Making chat completion request with OpenAI client...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What can you tell me about language models?"}
        ]
    )
    
    logger.info(f"Chat response: {response.choices[0].message.content}")

def main():
    parser = argparse.ArgumentParser(description="API server demo for LlamaForge")
    parser.add_argument("--model", required=True, help="Model to use")
    parser.add_argument("--host", default="127.0.0.1", help="Server hostname")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--start-server", action="store_true", help="Start server in a subprocess")
    args = parser.parse_args()
    
    server_process = None
    
    try:
        if args.start_server:
            # Start the server in a subprocess
            logger.info(f"Starting server on {args.host}:{args.port}...")
            server_process = subprocess.Popen(
                [
                    sys.executable, "-m", "llamaforge.main", 
                    "serve", 
                    "--host", args.host, 
                    "--port", str(args.port),
                    "--model", args.model
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            if not check_server_ready(args.host, args.port):
                if server_process:
                    server_process.terminate()
                return
        
        # Check if server is running
        try:
            requests.get(f"http://{args.host}:{args.port}/v1/models")
        except requests.RequestException:
            logger.error(f"Server not running at {args.host}:{args.port}")
            return
        
        # List available models
        logger.info("Listing available models...")
        try:
            response = requests.get(f"http://{args.host}:{args.port}/v1/models")
            models = response.json()
            logger.info(f"Available models: {models}")
        except requests.RequestException as e:
            logger.error(f"Error listing models: {e}")
            return
        
        # Make completion request
        logger.info("Making completion request...")
        try:
            response = make_completion_request(
                args.host, args.port, args.model,
                "Explain what a language model is in simple terms."
            )
            logger.info(f"Completion response: {response['choices'][0]['text']}")
        except (requests.RequestException, KeyError) as e:
            logger.error(f"Error making completion request: {e}")
        
        # Make chat request
        logger.info("Making chat request...")
        try:
            response = make_chat_request(
                args.host, args.port, args.model,
                [
                    {"role": "user", "content": "Hello, can you introduce yourself?"}
                ]
            )
            logger.info(f"Chat response: {response['choices'][0]['message']['content']}")
        except (requests.RequestException, KeyError) as e:
            logger.error(f"Error making chat request: {e}")
        
        # Demo OpenAI client
        logger.info("Demonstrating OpenAI client...")
        demo_openai_client(args.host, args.port, args.model)
        
    finally:
        if server_process:
            logger.info("Stopping server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()

if __name__ == "__main__":
    main() 