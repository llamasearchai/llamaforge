#!/usr/bin/env python3
"""
Example API client for LlamaForge API server.

This example demonstrates how to use the LlamaForge API server
as a drop-in replacement for the OpenAI API.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests

API_BASE = "http://localhost:8000/v1"
API_KEY = "dummy-api-key"  # Not used but included for OpenAI API compatibility


class LlamaForgeClient:
    """Simple client for the LlamaForge API server."""
    
    def __init__(self, base_url: str = API_BASE, api_key: str = API_KEY):
        """Initialize the client with the API base URL and key."""
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Author: Nik Jois
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        response = requests.get(f"{self.base_url}/models", headers=self.headers)
        response.raise_for_status()
        return response.json()["data"]
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        response = requests.get(f"{self.base_url}/models/{model_id}", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Create a chat completion."""
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        if stream:
            return self._stream_chat_completion(data)
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def _stream_chat_completion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Stream a chat completion."""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data,
            stream=True
        )
        response.raise_for_status()
        
        # Process the streaming response
        content = ""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if chunk.get("choices") and chunk["choices"][0].get("delta") and chunk["choices"][0]["delta"].get("content"):
                            content_chunk = chunk["choices"][0]["delta"]["content"]
                            content += content_chunk
                            print(content_chunk, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        
        print()  # Add a newline at the end
        
        # Return a simulated complete response
        return {
            "id": "chatcmpl-simulated",
            "object": "chat.completion",
            "created": 0,
            "model": data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    
    def completion(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Create a text completion."""
        data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        if stream:
            return self._stream_completion(data)
        
        response = requests.post(
            f"{self.base_url}/completions",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def _stream_completion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Stream a text completion."""
        response = requests.post(
            f"{self.base_url}/completions",
            headers=self.headers,
            json=data,
            stream=True
        )
        response.raise_for_status()
        
        # Process the streaming response
        content = ""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if chunk.get("choices") and chunk["choices"][0].get("text"):
                            content_chunk = chunk["choices"][0]["text"]
                            content += content_chunk
                            print(content_chunk, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        
        print()  # Add a newline at the end
        
        # Return a simulated complete response
        return {
            "id": "cmpl-simulated",
            "object": "text_completion",
            "created": 0,
            "model": data["model"],
            "choices": [
                {
                    "text": content,
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


def main():
    """Run the example API client."""
    print("LlamaForge API Client Example")
    print("=============================")
    
    client = LlamaForgeClient()
    
    # Check if the API server is running
    try:
        models = client.list_models()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the LlamaForge API server.")
        print("Make sure the server is running with: llamaforge api")
        return
    
    print(f"Connected to LlamaForge API server at {API_BASE}")
    print(f"Available models: {len(models)}")
    
    for model in models:
        print(f"- {model['id']}")
    
    if not models:
        print("No models available. Please add a model first.")
        return
    
    # Use the first model
    model_id = models[0]["id"]
    print(f"\nUsing model: {model_id}")
    
    # Example chat completion
    print("\nChat Completion Example:")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"}
    ]
    
    print("User: Hello, who are you?")
    print("Assistant: ", end="", flush=True)
    
    try:
        response = client.chat_completion(model_id, messages, stream=True)
    except requests.exceptions.HTTPError as e:
        print(f"Error: {e}")
        return
    
    # Example text completion
    print("\nText Completion Example:")
    prompt = "Once upon a time"
    
    print(f"Prompt: {prompt}")
    print("Completion: ", end="", flush=True)
    
    try:
        response = client.completion(model_id, prompt, stream=True)
    except requests.exceptions.HTTPError as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main() 