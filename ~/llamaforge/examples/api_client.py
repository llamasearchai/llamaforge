#!/usr/bin/env python3
"""
Example of using LlamaForge's API server with a client.

This example demonstrates how to interact with LlamaForge's API server
using the requests library, with the same interface as OpenAI's API.
"""

import os
import sys
import json
import argparse
import requests
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("llamaforge_api_client")

class LlamaForgeClient:
    """Client for the LlamaForge API server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client with the API server URL."""
        self.base_url = base_url
        self.session = requests.Session()
        
        # Check if the server is running
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code != 200:
                logger.error(f"API server returned unexpected status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                raise ConnectionError("API server is not healthy")
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to API server at {self.base_url}")
            logger.error("Make sure the server is running using: llamaforge api")
            raise ConnectionError("Could not connect to API server")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        response = self.session.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()["data"]
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Create a chat completion."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        if stream:
            return self._stream_chat_completion(payload)
        else:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            return response.json()
    
    def _stream_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Stream a chat completion."""
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            stream=True
        )
        response.raise_for_status()
        
        # Process the streaming response
        collected_messages = []
        for line in response.iter_lines():
            # Skip empty lines
            if not line:
                continue
            
            # Remove "data: " prefix
            if line.startswith(b"data: "):
                line = line[6:]
            
            # Check for end of stream
            if line.strip() == b"[DONE]":
                break
            
            try:
                # Parse the JSON chunk
                chunk = json.loads(line)
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    print(content, end="", flush=True)
                    collected_messages.append(content)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing chunk: {e}")
                logger.error(f"Chunk: {line}")
        
        # Print a newline after streaming is done
        print()
        
        # Return a constructed response similar to non-streaming response
        return {
            "id": "stream",
            "object": "chat.completion",
            "created": 0,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(collected_messages)
                },
                "finish_reason": "stop"
            }]
        }

def main():
    """Run the API client example."""
    parser = argparse.ArgumentParser(description="LlamaForge API Client Example")
    parser.add_argument("--host", default="localhost", help="API server host")
    parser.add_argument("--port", default=8000, type=int, help="API server port")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    try:
        # Initialize the client
        client = LlamaForgeClient(base_url)
        
        # List available models
        models = client.list_models()
        if not models:
            logger.error("No models available on the server.")
            return 1
        
        # Select the first model
        model_id = models[0]["id"]
        print(f"Using model: {model_id}")
        
        # Welcome message
        print("\n🌐 LlamaForge API Client Example")
        print("--------------------------------")
        print(f"Connected to API server at {base_url}")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("--------------------------------\n")
        
        # Chat loop
        messages = []
        while True:
            # Get user input
            user_input = input("You: ")
            
            # Check if the user wants to exit
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            # Add to messages
            messages.append({"role": "user", "content": user_input})
            
            try:
                # Get model response
                if not args.stream:
                    print("Model: ", end="")
                
                response = client.chat_completion(
                    messages=messages,
                    model=model_id,
                    stream=args.stream
                )
                
                # Process and display the response
                if not args.stream:
                    assistant_message = response["choices"][0]["message"]["content"]
                    print(assistant_message)
                    messages.append({"role": "assistant", "content": assistant_message})
                else:
                    assistant_message = response["choices"][0]["message"]["content"]
                    messages.append({"role": "assistant", "content": assistant_message})
                
            except Exception as e:
                logger.error(f"Error getting response: {e}")
                print("Sorry, there was an error generating a response.")
        
        return 0
    
    except ConnectionError:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 