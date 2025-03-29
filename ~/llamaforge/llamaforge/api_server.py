#!/usr/bin/env python3
"""
LlamaForge API Server

This module provides a FastAPI server that implements a subset of the OpenAI API,
allowing applications that use OpenAI's API to use LlamaForge as a drop-in replacement.
"""

import os
import sys
import json
import time
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

logger = logging.getLogger("llamaforge.api_server")

# Data models for API requests and responses
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    @validator('messages')
    def validate_messages(cls, messages):
        if not messages:
            raise ValueError("At least one message is required")
        return messages

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "llamaforge"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelData]

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: Usage

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: Usage

class APIServer:
    """API server that implements a subset of the OpenAI API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the API server with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.app = self._create_app()
        
        # Reference to the LlamaForge instance
        # In a real implementation, this would be initialized with a proper instance
        self.llamaforge = None
    
    def _create_app(self) -> FastAPI:
        """Create the FastAPI application.
        
        Returns:
            FastAPI application
        """
        app = FastAPI(
            title="LlamaForge API",
            description="LlamaForge API Server",
            version="0.1.0"
        )
        
        # Configure CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Define routes
        @app.get("/v1/models", response_model=ModelsResponse)
        async def list_models():
            """List available models."""
            # In a real implementation, this would get the list of models from LlamaForge
            models = []
            if self.llamaforge and hasattr(self.llamaforge, 'model_manager'):
                model_list = self.llamaforge.model_manager.list_models()
                for model_id, model_info in model_list.items():
                    models.append(ModelData(id=model_id))
            else:
                # Return a placeholder model when not properly initialized
                models.append(ModelData(id="llamaforge-placeholder"))
            
            return ModelsResponse(data=models)
        
        @app.get("/v1/models/{model_id}", response_model=ModelData)
        async def get_model(model_id: str):
            """Get information about a specific model."""
            # In a real implementation, this would get the model info from LlamaForge
            if self.llamaforge and hasattr(self.llamaforge, 'model_manager'):
                model_info = self.llamaforge.model_manager.get_model_info(model_id)
                if model_info:
                    return ModelData(id=model_id)
            
            # Return a placeholder when not properly initialized or model not found
            if model_id == "llamaforge-placeholder":
                return ModelData(id=model_id)
            
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def create_chat_completion(request: ChatCompletionRequest):
            """Create a chat completion."""
            if request.stream:
                # For streaming, return a StreamingResponse
                return StreamingResponse(
                    self._stream_chat_completion(request),
                    media_type="text/event-stream"
                )
            
            try:
                # In a real implementation, this would call the appropriate LlamaForge method
                response = self._generate_chat_completion(request)
                return response
            except Exception as e:
                logger.error(f"Error generating chat completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/v1/completions", response_model=CompletionResponse)
        async def create_completion(request: CompletionRequest):
            """Create a text completion."""
            if request.stream:
                # For streaming, return a StreamingResponse
                return StreamingResponse(
                    self._stream_completion(request),
                    media_type="text/event-stream"
                )
            
            try:
                # In a real implementation, this would call the appropriate LlamaForge method
                response = self._generate_completion(request)
                return response
            except Exception as e:
                logger.error(f"Error generating completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "ok"}
        
        return app
    
    async def _stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Stream a chat completion.
        
        Args:
            request: Chat completion request
            
        Yields:
            Streamed response chunks
        """
        # Generate a unique ID for this completion
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())
        model = request.model
        
        # Convert the request to a prompt
        prompt = self._messages_to_prompt(request.messages)
        
        # Initial response with empty content
        initial_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"
        
        # In a real implementation, this would call the appropriate LlamaForge method
        # For now, we'll just simulate streaming by yielding chunks of a placeholder message
        response_text = f"This is a placeholder streaming response from model {model}."
        
        for i, char in enumerate(response_text):
            await asyncio.sleep(0.02)  # Simulate thinking time
            
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": char},
                        "finish_reason": None
                    }
                ]
            }
            
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # Final chunk with finish_reason
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        
        # End of stream
        yield "data: [DONE]\n\n"
    
    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Stream a text completion.
        
        Args:
            request: Completion request
            
        Yields:
            Streamed response chunks
        """
        # Generate a unique ID for this completion
        completion_id = f"cmpl-{uuid.uuid4()}"
        created = int(time.time())
        model = request.model
        
        # Get the prompt
        prompt = request.prompt
        if isinstance(prompt, list):
            prompt = prompt[0]
        
        # In a real implementation, this would call the appropriate LlamaForge method
        # For now, we'll just simulate streaming by yielding chunks of a placeholder message
        response_text = f"This is a placeholder streaming response from model {model}."
        
        for i, char in enumerate(response_text):
            await asyncio.sleep(0.02)  # Simulate thinking time
            
            chunk = {
                "id": completion_id,
                "object": "text_completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "text": char,
                        "finish_reason": None
                    }
                ]
            }
            
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # Final chunk with finish_reason
        final_chunk = {
            "id": completion_id,
            "object": "text_completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        
        # End of stream
        yield "data: [DONE]\n\n"
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert a list of messages to a prompt format.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Formatted prompt string
        """
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"<|system|>\n{message.content}\n"
            elif message.role == "user":
                prompt += f"<|user|>\n{message.content}\n"
            elif message.role == "assistant":
                prompt += f"<|assistant|>\n{message.content}\n"
            else:
                prompt += f"<|{message.role}|>\n{message.content}\n"
        
        prompt += "<|assistant|>\n"
        return prompt
    
    def _generate_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a chat completion.
        
        Args:
            request: Chat completion request
            
        Returns:
            Chat completion response
        """
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        model = request.model
        
        # Convert the request to a prompt
        prompt = self._messages_to_prompt(request.messages)
        
        # In a real implementation, this would call the appropriate LlamaForge method
        # For now, we'll just return a placeholder response
        response_text = f"This is a placeholder response from model {model}."
        
        # Create the response
        response = ChatCompletionResponse(
            id=completion_id,
            model=model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt) // 4,  # Very rough approximation
                completion_tokens=len(response_text) // 4,
                total_tokens=(len(prompt) + len(response_text)) // 4
            )
        )
        
        return response
    
    def _generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a text completion.
        
        Args:
            request: Completion request
            
        Returns:
            Completion response
        """
        completion_id = f"cmpl-{uuid.uuid4()}"
        model = request.model
        
        # Get the prompt
        prompt = request.prompt
        if isinstance(prompt, list):
            prompt = prompt[0]
        
        # In a real implementation, this would call the appropriate LlamaForge method
        # For now, we'll just return a placeholder response
        response_text = f"This is a placeholder response from model {model}."
        
        # Create the response
        response = CompletionResponse(
            id=completion_id,
            model=model,
            choices=[
                CompletionChoice(
                    index=0,
                    text=response_text,
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt) // 4,  # Very rough approximation
                completion_tokens=len(response_text) // 4,
                total_tokens=(len(prompt) + len(response_text)) // 4
            )
        )
        
        return response
    
    def start(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Start the API server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port) 