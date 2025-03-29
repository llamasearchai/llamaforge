#!/usr/bin/env python3
"""
LlamaForge API Server
---------------------
This module provides a FastAPI server that implements a subset of the OpenAI API,
allowing you to use LlamaForge as a drop-in replacement for applications that use OpenAI's API.
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union

# Import FastAPI
from fastapi import FastAPI, HTTPException, Depends, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# Import LlamaForge components
from .model_manager import ModelManager

# Set up logging
logger = logging.getLogger("llamaforge.api_server")


# API request/response models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
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
    finish_reason: Optional[str] = "stop"


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = "stop"


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
    def __init__(self, config: Dict):
        self.config = config
        self.model_manager = ModelManager(config)
        self.app = self._create_app()
        
        # Register shutdown handlers
        import atexit
        atexit.register(self.shutdown)
    
    def _create_app(self):
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title="LlamaForge API",
            description="An API server compatible with OpenAI's API for local language models",
            version="1.0.0"
        )
        
        # Set up CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allow all methods
            allow_headers=["*"],  # Allow all headers
        )
        
        # Model endpoints
        @app.get("/v1/models", response_model=ModelsResponse)
        async def list_models():
            """List available models."""
            models = self.model_manager.list_models()
            model_data = [
                ModelData(
                    id=model["id"],
                    object="model",
                    created=int(time.time()),
                    owned_by="llamaforge"
                )
                for model in models
            ]
            return ModelsResponse(data=model_data)
        
        @app.get("/v1/models/{model_id}")
        async def get_model(model_id: str):
            """Get information about a specific model."""
            model_info = self.model_manager.get_model_info(model_id)
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
            return ModelData(
                id=model_info["id"],
                object="model",
                created=int(time.time()),
                owned_by="llamaforge"
            )
        
        # Chat completions endpoint
        @app.post("/v1/chat/completions")
        async def create_chat_completion(request: ChatCompletionRequest):
            """Create a chat completion."""
            model_id = request.model
            model_info = self.model_manager.get_model_info(model_id)
            
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
            # For streaming responses
            if request.stream:
                return StreamingResponse(
                    self._stream_chat_completion(request, model_info),
                    media_type="text/event-stream"
                )
            
            # For non-streaming responses
            try:
                backend = model_info.get("backend")
                
                # Convert messages to prompt
                messages = request.messages
                prompt = self._messages_to_prompt(messages, model_info)
                
                # Generate completion
                completion = self._generate_chat_completion(
                    prompt=prompt,
                    model_info=model_info,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p
                )
                
                # Estimate token counts (very approximate)
                prompt_tokens = len(prompt) // 4
                completion_tokens = len(completion) // 4
                
                # Create response
                response = ChatCompletionResponse(
                    id=f"llamaforge-{int(time.time())}-{model_id}",
                    model=model_id,
                    choices=[
                        ChatChoice(
                            index=0,
                            message=ChatMessage(
                                role="assistant",
                                content=completion
                            ),
                            finish_reason="stop"
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens
                    )
                )
                
                return response
            
            except Exception as e:
                logger.error(f"Error generating chat completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Text completions endpoint
        @app.post("/v1/completions")
        async def create_completion(request: CompletionRequest):
            """Create a text completion."""
            model_id = request.model
            model_info = self.model_manager.get_model_info(model_id)
            
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
            # For streaming responses
            if request.stream:
                return StreamingResponse(
                    self._stream_completion(request, model_info),
                    media_type="text/event-stream"
                )
            
            # For non-streaming responses
            try:
                # Handle either string or list of strings for prompt
                prompt = request.prompt
                if isinstance(prompt, list):
                    prompt = prompt[0]  # Just use the first prompt for now
                
                # Generate completion
                completion = self._generate_completion(
                    prompt=prompt,
                    model_info=model_info,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p
                )
                
                # Estimate token counts (very approximate)
                prompt_tokens = len(prompt) // 4
                completion_tokens = len(completion) // 4
                
                # Create response
                response = CompletionResponse(
                    id=f"llamaforge-{int(time.time())}-{model_id}",
                    model=model_id,
                    choices=[
                        CompletionChoice(
                            index=0,
                            text=completion,
                            finish_reason="stop"
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens
                    )
                )
                
                return response
            
            except Exception as e:
                logger.error(f"Error generating completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Health check endpoint
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "ok"}
        
        return app
    
    async def _stream_chat_completion(self, request: ChatCompletionRequest, model_info: Dict):
        """Stream a chat completion response."""
        model_id = request.model
        messages = request.messages
        prompt = self._messages_to_prompt(messages, model_info)
        
        # Create a response ID
        response_id = f"llamaforge-{int(time.time())}-{model_id}"
        
        # Placeholder implementation - in a real server this would stream tokens from the model
        # Here we'll simulate streaming by yielding one character at a time
        completion = self._generate_chat_completion(
            prompt=prompt,
            model_info=model_info,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Send the start of the stream
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_id, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
        
        # Stream each token (simulated as characters here)
        for i in range(len(completion)):
            chunk = completion[i]
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_id, 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
            await asyncio.sleep(0.01)  # Simulate token generation delay
        
        # End of stream
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield f"data: [DONE]\n\n"
    
    async def _stream_completion(self, request: CompletionRequest, model_info: Dict):
        """Stream a text completion response."""
        model_id = request.model
        prompt = request.prompt
        if isinstance(prompt, list):
            prompt = prompt[0]  # Just use the first prompt for now
        
        # Create a response ID
        response_id = f"llamaforge-{int(time.time())}-{model_id}"
        
        # Placeholder implementation - in a real server this would stream tokens from the model
        completion = self._generate_completion(
            prompt=prompt,
            model_info=model_info,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Stream each token (simulated as characters here)
        for i in range(len(completion)):
            chunk = completion[i]
            yield f"data: {json.dumps({'id': response_id, 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': model_id, 'choices': [{'index': 0, 'text': chunk, 'finish_reason': None}]})}\n\n"
            await asyncio.sleep(0.01)  # Simulate token generation delay
        
        # End of stream
        yield f"data: {json.dumps({'id': response_id, 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': model_id, 'choices': [{'index': 0, 'text': '', 'finish_reason': 'stop'}]})}\n\n"
        yield f"data: [DONE]\n\n"
    
    def _messages_to_prompt(self, messages: List[ChatMessage], model_info: Dict) -> str:
        """Convert a list of chat messages to a prompt for the model."""
        # Get the appropriate chat template based on model name
        model_name = model_info.get("name", "").lower()
        templates = self.config.get("chat_templates", {})
        
        # Select appropriate template
        template = None
        for key, value in templates.items():
            if key in model_name:
                template = value
                break
        
        # Use default template if no match found
        if not template:
            template = templates.get("default", "{prompt}")
        
        # For simplicity, we'll just concatenate messages with roles
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
        
        # Prepare for assistant's response
        prompt += "Assistant: "
        
        return prompt
    
    def _generate_chat_completion(self, prompt: str, model_info: Dict, max_tokens: int, temperature: float, top_p: float) -> str:
        """
        Generate a chat completion using the appropriate backend.
        This is a placeholder implementation that would be replaced with actual model calls.
        """
        # In a real implementation, this would call the appropriate backend
        # Here, we'll just return a placeholder response
        return "This is a placeholder response from the LlamaForge API server. In a real implementation, this would be generated by the selected model."
    
    def _generate_completion(self, prompt: str, model_info: Dict, max_tokens: int, temperature: float, top_p: float) -> str:
        """
        Generate a text completion using the appropriate backend.
        This is a placeholder implementation that would be replaced with actual model calls.
        """
        # In a real implementation, this would call the appropriate backend
        # Here, we'll just return a placeholder response
        return "This is a placeholder response from the LlamaForge API server. In a real implementation, this would be generated by the selected model."
    
    def start(self, host: str = "127.0.0.1", port: int = 8000):
        """Start the API server."""
        import uvicorn
        logger.info(f"Starting API server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)
    
    def shutdown(self):
        """Clean up resources when shutting down."""
        logger.info("Shutting down API server")
        # Clean up any resources here if needed 