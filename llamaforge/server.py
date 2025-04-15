"""
API server for LlamaForge.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from .forge import LlamaForge

logger = logging.getLogger(__name__)

try:
    import uvicorn
    from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
except ImportError:
    logger.error(
        "Server dependencies not installed. "
        'Install with: pip install "llamaforge[server]"'
    )
    raise


# API Models for request/response validation
class ModelRequest(BaseModel):
    model: str = Field(..., description="Model identifier to use")


class CompletionRequest(BaseModel):
    model: str = Field(..., description="Model identifier to use")
    prompt: str = Field(..., description="Prompt to generate completion for")
    max_tokens: Optional[int] = Field(256, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role (system, user, or assistant)")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model identifier to use")
    messages: List[ChatMessage] = Field(
        ..., description="Messages to generate completion for"
    )
    max_tokens: Optional[int] = Field(256, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")


def create_app(forge: LlamaForge) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        forge: LlamaForge instance

    Returns:
        FastAPI: API application
    """
    app = FastAPI(
        title="LlamaForge API",
        description="API for LlamaForge Language Model Interface",
        version="0.2.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store the forge instance in app state
    app.state.forge = forge

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        """
        List available models.
        """
        models = forge.list_models()
        model_data = []

        for model_name in models:
            model_info = forge.get_model_info(model_name)
            model_data.append(
                {
                    "id": model_name,
                    "object": "model",
                    "created": model_info.get("created", 0),
                    "owned_by": model_info.get("owner", "llamaforge"),
                    "permission": [],
                    "root": model_name,
                    "parent": None,
                }
            )

        return {"object": "list", "data": model_data}

    @app.post("/v1/models/{model_id}")
    async def load_model(model_id: str) -> Dict[str, Any]:
        """
        Load a specific model.
        """
        if not forge.load_model(model_id):
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

        model_info = forge.get_model_info(model_id)
        return {
            "id": model_id,
            "object": "model",
            "created": model_info.get("created", 0),
            "owned_by": model_info.get("owner", "llamaforge"),
            "status": "loaded",
        }

    @app.post("/v1/completions")
    async def create_completion(
        request: CompletionRequest, background_tasks: BackgroundTasks
    ) -> Union[Dict[str, Any], StreamingResponse]:
        """
        Generate a completion.
        """
        # Load the requested model if it's not the current model
        if forge.current_model is None or forge.current_model.name != request.model:
            if not forge.load_model(request.model):
                raise HTTPException(
                    status_code=404, detail=f"Model '{request.model}' not found"
                )

        # Set up generation parameters
        params = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        if request.stop:
            params["stop"] = request.stop

        try:
            if request.stream:
                # For streaming response
                async def generate_stream():
                    # Start header
                    yield "data: " + json.dumps(
                        {
                            "id": "cmpl-stream",
                            "object": "completion.chunk",
                            "created": 0,
                            "model": request.model,
                            "choices": [
                                {"text": "", "index": 0, "finish_reason": None}
                            ],
                        }
                    ) + "\n\n"

                    # Stream the content
                    for chunk in forge.generate_stream(request.prompt, **params):
                        yield "data: " + json.dumps(
                            {
                                "id": "cmpl-stream",
                                "object": "completion.chunk",
                                "created": 0,
                                "model": request.model,
                                "choices": [
                                    {"text": chunk, "index": 0, "finish_reason": None}
                                ],
                            }
                        ) + "\n\n"

                    # End marker
                    yield "data: " + json.dumps(
                        {
                            "id": "cmpl-stream",
                            "object": "completion.chunk",
                            "created": 0,
                            "model": request.model,
                            "choices": [
                                {"text": "", "index": 0, "finish_reason": "stop"}
                            ],
                        }
                    ) + "\n\n"

                    # Signal completion
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    generate_stream(), media_type="text/event-stream"
                )
            else:
                # For non-streaming response
                response = forge.generate(request.prompt, **params)

                return {
                    "id": "cmpl-" + request.model,
                    "object": "text_completion",
                    "created": 0,
                    "model": request.model,
                    "choices": [
                        {"text": response, "index": 0, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": len(request.prompt.split()),
                        "completion_tokens": len(response.split()),
                        "total_tokens": len(request.prompt.split())
                        + len(response.split()),
                    },
                }
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        request: ChatCompletionRequest,
    ) -> Union[Dict[str, Any], StreamingResponse]:
        """
        Generate a chat completion.
        """
        # Load the requested model if it's not the current model
        if forge.current_model is None or forge.current_model.name != request.model:
            if not forge.load_model(request.model):
                raise HTTPException(
                    status_code=404, detail=f"Model '{request.model}' not found"
                )

        # Format messages into a prompt
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"

        prompt += "Assistant: "

        # Set up generation parameters
        params = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        if request.stop:
            params["stop"] = request.stop

        try:
            if request.stream:
                # For streaming response
                async def generate_stream():
                    # Start header
                    yield "data: " + json.dumps(
                        {
                            "id": "chatcmpl-stream",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": request.model,
                            "choices": [
                                {
                                    "delta": {"role": "assistant", "content": ""},
                                    "index": 0,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    ) + "\n\n"

                    # Stream the content
                    for chunk in forge.generate_stream(prompt, **params):
                        yield "data: " + json.dumps(
                            {
                                "id": "chatcmpl-stream",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": request.model,
                                "choices": [
                                    {
                                        "delta": {"content": chunk},
                                        "index": 0,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        ) + "\n\n"

                    # End marker
                    yield "data: " + json.dumps(
                        {
                            "id": "chatcmpl-stream",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": request.model,
                            "choices": [
                                {"delta": {}, "index": 0, "finish_reason": "stop"}
                            ],
                        }
                    ) + "\n\n"

                    # Signal completion
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    generate_stream(), media_type="text/event-stream"
                )
            else:
                # For non-streaming response
                response = forge.generate(prompt, **params)
                # Extract just the assistant's response
                assistant_response = response.split("Assistant: ")[-1].strip()

                return {
                    "id": "chatcmpl-" + request.model,
                    "object": "chat.completion",
                    "created": 0,
                    "model": request.model,
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": assistant_response,
                            },
                            "index": 0,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(assistant_response.split()),
                        "total_tokens": len(prompt.split())
                        + len(assistant_response.split()),
                    },
                }
        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def run_server(forge: LlamaForge, host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    Run the API server.

    Args:
        forge: LlamaForge instance
        host: Host to bind server to
        port: Port to bind server to
    """
    app = create_app(forge)

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
