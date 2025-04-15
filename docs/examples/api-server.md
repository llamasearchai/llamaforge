# API Server Example

This example demonstrates how to set up and use LlamaForge's API server, which provides an OpenAI-compatible API for your models.

## Basic API Server

```python
from llamaforge import LlamaForge

def main():
    # Initialize LlamaForge
    forge = LlamaForge()
    
    # Load a model
    model_name = "llama-2-7b-chat"  # Change to your model name
    
    try:
        forge.load_model(model_name)
        print(f"Loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Start the server
    print(f"Starting API server with model: {model_name}")
    forge.start_server(
        host="127.0.0.1",
        port=8000,
        enable_cors=True
    )
    # The server runs in a new thread, so this will block until interrupted
    print("Server running at http://127.0.0.1:8000")
    try:
        # Keep the main thread alive
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down server...")

if __name__ == "__main__":
    main()
```

Save this code to a file (e.g., `api_server.py`) and run it:

```bash
python api_server.py
```

## Server with Multiple Models

```python
from llamaforge import LlamaForge

def main():
    # Initialize LlamaForge
    forge = LlamaForge()
    
    # Load multiple models
    models = [
        "llama-2-7b-chat",   # Change to your model names
        "gpt-4"
    ]
    
    for model_name in models:
        try:
            forge.load_model(model_name)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # Start the server with all loaded models
    print(f"Starting API server with models: {', '.join(models)}")
    forge.start_server(
        host="127.0.0.1",
        port=8000,
        enable_cors=True
    )
    
    print("Server running at http://127.0.0.1:8000")
    try:
        # Keep the main thread alive
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down server...")

if __name__ == "__main__":
    main()
```

## Server with Authentication

```python
from llamaforge import LlamaForge
import uuid

def main():
    # Initialize LlamaForge
    forge = LlamaForge()
    
    # Load a model
    model_name = "llama-2-7b-chat"  # Change to your model name
    
    try:
        forge.load_model(model_name)
        print(f"Loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate a random API key (or use a fixed one)
    api_key = str(uuid.uuid4())
    print(f"Generated API key: {api_key}")
    
    # Start the server with authentication
    print(f"Starting API server with model: {model_name}")
    forge.start_server(
        host="127.0.0.1",
        port=8000,
        enable_cors=True,
        api_key=api_key
    )
    
    print("Server running at http://127.0.0.1:8000")
    print("Remember to include the API key in your requests")
    try:
        # Keep the main thread alive
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down server...")

if __name__ == "__main__":
    main()
```

## Server with Plugins

```python
from llamaforge import LlamaForge
from llamaforge.plugins.preprocessing import TextFormatterPlugin
from llamaforge.plugins.postprocessing import TextCleanerPlugin

def main():
    # Initialize LlamaForge
    forge = LlamaForge()
    
    # Load a model
    model_name = "llama-2-7b-chat"  # Change to your model name
    
    try:
        forge.load_model(model_name)
        print(f"Loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create and configure preprocessing plugin
    formatter = TextFormatterPlugin()
    formatter.configure({
        "trim_whitespace": True,
        "add_system_instruction": True,
        "system_instruction": "You are a helpful assistant that gives concise answers.",
        "format_as_chat": True
    })
    forge.register_plugin(formatter)
    print("Registered preprocessing plugin: TextFormatterPlugin")
    
    # Create and configure postprocessing plugin
    cleaner = TextCleanerPlugin()
    cleaner.configure({
        "trim_whitespace": True,
        "remove_special_tokens": True
    })
    forge.register_plugin(cleaner)
    print("Registered postprocessing plugin: TextCleanerPlugin")
    
    # Start the server
    print(f"Starting API server with model: {model_name}")
    forge.start_server(
        host="127.0.0.1",
        port=8000,
        enable_cors=True
    )
    
    print("Server running at http://127.0.0.1:8000")
    try:
        # Keep the main thread alive
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down server...")

if __name__ == "__main__":
    main()
```

## Using the API

### With cURL

```bash
# Chat completions endpoint
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    "temperature": 0.7
  }'

# Completions endpoint
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat",
    "prompt": "Explain quantum computing in simple terms",
    "temperature": 0.7,
    "max_tokens": 500
  }'

# Models list endpoint
curl http://localhost:8000/v1/models
```

### With Python

```python
from openai import OpenAI

# Create a client that points to your local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"  # Use the API key you configured, or "dummy" if no auth
)

# Chat completion
response = client.chat.completions.create(
    model="llama-2-7b-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)

# Streaming response
stream = client.chat.completions.create(
    model="llama-2-7b-chat",
    messages=[
        {"role": "user", "content": "Write a short poem about AI"}
    ],
    temperature=0.7,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

## Production Deployment

For production deployment, consider using a WSGI server and a reverse proxy:

```python
# server.py
from llamaforge import LlamaForge
import os

# Initialize and set up LlamaForge
forge = LlamaForge()
model_name = os.environ.get("LLAMAFORGE_MODEL", "llama-2-7b-chat")
forge.load_model(model_name)

# Get the Flask app
# Note: The actual method may differ depending on LlamaForge's implementation
app = forge.create_server_app(
    enable_cors=True,
    api_key=os.environ.get("LLAMAFORGE_API_KEY")
)

# For WSGI servers (Gunicorn, uWSGI, etc.)
if __name__ == "__main__":
    # This is only used for direct execution, not for WSGI
    app.run(host="0.0.0.0", port=8000)
```

Then with Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:8000 server:app
```

## Next Steps

- Explore the [API Server Guide](../guides/api-server.md) for more details on the API endpoints
- Learn about [Plugins](../guides/plugins.md) to enhance your API server
- Check out the [Configuration Guide](../getting-started/configuration.md) for server configuration options 