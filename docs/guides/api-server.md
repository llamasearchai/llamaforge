# API Server Guide

LlamaForge includes a built-in API server that provides an OpenAI-compatible API for your models. This guide explains how to set up and use the API server.

## Starting the API Server

### From the Command Line

The simplest way to start the API server is using the command line:

```bash
llamaforge serve --model llama-2-7b-chat --host 127.0.0.1 --port 8000
```

### From Python

You can also start the server programmatically:

```python
from llamaforge import LlamaForge

forge = LlamaForge()
forge.load_model("llama-2-7b-chat")
forge.start_server(host="127.0.0.1", port=8000)
```

## API Endpoints

The server provides several API endpoints compatible with OpenAI's API:

### Chat Completions

```
POST /v1/chat/completions
```

Example request:

```json
{
  "model": "llama-2-7b-chat",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms"}
  ],
  "temperature": 0.7,
  "max_tokens": 500
}
```

### Completions

```
POST /v1/completions
```

Example request:

```json
{
  "model": "llama-2-7b-chat",
  "prompt": "Explain quantum computing in simple terms",
  "temperature": 0.7,
  "max_tokens": 500
}
```

### Models

```
GET /v1/models
```

This endpoint returns a list of available models.

## Using the API

### With cURL

```bash
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
```

### With Python and OpenAI's Client

You can use OpenAI's Python client by setting the base URL:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # API key is required but not used
)

response = client.chat.completions.create(
    model="llama-2-7b-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Streaming Responses

The API supports streaming responses:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

stream = client.chat.completions.create(
    model="llama-2-7b-chat",
    messages=[
        {"role": "user", "content": "Write a poem about AI"}
    ],
    temperature=0.7,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

## Configuring the API Server

### CORS Settings

To allow cross-origin requests (necessary for web applications):

```bash
llamaforge serve --model llama-2-7b-chat --enable-cors
```

Or in Python:

```python
forge.start_server(host="127.0.0.1", port=8000, enable_cors=True)
```

### Authentication

To add basic API key authentication:

```bash
llamaforge serve --model llama-2-7b-chat --api-key your-api-key
```

Or in Python:

```python
forge.start_server(host="127.0.0.1", port=8000, api_key="your-api-key")
```

### Multiple Models

You can make multiple models available through the API:

```bash
llamaforge serve --models llama-2-7b-chat,gpt-4
```

Or in Python:

```python
forge.load_model("llama-2-7b-chat")
forge.load_model("gpt-4")
forge.start_server(host="127.0.0.1", port=8000)
```

## Using Plugins with the API Server

Plugins can enhance the capabilities of your API server:

```python
from llamaforge import LlamaForge
from llamaforge.plugins.preprocessing import TextFormatterPlugin
from llamaforge.plugins.postprocessing import TextCleanerPlugin

forge = LlamaForge()
forge.load_model("llama-2-7b-chat")

# Register plugins
formatter = TextFormatterPlugin()
formatter.configure({"trim_whitespace": True})
forge.register_plugin(formatter)

cleaner = TextCleanerPlugin()
cleaner.configure({"remove_special_tokens": True})
forge.register_plugin(cleaner)

# Start server with plugins enabled
forge.start_server(host="127.0.0.1", port=8000)
```

## Production Deployment

For production deployments, consider:

1. **Running behind a proxy**: Use Nginx or Apache as a reverse proxy
2. **Setting up TLS/SSL**: Ensure secure connections
3. **Rate limiting**: Prevent abuse of your API
4. **Monitoring**: Set up monitoring for server health
5. **Load balancing**: Distribute requests across multiple instances

Example Nginx configuration:

```nginx
server {
    listen 443 ssl;
    server_name api.yourserver.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Next Steps

- Check the [Configuration Guide](../getting-started/configuration.md) for more configuration options
- See the [API Reference](../api/server.md) for detailed information on the API endpoints 