# Basic Chat Example

This example demonstrates how to create a simple interactive chat application using LlamaForge.

## Simple Chat Loop

```python
from llamaforge import LlamaForge

def main():
    # Initialize LlamaForge
    forge = LlamaForge()
    
    # Display available models
    models = forge.list_models()
    print("Available models:")
    for model_name in models:
        print(f"- {model_name}")
    
    # Prompt user to select a model
    model_name = input("\nEnter model name (or press Enter for default): ")
    if not model_name and forge.config.get("default_model"):
        model_name = forge.config.get("default_model")
        print(f"Using default model: {model_name}")
    elif not model_name:
        print("No model specified and no default model set. Exiting.")
        return
    
    # Load the model
    try:
        forge.load_model(model_name)
        print(f"Loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set up chat history
    chat_history = []
    
    # Main chat loop
    print("\nChat with the model (type 'exit' to quit)")
    print("---------------------------------------------")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        
        # Generate response
        try:
            response = forge.chat(user_input, chat_history)
            print(f"\nAI: {response}")
            
            # Update chat history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
```

Save this code to a file (e.g., `basic_chat.py`) and run it:

```bash
python basic_chat.py
```

## Chat with Streaming

This example shows how to create a chat application with streaming responses:

```python
from llamaforge import LlamaForge
import time
import sys

def main():
    # Initialize LlamaForge
    forge = LlamaForge()
    
    # Load a model
    model_name = input("Enter model name (or press Enter for default): ")
    if not model_name and forge.config.get("default_model"):
        model_name = forge.config.get("default_model")
        print(f"Using default model: {model_name}")
    elif not model_name:
        print("No model specified and no default model set. Exiting.")
        return
    
    # Load the model
    try:
        forge.load_model(model_name)
        print(f"Loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set up chat history
    chat_history = []
    
    # Main chat loop
    print("\nChat with the model (type 'exit' to quit)")
    print("---------------------------------------------")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        
        # Start streaming response
        try:
            print("\nAI: ", end="", flush=True)
            response_text = ""
            
            for chunk in forge.chat_stream(user_input, chat_history):
                print(chunk, end="", flush=True)
                response_text += chunk
                time.sleep(0.01)  # Small delay for a more natural reading experience
            print()
            
            # Update chat history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response_text})
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
```

## Chat with System Instructions

This example shows how to use system instructions to guide the model's behavior:

```python
from llamaforge import LlamaForge
from llamaforge.plugins.preprocessing import TextFormatterPlugin

def main():
    # Initialize LlamaForge
    forge = LlamaForge()
    
    # Load a model
    model_name = input("Enter model name (or press Enter for default): ")
    if not model_name and forge.config.get("default_model"):
        model_name = forge.config.get("default_model")
        print(f"Using default model: {model_name}")
    elif not model_name:
        print("No model specified and no default model set. Exiting.")
        return
    
    # Load the model
    try:
        forge.load_model(model_name)
        print(f"Loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create and configure the text formatter plugin with a system instruction
    formatter = TextFormatterPlugin()
    formatter.configure({
        "trim_whitespace": True,
        "add_system_instruction": True,
        "system_instruction": "You are a helpful assistant that gives concise answers.",
        "format_as_chat": True
    })
    
    # Register the plugin
    forge.register_plugin(formatter)
    
    # Set up chat history
    chat_history = []
    
    # Main chat loop
    print("\nChat with the model (type 'exit' to quit)")
    print("---------------------------------------------")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        
        # Generate response
        try:
            response = forge.chat(user_input, chat_history)
            print(f"\nAI: {response}")
            
            # Update chat history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
```

## Interface Options

Here are some enhancements you can add to your chat application:

1. **Add a welcome message**: Display model information and capabilities at startup
2. **Support special commands**: Add support for commands like `/clear` to clear chat history
3. **Save conversations**: Add options to save and load conversations
4. **Add rich formatting**: Use a library like `rich` for better terminal presentation

## Next Steps

- Check out the [API Server Example](api-server.md) to learn how to expose your chat functionality as an API
- Explore the [Plugins Guide](../guides/plugins.md) to enhance your chat application with additional capabilities 