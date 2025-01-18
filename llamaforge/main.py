"""
Command-line interface for LlamaForge.
"""

import sys
import os
import argparse
import logging
from typing import List, Optional, Dict, Any
import json

from .forge import LlamaForge
from .config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="LlamaForge - Ultimate Language Model Command Interface"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument("--model", help="Model to use for chat")
    chat_parser.add_argument("--backend", help="Backend to use for model")
    chat_parser.add_argument("--temperature", type=float, help="Temperature for generation")
    chat_parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Generate text from prompt")
    run_parser.add_argument("prompt", help="Input prompt")
    run_parser.add_argument("--model", help="Model to use for generation")
    run_parser.add_argument("--backend", help="Backend to use for model")
    run_parser.add_argument("--temperature", type=float, help="Temperature for generation")
    run_parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    run_parser.add_argument("--output", help="Output file path")
    
    # list command
    list_parser = subparsers.add_parser("list", help="List available resources")
    list_parser.add_argument("resource", choices=["models", "backends", "plugins"], 
                            help="Resource to list")
    
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind server")
    serve_parser.add_argument("--model", help="Model to preload")
    
    # config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Config command")
    
    # config get command
    config_get_parser = config_subparsers.add_parser("get", help="Get configuration value")
    config_get_parser.add_argument("key", help="Configuration key")
    
    # config set command
    config_set_parser = config_subparsers.add_parser("set", help="Set configuration value")
    config_set_parser.add_argument("key", help="Configuration key")
    config_set_parser.add_argument("value", help="Configuration value")
    
    # config init command
    config_init_parser = config_subparsers.add_parser("init", help="Initialize configuration")
    
    # tools command
    tools_parser = subparsers.add_parser("tools", help="Use built-in tools")
    tools_subparsers = tools_parser.add_subparsers(dest="tool_name", help="Tool to use")
    
    # Calculator tool
    calc_parser = tools_subparsers.add_parser("calculator", help="Perform mathematical calculations")
    calc_parser.add_argument("expression", help="Mathematical expression to evaluate")
    
    # benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark model performance")
    benchmark_parser.add_argument("--model", help="Model to benchmark")
    benchmark_parser.add_argument("--prompts", nargs="+", help="Prompts to use for benchmarking")
    benchmark_parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    benchmark_parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    benchmark_parser.add_argument("--output", help="Output file for benchmark results")
    benchmark_parser.add_argument("--no-warmup", action="store_true", help="Skip warmup run")
    
    # Add global arguments
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--config", help="Path to configuration file")
    
    return parser.parse_args(args)

def chat_command(args: argparse.Namespace, forge: LlamaForge) -> None:
    """
    Run the chat command.
    
    Args:
        args: Command-line arguments
        forge: LlamaForge instance
    """
    print("Starting interactive chat. Type 'exit' to quit.")
    
    # Load model if specified
    if args.model:
        if not forge.load_model(args.model):
            logger.error(f"Failed to load model: {args.model}")
            return
    elif forge.current_model is None:
        logger.error("No model loaded. Please specify a model with --model or set a default model in the configuration.")
        return
    
    # Get generation parameters
    params: Dict[str, Any] = {}
    if args.temperature is not None:
        params["temperature"] = args.temperature
    if args.max_tokens is not None:
        params["max_tokens"] = args.max_tokens
    
    # Chat loop
    history = []
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            # Add to history and create prompt
            history.append(f"User: {user_input}")
            full_prompt = "\n".join(history) + "\nAssistant: "
            
            # Generate response
            try:
                response = forge.generate(full_prompt, **params)
                # Extract only the assistant response
                assistant_response = response.split("Assistant: ")[-1].strip()
                print(f"\nAssistant: {assistant_response}")
                # Add to history
                history.append(f"Assistant: {assistant_response}")
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
    except KeyboardInterrupt:
        print("\nExiting chat...")

def run_command(args: argparse.Namespace, forge: LlamaForge) -> None:
    """
    Run the run command.
    
    Args:
        args: Command-line arguments
        forge: LlamaForge instance
    """
    # Load model if specified
    if args.model:
        if not forge.load_model(args.model):
            logger.error(f"Failed to load model: {args.model}")
            return
    elif forge.current_model is None:
        logger.error("No model loaded. Please specify a model with --model or set a default model in the configuration.")
        return
    
    # Get generation parameters
    params: Dict[str, Any] = {}
    if args.temperature is not None:
        params["temperature"] = args.temperature
    if args.max_tokens is not None:
        params["max_tokens"] = args.max_tokens
    
    # Generate response
    try:
        response = forge.generate(args.prompt, **params)
        
        # Output to file or stdout
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(response)
            logger.info(f"Output written to {args.output}")
        else:
            print(response)
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")

def list_command(args: argparse.Namespace, forge: LlamaForge) -> None:
    """
    Run the list command.
    
    Args:
        args: Command-line arguments
        forge: LlamaForge instance
    """
    if args.resource == "models":
        models = forge.list_models()
        if models:
            print("Available models:")
            for model in models:
                print(f"  - {model}")
        else:
            print("No models available.")
    elif args.resource == "backends":
        from .backends import BACKENDS
        print("Available backends:")
        for backend in BACKENDS.keys():
            print(f"  - {backend}")
    elif args.resource == "plugins":
        try:
            # Import plugin registries
            from .plugins.preprocessing import PREPROCESSORS
            from .plugins.postprocessing import POSTPROCESSORS
            from .plugins.tools import TOOLS
            from .plugins.commands import COMMANDS
            
            print("Available plugins:")
            
            print("\nPreprocessing plugins:")
            for name, plugin_class in PREPROCESSORS.items():
                print(f"  - {name}: {plugin_class().description}")
            
            print("\nPostprocessing plugins:")
            for name, plugin_class in POSTPROCESSORS.items():
                print(f"  - {name}: {plugin_class().description}")
            
            print("\nTools plugins:")
            for name, plugin_class in TOOLS.items():
                print(f"  - {name}: {plugin_class().description}")
            
            print("\nCommand plugins:")
            for name, plugin_class in COMMANDS.items():
                print(f"  - {name}: {plugin_class().description}")
                
        except ImportError as e:
            logger.error(f"Error loading plugin registry: {str(e)}")
            print("Error loading plugin registry.")

def serve_command(args: argparse.Namespace, forge: LlamaForge) -> None:
    """
    Run the serve command.
    
    Args:
        args: Command-line arguments
        forge: LlamaForge instance
    """
    try:
        # Dynamically import server module to avoid dependencies if not needed
        from .server import run_server
        
        # Preload model if specified
        if args.model:
            if not forge.load_model(args.model):
                logger.error(f"Failed to load model: {args.model}")
                return
        
        # Start server
        run_server(forge, host=args.host, port=args.port)
    except ImportError:
        logger.error(
            "Server dependencies not installed. "
            "Install with: pip install \"llamaforge[server]\""
        )

def config_command(args: argparse.Namespace, forge: LlamaForge) -> None:
    """
    Run the config command.
    
    Args:
        args: Command-line arguments
        forge: LlamaForge instance
    """
    if args.config_command == "get":
        value = forge.config.get(args.key)
        if isinstance(value, dict):
            print(json.dumps(value, indent=2))
        else:
            print(value)
    elif args.config_command == "set":
        # Try to parse as JSON
        try:
            value = json.loads(args.value)
        except json.JSONDecodeError:
            value = args.value
        
        forge.config.set(args.key, value)
        forge.config.save()
        logger.info(f"Configuration key '{args.key}' set to '{value}'")
    elif args.config_command == "init":
        forge.config._create_default_config()
        forge.config.save()
        logger.info(f"Default configuration initialized at {forge.config.config_path}")

def tools_command(args: argparse.Namespace, forge: LlamaForge) -> None:
    """
    Run the tools command.
    
    Args:
        args: Command-line arguments
        forge: LlamaForge instance
    """
    if not args.tool_name:
        logger.error("No tool specified")
        return
    
    try:
        # Import tools registry
        from .plugins.tools import TOOLS
        
        if args.tool_name == "calculator":
            if "calculator" not in TOOLS:
                logger.error("Calculator tool not available")
                return
            
            # Create calculator plugin
            calculator = TOOLS["calculator"]()
            
            # Process expression
            result = calculator.process(args.expression)
            print(f"Result: {result}")
        else:
            logger.error(f"Unknown tool: {args.tool_name}")
    except ImportError:
        logger.error("Error loading tools registry")
    except Exception as e:
        logger.error(f"Error using tool: {str(e)}")

def benchmark_command(args: argparse.Namespace, forge: LlamaForge) -> None:
    """
    Run the benchmark command.
    
    Args:
        args: Command-line arguments
        forge: LlamaForge instance
    """
    try:
        # Import benchmark plugin
        from .plugins.commands import COMMANDS
        
        if "benchmark" not in COMMANDS:
            logger.error("Benchmark command not available")
            return
        
        # Create benchmark plugin
        benchmark = COMMANDS["benchmark"]()
        
        # Prepare benchmark parameters
        benchmark_params = {
            "forge": forge,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "iterations": args.iterations,
            "warmup": not args.no_warmup,
        }
        
        if args.prompts:
            benchmark_params["prompts"] = args.prompts
        
        # Run benchmark
        results = benchmark.process(benchmark_params)
        
        if "error" in results:
            logger.error(f"Benchmark failed: {results['error']}")
            return
        
        # Print results
        print(f"\nBenchmark Results for {results['model']} ({results['backend']} backend)")
        print(f"Iterations per prompt: {results['iterations']}")
        print(f"Max tokens per generation: {results['max_tokens']}")
        
        print("\nOverall Metrics:")
        print(f"Total tokens generated: {results['overall']['total_tokens']}")
        print(f"Total time: {results['overall']['total_time']:.2f} seconds")
        print(f"Average tokens per second: {results['overall']['avg_tokens_per_second']:.2f}")
        
        # Output to file if specified
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Benchmark results written to {args.output}")
            
    except ImportError:
        logger.error("Error loading benchmark plugin")
    except Exception as e:
        logger.error(f"Error during benchmark: {str(e)}")

def main() -> int:
    """
    Main entry point for the CLI.
    
    Returns:
        int: Exit code
    """
    args = parse_args(sys.argv[1:])
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create LlamaForge instance
    try:
        forge = LlamaForge(config_path=args.config)
    except Exception as e:
        logger.error(f"Error initializing LlamaForge: {str(e)}")
        return 1
    
    # Run command
    try:
        if args.command == "chat":
            chat_command(args, forge)
        elif args.command == "run":
            run_command(args, forge)
        elif args.command == "list":
            list_command(args, forge)
        elif args.command == "serve":
            serve_command(args, forge)
        elif args.command == "config":
            config_command(args, forge)
        elif args.command == "tools":
            tools_command(args, forge)
        elif args.command == "benchmark":
            benchmark_command(args, forge)
        else:
            # No command specified, print help
            parse_args(["--help"])
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
# Updated in commit 3 - 2025-04-04 17:03:59

# Updated in commit 11 - 2025-04-04 17:04:05

# Updated in commit 19 - 2025-04-04 17:04:10

# Updated in commit 27 - 2025-04-04 17:04:15

# Updated in commit 3 - 2025-04-05 14:24:48

# Updated in commit 11 - 2025-04-05 14:24:49
