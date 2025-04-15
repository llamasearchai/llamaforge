"""
Benchmark command plugin for LlamaForge.
"""

import json
import statistics
import time
from typing import Any, Dict, List, Optional

from ...forge import LlamaForge
from ..base import BasePlugin


class BenchmarkPlugin(BasePlugin):
    """
    Plugin for benchmarking model performance.

    This plugin measures the generation speed and other metrics
    for loaded models.
    """

    @property
    def name(self) -> str:
        """
        Get the name of the plugin.

        Returns:
            str: Plugin name
        """
        return "benchmark"

    @property
    def description(self) -> str:
        """
        Get the description of the plugin.

        Returns:
            str: Plugin description
        """
        return "Benchmarks model performance on text generation tasks"

    def process(self, data: Any) -> Dict[str, Any]:
        """
        Process the benchmark request.

        Args:
            data: Benchmark parameters

        Returns:
            Dict[str, Any]: Benchmark results
        """
        try:
            # Get benchmark parameters
            prompts = self.get_config(
                "prompts", data.get("prompts", ["Hello, how are you?"])
            )
            iterations = self.get_config("iterations", data.get("iterations", 3))
            max_tokens = self.get_config("max_tokens", data.get("max_tokens", 100))
            model_name = self.get_config("model", data.get("model"))
            warmup = self.get_config("warmup", data.get("warmup", True))

            # Get LlamaForge instance
            forge = data.get("forge")
            if not forge or not isinstance(forge, LlamaForge):
                return {"error": "LlamaForge instance required"}

            # Load model if specified
            if model_name and (
                forge.current_model is None or forge.current_model.name != model_name
            ):
                if not forge.load_model(model_name):
                    return {"error": f"Failed to load model: {model_name}"}

            # Ensure a model is loaded
            if forge.current_model is None:
                return {"error": "No model loaded"}

            # Warmup run if requested
            if warmup:
                _ = forge.generate(prompts[0], max_tokens=10)

            # Run benchmark
            results = self._run_benchmark(forge, prompts, iterations, max_tokens)

            return results

        except Exception as e:
            return {"error": str(e)}

    def _run_benchmark(
        self, forge: LlamaForge, prompts: List[str], iterations: int, max_tokens: int
    ) -> Dict[str, Any]:
        """
        Run the benchmark.

        Args:
            forge: LlamaForge instance
            prompts: List of prompts to benchmark
            iterations: Number of iterations per prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Dict[str, Any]: Benchmark results
        """
        results = {
            "model": forge.current_model.name,
            "backend": forge.backend.name,
            "iterations": iterations,
            "max_tokens": max_tokens,
            "prompts": [],
        }

        total_tokens = 0
        total_time = 0

        for prompt in prompts:
            prompt_results = {
                "prompt": prompt,
                "runs": [],
                "metrics": {},
            }

            # Run iterations
            times = []
            tokens = []
            characters = []

            for i in range(iterations):
                start_time = time.time()
                response = forge.generate(prompt, max_tokens=max_tokens)
                end_time = time.time()

                run_time = end_time - start_time
                token_count = len(response.split())
                char_count = len(response)

                prompt_results["runs"].append(
                    {
                        "time": run_time,
                        "tokens": token_count,
                        "characters": char_count,
                        "tokens_per_second": (
                            token_count / run_time if run_time > 0 else 0
                        ),
                    }
                )

                times.append(run_time)
                tokens.append(token_count)
                characters.append(char_count)

                total_tokens += token_count
                total_time += run_time

            # Calculate metrics for this prompt
            prompt_results["metrics"] = {
                "avg_time": statistics.mean(times),
                "min_time": min(times),
                "max_time": max(times),
                "std_dev_time": statistics.stdev(times) if len(times) > 1 else 0,
                "avg_tokens": statistics.mean(tokens),
                "avg_tokens_per_second": (
                    statistics.mean(tokens) / statistics.mean(times)
                    if statistics.mean(times) > 0
                    else 0
                ),
            }

            results["prompts"].append(prompt_results)

        # Calculate overall metrics
        results["overall"] = {
            "total_tokens": total_tokens,
            "total_time": total_time,
            "avg_tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
        }

        return results
