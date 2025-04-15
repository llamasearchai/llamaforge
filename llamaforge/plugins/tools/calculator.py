"""
Calculator tool plugin for LlamaForge.
"""

import math
import re
from typing import Any, Dict, Optional, Tuple, Union

from ..base import BasePlugin


class CalculatorPlugin(BasePlugin):
    """
    Plugin for performing mathematical calculations.

    This plugin can evaluate mathematical expressions safely.
    """

    @property
    def name(self) -> str:
        """
        Get the name of the plugin.

        Returns:
            str: Plugin name
        """
        return "calculator"

    @property
    def description(self) -> str:
        """
        Get the description of the plugin.

        Returns:
            str: Plugin description
        """
        return "Performs mathematical calculations from text input"

    def process(self, expression: str) -> Union[str, float, int]:
        """
        Evaluate a mathematical expression.

        Args:
            expression: Mathematical expression as string

        Returns:
            Union[str, float, int]: Result of evaluation or error message
        """
        try:
            # Clean and validate the expression
            cleaned_expr = self._clean_expression(expression)
            if cleaned_expr is None:
                return "Invalid expression format"

            # Evaluate the expression
            result = self._safe_eval(cleaned_expr)
            return result
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    def _clean_expression(self, expression: str) -> Optional[str]:
        """
        Clean and validate a mathematical expression.

        Args:
            expression: Raw expression

        Returns:
            Optional[str]: Cleaned expression or None if invalid
        """
        # Extract the expression part if in a sentence
        match = re.search(r"calculate\s+(.+)", expression, re.IGNORECASE)
        if match:
            expression = match.group(1)

        # Remove any non-math characters
        expression = re.sub(r"[^0-9+\-*/().\s^%]", "", expression)

        # Check if expression is not empty after cleaning
        if not expression or expression.isspace():
            return None

        return expression

    def _safe_eval(self, expression: str) -> Union[float, int]:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Cleaned expression

        Returns:
            Union[float, int]: Result of evaluation

        Raises:
            ValueError: If expression contains invalid operations
        """
        # Replace ^ with ** for exponentiation
        expression = expression.replace("^", "**")

        # Define safe operations
        safe_dict = {
            "abs": abs,
            "round": round,
            "max": max,
            "min": min,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            "log": math.log,
            "log10": math.log10,
            "ceil": math.ceil,
            "floor": math.floor,
        }

        # Evaluate the expression in the safe environment
        result = eval(expression, {"__builtins__": {}}, safe_dict)

        # Convert to int if result is a whole number
        if isinstance(result, float) and result.is_integer():
            return int(result)

        return result
