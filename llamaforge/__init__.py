"""
LlamaForge - Ultimate Language Model Command Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LlamaForge is a powerful command-line tool and Python library designed to streamline
working with large language models. It provides a unified interface for managing,
running, and optimizing various language models from different providers.

Basic usage:

    >>> from llamaforge import LlamaForge
    >>> forge = LlamaForge()
    >>> response = forge.generate("Explain quantum computing in simple terms")
    >>> print(response)

:copyright: (c) 2023 by LlamaSearch AI.
:license: MIT, see LICENSE for more details.
"""

__version__ = "0.2.0"

from .forge import LlamaForge
from .model import Model

__all__ = ["LlamaForge", "Model"] 
# Updated in commit 2 - 2025-04-04 17:03:58

# Updated in commit 10 - 2025-04-04 17:04:03

# Updated in commit 18 - 2025-04-04 17:04:10

# Updated in commit 26 - 2025-04-04 17:04:15

# Updated in commit 2 - 2025-04-05 14:24:48

# Updated in commit 10 - 2025-04-05 14:24:49

# Updated in commit 18 - 2025-04-05 14:24:49

# Updated in commit 26 - 2025-04-05 14:24:49

# Updated in commit 2 - 2025-04-05 15:00:42

# Updated in commit 10 - 2025-04-05 15:00:42

# Updated in commit 18 - 2025-04-05 15:00:43

# Updated in commit 26 - 2025-04-05 15:00:43
