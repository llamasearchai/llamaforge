#!/usr/bin/env python3
"""
Test runner for LlamaForge.
"""

import unittest
import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests():
    """Run all tests in the tests directory."""
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(str(Path(__file__).parent), pattern="test_*.py")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests()) 