#!/usr/bin/env python
"""Loom entry point wrapper."""
import sys
import os

# Add current dir to path so relative imports work
sys.path.insert(0, os.path.dirname(__file__))

from cli import main

if __name__ == "__main__":
    main()
