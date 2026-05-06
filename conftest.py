"""
Root conftest.py — adds the project root to sys.path so that all test modules
can import project packages (core, strategies, factories, etc.) without any
special install step.

Also globally patches `dotenv.load_dotenv` so that tests never attempt to read
a real .env file from disk.
"""
import sys
import os
from unittest.mock import patch

# Make the project root importable
sys.path.insert(0, os.path.dirname(__file__))

# Suppress dotenv so missing .env files don't cause import errors
patch("dotenv.load_dotenv", lambda *a, **kw: None).start()

