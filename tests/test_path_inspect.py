"""Utility test to introspect the import path during pytest collection."""

import sys

def test_path():  # pragma: no cover
    print("sys.path", sys.path)