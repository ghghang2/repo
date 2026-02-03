import sys, os
import json
from pathlib import Path

# Ensure the repository root is on sys.path for imports
sys.path.append(os.path.abspath("."))

from app.tools.create_file import func as create_file


def test_create_file_creates_file_with_content(tmp_path):
    """Test that the create_file tool creates a file with the specified content."""
    # The tool expects a path relative to the repo root.  Use a subdirectory
    # inside the repository root.
    relative_path = "tests/tmp_test_file.txt"
    content = "Hello, world!"

    # Call the tool
    result_json = create_file(relative_path, content)
    result = json.loads(result_json)

    # Verify the result indicates success
    assert "result" in result, f"Tool returned error: {result.get('error')}"
    assert result["result"] == f"File created: {relative_path}"

    # Verify the file was created with the correct content
    repo_root = Path(__file__).resolve().parents[1]
    created_file = repo_root / relative_path
    assert created_file.exists(), "File was not created"
    assert created_file.read_text(encoding="utf-8") == content