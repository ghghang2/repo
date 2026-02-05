import sys
import os
import json
from pathlib import Path

# Ensure the repository root is on sys.path for imports
sys.path.append(os.path.abspath("."))

from app.tools.apply_patch import apply_patch

def test_apply_patch_create_file():
    # Create a simple diff that creates a file with two lines
    diff = "+Hello\n+World\n"
    file_path = "tmp_test_file_create.txt"
    result_json = apply_patch(file_path, "create", diff)
    result = json.loads(result_json)
    assert "result" in result
    assert result["result"].startswith("File created")
    created_path = Path(file_path)
    assert created_path.exists()
    content = created_path.read_text(encoding="utf-8")
    assert content == "Hello\nWorld"
    created_path.unlink(missing_ok=True)
