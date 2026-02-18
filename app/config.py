# app/config.py
"""
Application‑wide constants.
"""
from app.tools.repo_overview import func
# --------------------------------------------------------------------------- #
#  General settings
# --------------------------------------------------------------------------- #
# Base URL of the local llama-server.  Historically this was called NGROK_URL
SERVER_URL = "http://localhost:8000"
MODEL_NAME = "unsloth/gpt-oss-20b-GGUF:F16"
# SERVER_URL = "https://api.deepseek.com"
# MODEL_NAME = "deepseek-reasoner"
DEFAULT_SYSTEM_PROMPT = f'''
You are a helpful assistant. Whenever the user’s request can be fulfilled with a tool, you must call that tool first and only give the user the tool’s output. Do not respond directly unless no tool applies.
Respond concisely and accurately.
Use the provided tools to satisfy user requests.
important! apply_patch in small patches. important!
When running grep, omit the -n flag to avoid timeouts.
When using sed -n, read at least 500 lines of the target file.
When using apply_patch, always reread the code from file to ensure the current content input to apply_patch will match with the current content in file.
When using apply_patch, prefer to make small patches to large patches. Break large patches into multiple smaller patches.
'''

# --------------------------------------------------------------------------- #
#  GitHub repository details
# --------------------------------------------------------------------------- #
USER_NAME = "ghghang2"
REPO_NAME = "v1.1"

# --------------------------------------------------------------------------- #
#  Items to ignore in the repo
# --------------------------------------------------------------------------- #
IGNORED_ITEMS = [
    ".*",
    "sample_data",
    "llama-server",
    "__pycache__",
    "*.log",
    "*.yml",
    "*.json",
    "*.out",
]