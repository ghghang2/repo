# Configuration – tweak these values as needed
NGROK_URL = "http://localhost:8000"
MODEL_NAME = "unsloth/gpt-oss-20b-GGUF:F16"
DEFAULT_SYSTEM_PROMPT = "Be concise and accurate at all times"

USER_NAME = "ghghang2"          # GitHub user / org name
REPO_NAME = "v1.1"              # Repository to push to
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