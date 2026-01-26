#!/usr/bin/env python3
"""
Launch the llama‑server demo in true head‑less mode.
"""

import os
import subprocess
import sys
import time
import socket
import json
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def run(cmd, *, shell=False, cwd=None, env=None, capture=False):
    """Run a command and return its stdout (if capture=True)."""
    if env is None:
        env = os.environ.copy()
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            cwd=cwd,
            env=env,
            check=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout.strip() if capture else None
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Command failed: {exc.cmd}\n{exc.output or exc.stderr}", file=sys.stderr)
        sys.exit(exc.returncode)

def is_port_in_use(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def get_ngrok_url(max_attempts=30, delay=0.5):
    """Get ngrok URL with retries."""
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:4040/api/tunnels",
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
                for t in data["tunnels"]:
                    if t["proto"] == "http" and t["addr"] == "8002":
                        return t["public_url"]
        except urllib.error.URLError as e:
            if attempt == 0:  # Only print on first attempt
                print(f"Waiting for ngrok API to start...")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)[:100]}")
        time.sleep(delay)
    return None

# --------------------------------------------------------------------------- #
# Main script
# --------------------------------------------------------------------------- #

# 1. Check prerequisites
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    sys.exit("[ERROR] GITHUB_TOKEN not set")

NGROK_TOKEN = os.getenv("NGROK_TOKEN")
if not NGROK_TOKEN:
    sys.exit("[ERROR] NGROK_TOKEN not set")

# Check for port conflicts
if is_port_in_use(4040):
    print("[WARNING] Port 4040 (ngrok API) is already in use")
if is_port_in_use(8000):
    print("[WARNING] Port 8000 (llama-server) is already in use")
if is_port_in_use(8002):
    sys.exit("[ERROR] Port 8002 (Streamlit) is already in use")

# 2. Download the pre‑built llama‑server binary
REPO = "ghghang2/llamacpp_t4_v1"
FILES = ["llama-server"]

env = os.environ.copy()
env["GITHUB_TOKEN"] = GITHUB_TOKEN

for f in FILES:
    run(f"gh release download --repo {REPO} --pattern {f}", shell=True, env=env)
    run(f"chmod +x ./{f}", shell=True)

# 3. Start llama‑server
log_path = Path("llama_server.log").open("w", encoding="utf-8")
llama_proc = subprocess.Popen(
    ["./llama-server", "-hf", "unsloth/gpt-oss-20b-GGUF:F16", "--port", "8000"],
    stdout=log_path,
    stderr=log_path,
    start_new_session=True,
)
print("✅ llama-server started")

# 4. Install required Python packages
run("pip install --upgrade pip", shell=True)
run("pip install streamlit pyngrok pygithub", shell=True)

# 5. Start the Streamlit UI
streamlit_proc = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8002", "--server.headless", "true"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
print("✅ Streamlit started on port 8002")

# Give Streamlit time to start
time.sleep(3)

# 6. Start ngrok
ngrok_cmd = [
    "ngrok", "http",
    "--authtoken", NGROK_TOKEN,
    "--log", "ngrok.log",
    "8002"
]
ngrok_proc = subprocess.Popen(
    ngrok_cmd,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
print("✅ ngrok started")

# Wait a moment for ngrok to initialize
time.sleep(2)

# Check if ngrok is still running
if ngrok_proc.poll() is not None:
    sys.exit("[ERROR] ngrok process failed to start or exited immediately")

# 7. Get the public URL
tunnel_url = get_ngrok_url(max_attempts=40, delay=0.5)

if not tunnel_url:
    # Try alternative approach using pyngrok
    try:
        from pyngrok import ngrok
        tunnels = ngrok.get_tunnels()
        for tunnel in tunnels:
            if "8002" in tunnel.config["addr"]:
                tunnel_url = tunnel.public_url
                break
    except Exception as e:
        print(f"[WARNING] Failed to use pyngrok: {e}")

if not tunnel_url:
    sys.exit("[ERROR] Could not obtain ngrok URL after multiple attempts")

print(f"✅ Streamlit UI is publicly available at: {tunnel_url}")
print("Services are running in the background.")
print("Check llama_server.log and ngrok.log for details.")