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
        print(
            f"[ERROR] Command failed: {exc.cmd}\n{exc.output or exc.stderr}",
            file=sys.stderr,
        )
        sys.exit(exc.returncode)

def is_port_in_use(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def main():
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
    llama_log = Path("llama_server.log").open("w", encoding="utf-8")
    subprocess.Popen(
        [
            "./llama-server",
            "-hf",
            "unsloth/gpt-oss-20b-GGUF:F16",
            "--port",
            "8000",
        ],
        stdout=llama_log,
        stderr=llama_log,
        start_new_session=True,
    )
    llama_log.close()
    print("✅ llama-server started")

    # 4. Install required Python packages
    run("pip install streamlit pyngrok pygithub", shell=True)

    # 5. Start the Streamlit UI
    streamlit_proc = subprocess.Popen(
        [
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            "8002",
            "--server.headless",
            "true",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print("✅ Streamlit started on port 8002")

    # Give Streamlit time to start
    time.sleep(3)

    # 6. Create ngrok tunnel via pyngrok
    try:
        from pyngrok import ngrok
    except ImportError:
        sys.exit("[ERROR] pyngrok is not installed")

    # Configure ngrok
    ngrok.set_auth_token(NGROK_TOKEN)
    # Create the tunnel (http -> 8002)
    tunnel = ngrok.connect(8002, "http")
    tunnel_url = tunnel.public_url
    print(f"✅ Streamlit UI is publicly available at: {tunnel_url}")
    print("Services are running in the background.")
    print("Check llama_server.log for details.")

    # Exit immediately – all services are detached.
    sys.exit(0)

if __name__ == "__main__":
    main()