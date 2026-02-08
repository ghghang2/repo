# Browser tool for OpenAI function calling

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# Lazy import of Playwright; the module-level variable will be patched in tests.
sync_playwright: Any | None = None

# ---------------------------------------------------------------------------
# BrowserManager – thin wrapper around Playwright
# ---------------------------------------------------------------------------

class BrowserManager:
    """Manage a single Firefox browser session.

    Parameters
    ----------
    headless: bool, default ``True``
        Whether to run Firefox headlessly.
    user_data_dir: str | None
        Path to a persistent user data directory.
    proxy: str | None
        Proxy URL in the form ``http://host:port``.
    """

    def __init__(self, *, headless: bool = True, user_data_dir: Optional[str] = None, proxy: Optional[str] = None):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.proxy = proxy
        self.playwright: Any | None = None
        self.browser: Any | None = None
        self.context: Any | None = None
        self.page: Any | None = None

    def start(self) -> None:
        if self.browser:
            return  # already started
        global sync_playwright
        if sync_playwright is None:
            from playwright.sync_api import sync_playwright as _sp
            sync_playwright = _sp
        self.playwright = sync_playwright().start()
        launch_args: dict = {
            "headless": self.headless,
            "args": ["--no-sandbox", "--disable-dev-shm-usage"],
        }
        if self.proxy:
            launch_args["proxy"] = {"server": self.proxy}
        self.browser = self.playwright.firefox.launch(**launch_args)
        context_args: dict = {}
        if self.user_data_dir:
            context_args["user_data_dir"] = str(Path(self.user_data_dir).expanduser().resolve())
        self.context = self.browser.new_context(**context_args)
        self.page = self.context.new_page()

    def stop(self) -> None:
        if self.context:
            self.context.close()
            self.context = None
        if self.browser:
            self.browser.close()
            self.browser = None
        if self.playwright:
            self.playwright.stop()
            self.playwright = None
        self.page = None

    # ---------------------------------------------------------------------
    # Browser actions
    # ---------------------------------------------------------------------
    def navigate(self, url: str, timeout: int = 30_000) -> dict:
        if not self.page:
            raise RuntimeError("Browser not started – call start() first")
        self.page.goto(url, timeout=timeout)
        return {"url": url}

    def screenshot(self, path: str, full_page: bool = True, **kwargs) -> dict:
        if not self.page:
            raise RuntimeError("Browser not started – call start() first")
        img = self.page.screenshot(full_page=full_page, **kwargs)
        Path(path).write_bytes(img)
        return {"path": path}

    def click(self, selector: str, **kwargs) -> dict:
        if not self.page:
            raise RuntimeError("Browser not started – call start() first")
        self.page.click(selector, **kwargs)
        return {"selector": selector}

    def type_text(self, selector: str, text: str, **kwargs) -> dict:
        if not self.page:
            raise RuntimeError("Browser not started – call start() first")
        self.page.fill(selector, text, **kwargs)
        return {"selector": selector, "text": text}

# ---------------------------------------------------------------------------
# Public function for OpenAI function calling
# ---------------------------------------------------------------------------

_mgr: BrowserManager | None = None


def browser(action: str, *, url: str | None = None, path: str | None = None, selector: str | None = None, text: str | None = None, headless: bool | None = None, user_data_dir: str | None = None, proxy: str | None = None, timeout: int | None = None) -> str:
    """Perform a browser action.

    Parameters
    ----------
    action:
        One of ``start``, ``stop``, ``navigate``, ``screenshot``, ``click`` or ``type``.
    url:
        Target URL for navigation.
    path:
        File path for screenshots.
    selector:
        CSS selector for click or type actions.
    text:
        Text to type.
    headless, user_data_dir, proxy:
        Browser configuration used only on ``start``.
    timeout:
        Navigation timeout in ms.
    """
    global _mgr
    try:
        if action == "start":
            if _mgr is None:
                _mgr = BrowserManager(
                    headless=headless if headless is not None else True,
                    user_data_dir=user_data_dir,
                    proxy=proxy,
                )
            _mgr.start()
            return json.dumps({"result": {"action": "start", "status": "ok"}})
        if _mgr is None:
            return json.dumps({"error": "Browser not started. Call start first."})
        if action == "stop":
            _mgr.stop()
            _mgr = None
            return json.dumps({"result": {"action": "stop", "status": "ok"}})
        if action == "navigate":
            if not url:
                raise ValueError("url is required for navigate")
            _mgr.navigate(url, timeout=timeout or 30_000)
            return json.dumps({"result": {"action": "navigate", "url": url}})
        if action == "screenshot":
            if not path:
                raise ValueError("path is required for screenshot")
            _mgr.screenshot(path)
            return json.dumps({"result": {"action": "screenshot", "path": path}})
        if action == "click":
            if not selector:
                raise ValueError("selector is required for click")
            _mgr.click(selector)
            return json.dumps({"result": {"action": "click", "selector": selector}})
        if action == "type":
            if not selector or text is None:
                raise ValueError("selector and text are required for type")
            _mgr.type_text(selector, text)
            return json.dumps({"result": {"action": "type", "selector": selector, "text": text}})
        # New action to retrieve current page HTML
        if action == "get_html":
            if not _mgr.page:
                raise RuntimeError("Browser not started \u2013 call start() first")
            html = _mgr.page.content()
            return json.dumps({"result": {"action": "get_html", "html": html}})
        return json.dumps({"error": f"Unknown action '{action}'"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

# ---------------------------------------------------------------------------
# OpenAI function call metadata – auto‑discovery reads these
# ---------------------------------------------------------------------------

name = "browser"
func = browser
description = "Control a browser: start, stop, navigate, screenshot, click, type."
# Provide a custom schema so optional fields are optional
schema = {
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["start", "stop", "navigate", "screenshot", "click", "type", "get_html"]},
            "url": {"type": "string"},
            "path": {"type": "string"},
            "selector": {"type": "string"},
            "text": {"type": "string"},
            "headless": {"type": "boolean"},
            "user_data_dir": {"type": "string"},
            "proxy": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": ["action"],
    }
}

# ---------------------------------------------------------------------------
# CLI wrapper for manual testing – optional
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple manual test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m app.tools.browser start|stop|navigate|screenshot|click|type ...")
        sys.exit(1)
    action = sys.argv[1]
    params = {k: v for k, v in (p.split('=') for p in sys.argv[2:])}
    print(browser(action, **params))
