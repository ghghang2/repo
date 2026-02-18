# app/tools/__init__.py
# --------------------
# Automatically discovers any *.py file in this package that defines
# a callable (either via a `func` attribute or the first callable
# in the module).  It generates a minimal JSON‑schema from the
# function’s signature and exposes a list of :class:`Tool` objects
# as well as :func:`get_tools()` for the OpenAI API.

from __future__ import annotations

import inspect
import pkgutil
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

# ----- Schema generator -------------------------------------------------
def _generate_schema(func: Callable) -> Dict[str, Any]:
    sig = inspect.signature(func)
    properties: Dict[str, Dict[str, str]] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect._empty:
            ann_type = "string"
        elif ann in (int, float, complex):
            ann_type = "number"
        else:
            ann_type = "string"
        properties[name] = {"type": ann_type}
        if param.default is inspect._empty:
            required.append(name)
    return {
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    }

# ----- Tool dataclass ---------------------------------------------------
@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    schema: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.schema:
            self.schema = _generate_schema(self.func)

# ----- Automatic discovery ----------------------------------------------
TOOLS: List[Tool] = []

package_path = Path(__file__).parent
for _, module_name, is_pkg in pkgutil.iter_modules([str(package_path)]):
    if is_pkg or module_name == "__init__":
        continue
    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
    except Exception:
        continue

    func: Callable | None = getattr(module, "func", None)
    if func is None:
        # Fallback: first callable in the module
        for attr in module.__dict__.values():
            if callable(attr):
                func = attr
                break
    if not callable(func):
        continue

    name: str = getattr(module, "name", func.__name__)
    description: str = getattr(module, "description", func.__doc__ or "")
    schema: Dict[str, Any] = getattr(module, "schema", _generate_schema(func))

    TOOLS.append(Tool(name=name, description=description, func=func, schema=schema))

# ----- OpenAI helper ----------------------------------------------------
def get_tools() -> List[Dict]:
    """Return the list of tools formatted for chat.completions.create."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema.get("parameters", {}),
            },
        }
        for t in TOOLS
    ]

# ----- Debug ------------------------------------------------------------
if __name__ == "__main__":
    import json
    print(json.dumps([t.__dict__ for t in TOOLS], indent=2))