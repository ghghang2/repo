# app/tools/apply_patch.py
"""
Tool that applies a unified diff to a file inside the repository.

The function accepts three parameters:

* ``path``   – file path *relative to the repository root* (may contain directories)
* ``op_type`` – one of ``create``, ``update`` or ``delete``
* ``diff``   – the unified‑diff string (or empty string for delete)

It returns a JSON string.  On success the JSON contains a ``result`` key;
on failure it contains an ``error`` key.  The format matches the
OpenAI function‑calling schema produced by ``app/tools/__init__``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from dataclasses import dataclass, make_dataclass
from typing import Callable, Literal, Sequence, Optional

def _safe_resolve(repo_root: Path, rel_path: str) -> Path:
    """Resolve ``rel_path`` against ``repo_root`` and guard against directory traversal."""
    target = (repo_root / rel_path).resolve()
    if not str(target).startswith(str(repo_root)):
        raise ValueError("Path escapes repository root")
    return target


# --- CORE DIFF ENGINE ---

ApplyDiffMode = Literal["default", "create"]

@dataclass
class Chunk:
    orig_index: int
    del_lines: list[str]
    ins_lines: list[str]

@dataclass
class ParserState:
    lines: list[str]
    index: int = 0
    fuzz: int = 0

@dataclass
class ReadSectionResult:
    next_context: list[str]
    section_chunks: list[Chunk]
    end_index: int
    eof: bool

# Correctly define the Match object
Match = make_dataclass("Match", [("new_index", int), ("fuzz", int)])

END_PATCH = "*** End Patch"
END_FILE = "*** End of File"
SECTION_TERMINATORS = [END_PATCH, "*** Update File:", "*** Delete File:", "*** Add File:"]
END_SECTION_MARKERS = [*SECTION_TERMINATORS, END_FILE]

def apply_diff(input_str: str, diff: str, mode: ApplyDiffMode = "default") -> str:
    newline = _detect_newline(input_str, diff, mode)
    diff_lines = _normalize_diff_lines(diff)
    if mode == "create":
        return _parse_create_diff(diff_lines, newline=newline)

    normalized_input = input_str.replace("\r\n", "\n")
    parsed_chunks, total_fuzz = _parse_update_diff(diff_lines, normalized_input)
    return _apply_chunks(normalized_input, parsed_chunks, newline=newline)

def _normalize_diff_lines(diff: str) -> list[str]:
    """Clean the diff and strip Unified Diff metadata headers."""
    raw_lines = [line.rstrip("\r") for line in re.split(r"\r?\n", diff)]
    
    clean_lines = []
    for line in raw_lines:
        # Skip headers that look like Unified Diff metadata
        if line.startswith(("--- ", "+++ ", "Index: ", "diff --git")):
            continue
        clean_lines.append(line)
        
    if clean_lines and clean_lines[-1] == "":
        clean_lines.pop()
    return clean_lines

def _detect_newline(input_str: str, diff: str, mode: ApplyDiffMode) -> str:
    text = input_str if mode != "create" and "\n" in input_str else diff
    return "\r\n" if "\r\n" in text else "\n"

def _is_done(state: ParserState, prefixes: Sequence[str]) -> bool:
    if state.index >= len(state.lines): return True
    return any(state.lines[state.index].startswith(p) for p in prefixes)

def _read_str(state: ParserState, prefix: str) -> str:
    if state.index >= len(state.lines): return ""
    current = state.lines[state.index]
    if current.startswith(prefix):
        state.index += 1
        return current[len(prefix) :]
    return ""

def _parse_create_diff(lines: list[str], newline: str) -> str:
    parser = ParserState(lines=[*lines, END_PATCH])
    output = []
    while not _is_done(parser, SECTION_TERMINATORS):
        line = parser.lines[parser.index]
        parser.index += 1
        if not line.startswith("+"): raise ValueError(f"Invalid Add Line: {line}")
        output.append(line[1:])
    return newline.join(output)

def _parse_update_diff(lines: list[str], input_str: str):
    parser = ParserState(lines=[*lines, END_PATCH])
    input_lines = input_str.split("\n")
    chunks, cursor = [], 0

    while not _is_done(parser, END_SECTION_MARKERS):
        anchor = _read_str(parser, "@@ ")
        if anchor == "" and parser.index < len(parser.lines) and parser.lines[parser.index] == "@@":
            parser.index += 1
        
        if anchor.strip():
            cursor = _advance_cursor(anchor, input_lines, cursor, parser)

        section = _read_section(parser.lines, parser.index)
        match = _find_context(input_lines, section.next_context, cursor, section.eof)
        
        if match.new_index == -1:
            raise ValueError(f"Context match failed. Could not find context lines in the file.")

        cursor = match.new_index + len(section.next_context)
        parser.fuzz += match.fuzz
        parser.index = section.end_index

        for ch in section.section_chunks:
            chunks.append(Chunk(orig_index=ch.orig_index + match.new_index,
                                del_lines=list(ch.del_lines), ins_lines=list(ch.ins_lines)))
    return chunks, parser.fuzz

def _advance_cursor(anchor: str, lines: list[str], cursor: int, parser: ParserState) -> int:
    # Try exact anchor match
    for i in range(cursor, len(lines)):
        if lines[i] == anchor:
            return i + 1
            
    # Try fuzzy anchor match (ignoring whitespace)
    for i in range(cursor, len(lines)):
        if lines[i].strip() == anchor.strip():
            parser.fuzz += 1
            return i + 1
            
    return cursor

def _read_section(lines: list[str], start_index: int) -> ReadSectionResult:
    context, del_lines, ins_lines, chunks = [], [], [], []
    mode, index = "keep", start_index
    
    while index < len(lines):
        raw = lines[index]
        if any(raw.startswith(term) for term in END_SECTION_MARKERS) or raw == "***":
            break
        index += 1
        line = raw if raw else " "
        prefix = line[0]
        
        last_mode, mode = mode, {"+": "add", "-": "delete", " ": "keep"}.get(prefix, "keep")
        if mode == "keep" and last_mode != "keep" and (del_lines or ins_lines):
            chunks.append(Chunk(len(context)-len(del_lines), list(del_lines), list(ins_lines)))
            del_lines, ins_lines = [], []
        
        if mode == "delete": del_lines.append(line[1:]); context.append(line[1:])
        elif mode == "add": ins_lines.append(line[1:])
        else: context.append(line[1:])

    if del_lines or ins_lines:
        chunks.append(Chunk(len(context)-len(del_lines), list(del_lines), list(ins_lines)))
    
    is_eof = index < len(lines) and lines[index] == END_FILE
    return ReadSectionResult(context, chunks, index + (1 if is_eof else 0), is_eof)

def _equals_slice(
    source: list[str], target: list[str], start: int, map_fn: Callable[[str], str]
) -> bool:
    """Helper to compare a slice of lines using a transformation function (like strip)."""
    if start + len(target) > len(source):
        return False
    for offset, target_value in enumerate(target):
        if map_fn(source[start + offset]) != map_fn(target_value):
            return False
    return True

def _find_context(lines: list[str], context: list[str], start: int, eof: bool) -> Match:
    if not context:
        return Match(start, 0)

    # Tier 1: Exact Match (The Gold Standard)
    for i in range(start, len(lines) - len(context) + 1):
        if _equals_slice(lines, context, i, lambda v: v):
            return Match(i, 0)

    # Tier 2: Ignore Trailing Whitespace (Common LLM hallucination)
    for i in range(start, len(lines) - len(context) + 1):
        if _equals_slice(lines, context, i, lambda v: v.rstrip()):
            return Match(i, 1)

    # Tier 3: Ignore All Leading/Trailing Whitespace (Indentation flexible)
    for i in range(start, len(lines) - len(context) + 1):
        if _equals_slice(lines, context, i, lambda v: v.strip()):
            return Match(i, 100)

    # Fallback for EOF: If the model thinks it's at the end of the file
    if eof:
        end_start = max(0, len(lines) - len(context))
        # Try a relaxed match at the very end of the file
        if _equals_slice(lines, context, end_start, lambda v: v.strip()):
            return Match(end_start, 10000)

    return Match(-1, 0)

def _apply_chunks(input_str: str, chunks: list[Chunk], newline: str) -> str:
    orig_lines = input_str.split("\n")
    dest_lines, cursor = [], 0
    for chunk in chunks:
        dest_lines.extend(orig_lines[cursor : chunk.orig_index])
        dest_lines.extend(chunk.ins_lines)
        cursor = chunk.orig_index + len(chunk.del_lines)
    dest_lines.extend(orig_lines[cursor:])
    return newline.join(dest_lines)


def apply_patch(path: str, op_type: str, diff: str) -> str:
    """
    Apply a unified diff to a file inside the repository.

    Parameters
    ----------
    path : str
        Relative file path (under the repo root).
    op_type : str
        One of ``create``, ``update`` or ``delete``.
    diff : str
        Unified diff string (ignored for ``delete``).

    Returns
    -------
    str
        JSON string with either ``result`` or ``error``.
    """
    try:
        repo_root = Path(__file__).resolve().parents[2]  # app/tools -> app -> repo root
        target = _safe_resolve(repo_root, path)

        if op_type == "create":
            target.parent.mkdir(parents=True, exist_ok=True)
            content = apply_diff("", diff, mode="create")
            target.write_text(content, encoding="utf-8")
            return json.dumps({"result": f"File created: {path}"})

        elif op_type == "update":
            if not target.exists():
                raise FileNotFoundError(f"File not found: {path}")
            original = target.read_text(encoding="utf-8")
            patched = apply_diff(original, diff)
            target.write_text(patched, encoding="utf-8")
            return json.dumps({"result": f"File updated: {path}"})

        elif op_type == "delete":
            target.unlink(missing_ok=True)
            return json.dumps({"result": f"File deleted: {path}"})

        else:
            raise ValueError(f"Unsupported operation type: {op_type}")

    except Exception as exc:
        # Any exception becomes an error JSON
        return json.dumps({"error": str(exc)})

# Exported names for automatic tool discovery
func = apply_patch
name = "apply_patch"
description = (
    "Apply a unified diff to a file inside the repository. "
    "Supports create, update and delete operations. "
    "Returns a JSON string with either a `result` key or an `error` key."
)

__all__ = ["func", "name", "description"]