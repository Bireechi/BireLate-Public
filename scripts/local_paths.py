"""Resolve the same dependency manifest as local-paths.ps1, independently of cwd."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def dependency_path(name):
    paths = json.loads((ROOT / 'config' / 'local-paths.json').read_text(encoding='utf-8'))
    value = paths.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Unknown or empty BireLate path: {name}')
    path = Path(value)
    # absolute() preserves the configured spelling instead of dereferencing
    # compatibility junctions; the canonical local path is what callers report.
    return (path if path.is_absolute() else ROOT / path).absolute()
