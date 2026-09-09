from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def executable() -> Path:
    path = _ROOT / "bin" / "proseg"
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError("the pinned Proseg executable is unavailable")
    return path


def subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    library_path = str(_ROOT / "lib")
    existing = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        f"{library_path}{os.pathsep}{existing}" if existing else library_path
    )
    return environment


__all__ = ["executable", "subprocess_environment"]
