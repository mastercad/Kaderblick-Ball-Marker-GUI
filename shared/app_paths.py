"""Central paths for bundled resources and runtime data."""

from __future__ import annotations

import os
import sys
from pathlib import Path


APP_VENDOR = "Kaderblick"
APP_NAME = "BallMarker"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    return project_root().joinpath(*parts)


def runtime_root() -> Path:
    if is_frozen():
        return user_data_dir()
    return project_root()


def runtime_path(*parts: str) -> Path:
    return runtime_root().joinpath(*parts)


def user_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / APP_VENDOR / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_VENDOR / APP_NAME

    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "kaderblick-ballmarker"


def user_data_path(*parts: str) -> Path:
    return user_data_dir().joinpath(*parts)
