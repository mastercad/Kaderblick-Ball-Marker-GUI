"""Optional external Python runtime paths for locally installed GPU packages."""

from __future__ import annotations

import json
import os
import site
import sys
import sysconfig
from pathlib import Path
from typing import Iterable

from shared.app_paths import runtime_path


CONFIG_PATH = runtime_path("runtime_config.json")
ENV_SITE_PACKAGES = "KADERBLICK_EXTRA_SITE_PACKAGES"

_applied_paths: list[str] = []


def external_runtime_supported() -> bool:
    return sys.platform.startswith("linux")


def _existing_dirs(paths: Iterable[str | Path]) -> list[str]:
    result: list[str] = []
    for path in paths:
        expanded = Path(path).expanduser()
        if expanded.is_dir():
            resolved = str(expanded.resolve())
            if resolved not in result:
                result.append(resolved)
    return result


def _site_packages_from_venv(path: Path) -> list[Path]:
    candidates: list[Path] = []
    if sys.platform == "win32":
        candidates.append(path / "Lib" / "site-packages")

    lib_dir = path / "lib"
    if lib_dir.is_dir():
        candidates.extend(sorted(lib_dir.glob("python*/site-packages")))
    return candidates


def _looks_like_site_packages(path: Path) -> bool:
    return (path / "torch").exists() or (path / "ultralytics").exists()


def normalize_python_package_paths(paths: Iterable[str | Path]) -> list[str]:
    candidates: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if _looks_like_site_packages(path):
            candidates.append(path)
        candidates.extend(
            candidate for candidate in _site_packages_from_venv(path)
            if _looks_like_site_packages(candidate)
        )
    return _existing_dirs(candidates)


def _glob_existing(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path().glob(pattern) if not pattern.startswith("/") else Path("/").glob(pattern[1:]))
    return paths


def auto_discovered_package_paths() -> list[str]:
    if not external_runtime_supported():
        return []

    paths: list[str | Path] = [runtime_path("gpu-runtime")]

    for env_name in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        env_value = os.environ.get(env_name)
        if env_value:
            paths.append(env_value)

    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        paths.append(user_site)

    purelib = sysconfig.get_paths().get("purelib")
    platlib = sysconfig.get_paths().get("platlib")
    if purelib:
        paths.append(purelib)
    if platlib:
        paths.append(platlib)

    if sys.platform.startswith("linux"):
        home = Path.home()
        paths.extend(_glob_existing([
            str(home / ".local/lib/python*/site-packages"),
            str(home / ".local/lib/python*/dist-packages"),
            "/usr/local/lib/python*/site-packages",
            "/usr/local/lib/python*/dist-packages",
            "/usr/lib/python*/site-packages",
            "/usr/lib/python*/dist-packages",
        ]))

    return normalize_python_package_paths(paths)


def load_runtime_config() -> dict:
    if not external_runtime_supported():
        return {}
    if not CONFIG_PATH.is_file():
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_external_package_paths(paths: Iterable[str | Path]) -> list[str]:
    if not external_runtime_supported():
        return []

    normalized = normalize_python_package_paths(paths)
    os.makedirs(CONFIG_PATH.parent, exist_ok=True)
    data = load_runtime_config()
    data["external_python_package_paths"] = normalized
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return normalized


def configured_external_package_paths() -> list[str]:
    if not external_runtime_supported():
        return []

    paths: list[str] = auto_discovered_package_paths()

    env_value = os.environ.get(ENV_SITE_PACKAGES, "")
    if env_value:
        paths.extend(env_value.split(os.pathsep))

    configured = load_runtime_config().get("external_python_package_paths", [])
    if isinstance(configured, list):
        paths.extend(str(path) for path in configured)

    return normalize_python_package_paths(paths)


def apply_external_python_paths() -> list[str]:
    global _applied_paths

    if not external_runtime_supported():
        _applied_paths = []
        return []

    paths = configured_external_package_paths()
    for path in reversed(paths):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
        if path not in site.getsitepackages():
            site.addsitedir(path)

    _applied_paths = paths
    return paths


def applied_external_python_paths() -> list[str]:
    return list(_applied_paths)
