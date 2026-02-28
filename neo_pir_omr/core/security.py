from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SecurityPolicy:
    """Basic safety controls for user-provided images & exports."""

    max_upload_mb: int = 15
    allowed_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


def ensure_within_dir(path: str | os.PathLike, base_dir: str | os.PathLike) -> Path:
    """Prevents path traversal when writing exports."""
    base = Path(base_dir).resolve()
    p = Path(path).resolve()
    if base not in p.parents and p != base:
        raise ValueError("Refused path outside output directory")
    return p


def validate_file_bytes(filename: str, size_bytes: int, policy: SecurityPolicy) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in policy.allowed_exts:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: {policy.allowed_exts}")

    max_bytes = int(policy.max_upload_mb) * 1024 * 1024
    if size_bytes > max_bytes:
        raise ValueError(f"File too large: {size_bytes/1024/1024:.1f}MB (max {policy.max_upload_mb}MB)")
