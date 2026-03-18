from __future__ import annotations

from typing import Any, Optional


def require_optional(module: str, pip_name: Optional[str] = None) -> Any:
    try:
        return __import__(module)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{module}'. Install with: pip install {pip_name or module}"
        ) from e
