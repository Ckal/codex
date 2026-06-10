from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import SimpleNamespace
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
spaces: Any

try:
    spaces = import_module("spaces")
except ImportError:

    def _gpu(*_args: Any, **_kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        return decorator

    spaces = SimpleNamespace(GPU=_gpu)
