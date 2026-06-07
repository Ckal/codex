from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Small name-to-service registry."""

    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        self._items[name] = item

    def get(self, name: str) -> T:
        if name not in self._items:
            available = ", ".join(self._items)
            raise KeyError(f"{name!r} is not registered. Available: {available}")
        return self._items[name]

    def list(self) -> list[str]:
        return list(self._items)
