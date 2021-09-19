import inspect
from dataclasses import dataclass
from typing import Callable, Generic, Type, TypeVar

T = TypeVar("T")


def with_target(cls: Type[T], fn: Callable) -> Type[T]:
    cls._target_ = f"{fn.__module__}.{fn.__qualname__}"  # type: ignore

    return cls
