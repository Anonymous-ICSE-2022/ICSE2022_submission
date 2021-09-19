from __future__ import annotations

from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Any


class Task(ABC):
    @abstractstaticmethod
    def from_config(cfg: Any) -> Task:
        ...

    @abstractmethod
    def run_training(self) -> None:
        ...
