from dataclasses import dataclass
from typing import Callable, Dict, Type

from hydra.core.config_store import ConfigStore

MODULES: Dict[str, Callable] = {}

cs = ConfigStore.instance()


def register_module(name: str, config_dataclass: Type):
    assert name not in MODULES, "Attempting to create duplicate model"

    def _register(fn: Callable) -> None:
        MODULES[name] = fn
        node = config_dataclass()
        node._name = name
        cs.store(group="module", name=name, node=node)

    return _register
