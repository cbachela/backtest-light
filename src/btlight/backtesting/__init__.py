from __future__ import annotations
import importlib
import inspect
import pkgutil
from typing import Any, Dict

"""
Re-export all public classes and functions from backtesting package.
"""


__all__: list[str] = []

_package = __name__
_package_path = __path__  # type: ignore[name-defined]

_objects: Dict[str, Any] = {}

for _, module_name, _ in pkgutil.iter_modules(_package_path):
    module = importlib.import_module(f"{_package}.{module_name}")
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if inspect.isclass(obj) or inspect.isfunction(obj):
            _objects[name] = obj

globals().update(_objects)
__all__ = sorted(_objects.keys())