"""Registry"""

import importlib
import sys
from typing import List, Type


class TypeRegistry(object):
    def __init__(self, bound: Type = None):
        self._registry = {}

    def __contains__(self, key):
        return key in self._registry

    def __getitem__(self, key):
        if key not in self._registry:
            raise KeyError(f"{key} not registered")
        return self._registry[key]

    def __setitem__(self, key, value):
        if key is None:
            key = value.__name__
        if key in self:
            raise ValueError(f"Name {key} already registered")
        self._registry[key] = value

    def __iter__(self):
        return iter(self._registry)

    def keys(self):
        return self._registry.keys()

    def values(self):
        return self._registry.values()

    def register(self, key_or_value=None):
        def decorator(value, key):
            self[key] = value
            return value

        if callable(key_or_value):
            return decorator(key_or_value, None)
        else:
            return lambda value: decorator(value, key_or_value)


TYPE_REGISTRY = TypeRegistry()

register_type = TYPE_REGISTRY.register


def type_from_path(t: str) -> Type:
    # inspired by https://github.com/django/django/blob/main/django/utils/module_loading.py
    try:
        module_path, class_name = t.rsplit(".", 1)
    except ValueError:
        raise Exception(f"Type string {t} is not a class name nor a valid path")
    try:
        importlib.import_module(module_path)
        module = sys.modules.get(module_path)
        return getattr(module, class_name)
    except AttributeError:
        raise Exception(f"Type class {class_name} not defined in module {module_path}")


def type_from_str(s: str, bound: Type = None) -> Type:
    if s in TYPE_REGISTRY and (bound is None or issubclass(TYPE_REGISTRY[s], bound)):
        return TYPE_REGISTRY[s]
    return type_from_path(s)


def list_type_keys(bound: Type = None) -> List[str]:
    return [
        k for k in TYPE_REGISTRY.keys() if bound is None or issubclass(TYPE_REGISTRY[k], bound)
    ]


def list_types(bound: Type = None) -> List[Type]:
    return [t for t in TYPE_REGISTRY.values() if bound is None or issubclass(t, bound)]
