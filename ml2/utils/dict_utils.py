"""Dictionary utilities"""

from typing import Callable, Dict


def map_nested_dict(f: Callable, d: Dict) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            map_nested_dict(f, v)
        else:
            d[k] = f(v)
