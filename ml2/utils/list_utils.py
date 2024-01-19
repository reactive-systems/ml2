"""List utilities"""

from typing import List, TypeVar

T = TypeVar("T")


def join_lists(delimiter: T, lists: List[List[T]]) -> List[T]:
    result: List[T] = []
    for l in lists:
        if result != []:
            result.append(delimiter)
        result.extend(l)
    return result
