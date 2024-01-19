"""Typing utilities"""

from typing import Optional, TypeVar, Union, _GenericAlias, _SpecialForm, get_args, get_origin


def process_typevar(t):
    if isinstance(t, TypeVar):
        if t.__bound__ is not None:
            return process_typevar(t.__bound__)
        else:
            return object
    else:
        return t


# TODO commutativity of Union
def is_subclass_generic(t1, t2) -> bool:
    t1 = process_typevar(t1)
    t2 = process_typevar(t2)
    # subclass check for types
    if isinstance(t1, type) and isinstance(t2, type):
        return issubclass(t1, t2)
    # recurse on generics, e.g. List[int]
    if isinstance(t1, _GenericAlias) or isinstance(t2, _GenericAlias):
        origin_t1 = get_origin(t1) if isinstance(t1, _GenericAlias) else t1
        origin_t2 = get_origin(t2) if isinstance(t2, _GenericAlias) else t2
        if not is_subclass_generic(origin_t1, origin_t2):
            return False
        args_t1 = get_args(t1)
        args_t2 = get_args(t2)
        # return true if args not specified, e.g., Dict
        if len(args_t2) == 0:
            return True
        # return false if number of args do not match, e.g. Tuple[int, str] and Tuple[int]
        if len(args_t1) != len(args_t2):
            return False
        # recurse on args
        for a_t1, a_t2 in zip(args_t1, args_t2):
            if not is_subclass_generic(a_t1, a_t2):
                return False
        return True
    # check equality for special forms, e.g. Optional
    if isinstance(t1, _SpecialForm) or isinstance(t2, _SpecialForm):
        # Optional and Union are effectively the same as Optional is a Union with NoneType
        if (t1, t2) in [(Union, Optional), (Optional, Union)]:
            return True
        return t1 == t2
    raise ValueError(f"Invalid types: {t1}, {t2}")
