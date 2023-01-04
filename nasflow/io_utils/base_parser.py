from typing import (
    Dict,
    Any
)

def parse_args_from_kwargs(kwargs: Dict[Any, Any], key: str, default_value=None):
    if not key in kwargs:
        return default_value
    return kwargs[key]
