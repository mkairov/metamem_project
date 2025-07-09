import torch
import transformers

import importlib
from typing import List
import inspect


def get_fn_param_names(fn) -> List[str]:
    """get function parameters names except *args, **kwargs

    Args:
        fn: function or method

    Returns:
        List[str]: list of function parameters names
    """
    params = []
    for p in inspect.signature(fn).parameters.values():
        if p.kind not in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]:
            params += [p.name]
    return params


def get_cls_by_name(name: str) -> type:
    """Get class by its name and module path.

    Args:
        name (str): e.g., transfomers:T5ForConditionalGeneration, modeling_t5:my_class

    Returns:
        type: found class for `name`
    """
    module_name, cls_name = name.split(':')
    return getattr(importlib.import_module(module_name), cls_name)


def get_optimizer(name: str):
    if ':' in name:
        return get_cls_by_name(name)
    # if hasattr(lm_experiments_tools.optimizers, name):
    #     return getattr(lm_experiments_tools.optimizers, name)
    if hasattr(torch.optim, name):
        return getattr(torch.optim, name)
    if hasattr(transformers.optimization, name):
        return getattr(transformers.optimization, name)
    # if hasattr(schedulefree, name):
    #     return getattr(schedulefree, name)
    try:
        apex_opt = importlib.import_module('apex.optimizers')
        return getattr(apex_opt, name)
    except (ImportError, AttributeError):
        pass
    return None


class ObjectView(dict):
    def __init__(self, *args, **kwargs):
        super(ObjectView, self).__init__(**kwargs)
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    self[key] = val
            else:
                raise TypeError()
        for key, val in kwargs.items():
            self[key] = val

    def __setattr__(self, key, value):
        if not hasattr(ObjectView, key):
            self[key] = value
        else:
            raise

    def __setitem__(self, name, value):
        value = ObjectView(value) if isinstance(value, dict) else value
        super(ObjectView, self).__setitem__(name, value)

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __getitem__(self, name):
        if name not in self:
            self[name] = {}
        return super(ObjectView, self).__getitem__(name)

    def __delattr__(self, name):
        del self[name]
