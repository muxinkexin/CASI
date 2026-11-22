"""Mode transitions for the standard modules used by this Lite-Mono DSI run.

Set Python bool flags explicitly instead of recursively invoking possibly wrapped
train()/eval() methods. No PyTorch globals, weights, buffers or optimizer state
are changed. This is not a generic replacement for custom train() side effects.
"""


def set_mode_tree(module, training):
    if training is not True and training is not False:
        raise TypeError("DSI mode must be a native Python True or False")
    for child in module.modules():
        object.__setattr__(child, "training", training)
    return module


def freeze_module(module):
    set_mode_tree(module, False)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return module
