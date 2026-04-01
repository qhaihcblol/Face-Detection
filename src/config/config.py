from copy import deepcopy

from .mobilenet_v2 import cfg_mobilenet_v2


_CONFIGS = {
    "mobilenet_v2": cfg_mobilenet_v2,
}


def get_config(name: str):
    """
    Get configuration by model name.

    Args:
        name (str): Name of the backbone/model.

    Returns:
        dict: A deep copy of the configuration.

    Raises:
        ValueError: If config name is not found.
    """
    if name not in _CONFIGS:
        available = ", ".join(_CONFIGS.keys())
        raise ValueError(f"Unknown config '{name}'. Available: [{available}]")

    return deepcopy(_CONFIGS[name])
