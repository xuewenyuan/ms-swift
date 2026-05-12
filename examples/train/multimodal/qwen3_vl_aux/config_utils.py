from typing import Any


class ConfigNode(dict):

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def to_config_node(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value
    if isinstance(value, dict):
        return ConfigNode({key: to_config_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_config_node(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_config_node(item) for item in value)
    return value
