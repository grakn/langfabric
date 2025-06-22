from pathlib import Path
from typing import Union, Sequence, Any
from pydantic import TypeAdapter
from seyaml import load_seyaml

from .schema import ModelConfig
from .fabric import build_model

def load_model_configs(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    secrets: dict = None,
) -> dict[str, ModelConfig]:
    """
    Load model configurations from one or more YAML files or directories.
    """
    secrets = secrets or {}
    model_configs = {}

    files = _resolve_yaml_files(paths)

    for file_path in files:
        for model_vars in load_seyaml(file_path, secrets=secrets):
            try:
                model_config = TypeAdapter(ModelConfig).validate_python(model_vars)
                model_configs[model_config.name] = model_config
            except Exception:
                name = model_vars.get("name", None)
                raise Exception(f"Error parsing model config with name: {name}")
    return model_configs


def load_models(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    secrets: dict = None,
) -> dict[str, Any]:
    """
    Load and instantiate models from YAML files or directories.
    """
    secrets = secrets or {}
    models = {}

    files = _resolve_yaml_files(paths)

    for file_path in files:
        for model_vars in load_seyaml(file_path, secrets=secrets):
            try:
                model_config = TypeAdapter(ModelConfig).validate_python(model_vars)
                models[model_config.name] = build_model(model_config)
            except Exception:
                name = model_vars.get("name", None)
                raise Exception(f"Error building model config with name: {name}")
    return models


def _resolve_yaml_files(
    paths: Union[str, Path, Sequence[Union[str, Path]]]
) -> list[Path]:
    """
    Accepts paths or list of paths (directories or files) and returns all YAML files.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    resolved = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.yaml")))
        elif path.is_file():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"No such file or directory: {path}")
    return resolved
