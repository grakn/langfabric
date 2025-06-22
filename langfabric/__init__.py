from .rebuild_models import rebuild_langchain_models

# Automatically rebuild when the library is imported
rebuild_langchain_models()

from langfabric.schema import (
    AzureOpenAIModelConfig,
    OpenAIModelConfig,
    GroqModelConfig,
    OllamaModelConfig,
    AzureMLModelConfig,
    DeepSeekModelConfig,
    ModelConfig,
)

from .fabric import build_model, build_embeddings
from .manager import ModelManager
from .loader import load_model_configs, load_models

__all__ = [
    "AzureOpenAIModelConfig",
    "OpenAIModelConfig",
    "GroqModelConfig",
    "OllamaModelConfig",
    "AzureMLModelConfig",
    "DeepSeekModelConfig",
    "ModelConfig",
    "ModelManager"
    "build_model",
    "build_embeddings",
    "load_model_configs"
    "load_models"
]
