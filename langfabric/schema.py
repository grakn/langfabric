from typing import Literal, Optional, Union, Dict
from pydantic import BaseModel, SecretStr


class BaseModelConfig(BaseModel):
    name: str
    model: Optional[str] = None
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.1
    streaming: Optional[bool] = False


class AzureOpenAIModelConfig(BaseModelConfig):
    provider: Literal["azure_openai"]
    deployment_name: str
    api_key: SecretStr
    endpoint: str
    api_version: str
    openai_api_type: Optional[str] = "azure"  # override support
    embeddings_deployment_name: Optional[str] = None

class OpenAIModelConfig(BaseModelConfig):
    provider: Literal["openai"]
    api_key: SecretStr
    api_base: Optional[str] = None
    api_version: Optional[str] = None


class GroqModelConfig(BaseModelConfig):
    provider: Literal["groq"]
    api_key: SecretStr


class OllamaModelConfig(BaseModelConfig):
    provider: Literal["ollama"]
    local: Optional[bool] = True


class DeepSeekModelConfig(BaseModelConfig):
    provider: Literal["deepseek"]
    api_key: SecretStr
    endpoint_url: str
    timeout: Optional[int] = 10


class AzureMLModelConfig(BaseModelConfig):
    provider: Literal["azureml"]
    endpoint_url: str
    api_key: SecretStr


ModelConfig = Union[
    AzureOpenAIModelConfig,
    OpenAIModelConfig,
    GroqModelConfig,
    OllamaModelConfig,
    DeepSeekModelConfig,
    AzureMLModelConfig,
]
