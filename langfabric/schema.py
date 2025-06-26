from typing import Union, Tuple, Literal, Optional, Union, Dict, Any
from pydantic import BaseModel, SecretStr


class BaseModelConfig(BaseModel):
    name: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    streaming: Optional[bool] = False
    request_timeout: Union[float, Tuple[float, float], Any, None] = None
    verbose: Optional[bool] = False


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
    organization: Optional[str] = None

class GroqModelConfig(BaseModelConfig):
    provider: Literal["groq"]
    api_key: SecretStr


class OllamaModelConfig(BaseModelConfig):
    provider: Literal["ollama"]
    base_url: Optional[str] = "http://localhost:11434"


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
