from typing import Optional


from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLChatOnlineEndpoint,
    AzureMLEndpointApiType,
    LlamaChatContentFormatter,
)

from langfabric.schema import (
    AzureOpenAIModelConfig,
    OpenAIModelConfig,
    GroqModelConfig,
    OllamaModelConfig,
    AzureMLModelConfig,
    DeepSeekModelConfig,
    ModelConfig,
)

def build_model(
    config: ModelConfig,
    json_response: bool = False,
    max_retries: int = 2,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    temperature = temperature if temperature is not None else getattr(config, "temperature", 0.1)
    max_tokens = max_tokens if max_tokens is not None else getattr(config, "max_tokens", 2048)

    if isinstance(config, AzureOpenAIModelConfig):
        return AzureChatOpenAI(
            model=config.model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            streaming=config.streaming,
            openai_api_version=config.api_version,
            azure_endpoint=config.endpoint,
            deployment_name=config.deployment_name,
            openai_api_type=config.openai_api_type,
            api_key=config.api_key.get_secret_value(),
            model_kwargs={"response_format": {"type": "json_object"}} if json_response else {},
        )

    elif isinstance(config, OpenAIModelConfig):
        return ChatOpenAI(
            model=config.model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            streaming=config.streaming,
            api_key=config.api_key.get_secret_value(),
            openai_api_base=config.api_base,
            openai_api_version=config.api_version,
            model_kwargs={"response_format": {"type": "json_object"}} if json_response else {},
        )


    elif isinstance(config, GroqModelConfig):
        return ChatGroq(
            model_name=getattr(config, "model", config.name),
            temperature=temperature,
            groq_api_key=config.api_key.get_secret_value(),
            model_kwargs={"response_format": {"type": "json_object"}} if json_response else {},
        )

    elif isinstance(config, OllamaModelConfig):
        return ChatOllama(
            model=getattr(config, "model", config.name),
            temperature=temperature,
        )

    elif isinstance(config, DeepSeekModelConfig):
        from langchain_community.chat_models.azureml_endpoint import CustomOpenAIChatContentFormatter

        return AzureMLChatOnlineEndpoint(
            endpoint_url=config.endpoint,
            endpoint_api_type=AzureMLEndpointApiType.serverless,
            endpoint_api_key=config.api_key.get_secret_value(),
            content_formatter=CustomOpenAIChatContentFormatter(),
            model_kwargs={
                "temperature": temperature,
                "max_tokens": max_tokens,
                **({"response_format": {"type": "json_object"}} if json_response else {}),
            },
            timeout=config.timeout,
        )
            
    elif isinstance(config, AzureMLModelConfig):
        return AzureMLChatOnlineEndpoint(
            endpoint_url=config.endpoint_url,
            endpoint_api_type=AzureMLEndpointApiType.serverless,
            endpoint_api_key=config.api_key.get_secret_value(),
            content_formatter=LlamaChatContentFormatter(),
            model_kwargs={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    raise ValueError(f"Unsupported provider type: {config}")
