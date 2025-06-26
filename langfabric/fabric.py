from typing import Union, Tuple, Literal, Optional, Union, Dict, Any

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
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
    json_response: Optional[bool] = None,
    max_retries: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: Optional[bool] = None,
    request_timeout: Union[float, Tuple[float, float], Any, None] = None,
    verbose: Optional[bool] = None,
):
    temperature = temperature or getattr(config, "temperature", None)
    max_tokens = max_tokens or getattr(config, "max_tokens", None)
    streaming = streaming or getattr(config, "streaming", False)
    request_timeout = request_timeout or getattr(config, "request_timeout", None)
    verbose = verbose or getattr(config, "verbose", False)

    if isinstance(config, AzureOpenAIModelConfig):
        return AzureChatOpenAI(
            model=config.model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            streaming=streaming,
            request_timeout=request_timeout,
            verbose=verbose,
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
            streaming=streaming,
            request_timeout=request_timeout,
            verbose=verbose,
            api_key=config.api_key.get_secret_value(),
            openai_api_base=config.api_base,
            openai_organization=config.organization,
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
            base_url=config.base_url,
            temperature=temperature,
        )

    elif isinstance(config, DeepSeekModelConfig):
        from langchain_community.chat_models.azureml_endpoint import CustomOpenAIChatContentFormatter

        return AzureMLChatOnlineEndpoint(
            endpoint_url=config.endpoint_url,
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
        from langchain_community.chat_models.azureml_endpoint import CustomOpenAIChatContentFormatter

        return AzureMLChatOnlineEndpoint(
            endpoint_url=config.endpoint_url,
            endpoint_api_type=AzureMLEndpointApiType.serverless,
            endpoint_api_key=config.api_key.get_secret_value(),
            content_formatter=CustomOpenAIChatContentFormatter(),
            model_kwargs={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    raise ValueError(f"Unsupported provider type: {type(config)}")


def build_embeddings(
    config,
    chunk_size: Optional[int] = 1,
    max_retries: Optional[int] = 2,
    **kwargs
):
    """
    Instantiate an embeddings object based on the model config/provider.
    """
    # --- Azure OpenAI ---
    if isinstance(config, AzureOpenAIModelConfig):
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            deployment=config.embeddings_deployment_name or config.deployment_name,
            azure_endpoint=config.endpoint,
            api_key=config.api_key.get_secret_value(),
            openai_api_type=config.openai_api_type,
            api_version=config.api_version,
            chunk_size=chunk_size,
            max_retries=max_retries,
            **kwargs,
        )

    # --- OpenAI ---
    elif isinstance(config, OpenAIModelConfig):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.model,
            api_key=config.api_key.get_secret_value(),
            openai_api_base=config.api_base,
            chunk_size=chunk_size,
            max_retries=max_retries,
            **kwargs,
        )

    # --- Groq (if supported) ---
    elif isinstance(config, GroqModelConfig):
        try:
            from langchain_groq import GroqEmbeddings
        except ImportError:
            raise ImportError("Install langchain-groq for Groq embeddings support.")
        return GroqEmbeddings(
            model=config.model or config.name,
            api_key=config.api_key.get_secret_value(),
            chunk_size=chunk_size,
            max_retries=max_retries,
            **kwargs,
        )

    # --- Ollama (if supported) ---
    elif isinstance(config, OllamaModelConfig):
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            raise ImportError("Install langchain-community for Ollama embeddings support.")
        return OllamaEmbeddings(
            model=config.model or config.name,
            base_url=config.base_url,
            **kwargs,
        )

    # --- AzureML (if/when supported) ---
    elif isinstance(config, AzureMLModelConfig):
        # Placeholder, adjust if AzureML embeddings class exists
        raise NotImplementedError("Embeddings not implemented for AzureML endpoints yet.")

    # --- DeepSeek and others (add as needed) ---
    elif isinstance(config, DeepSeekModelConfig):
        raise NotImplementedError("Embeddings not implemented for DeepSeek provider.")

    # --- Not supported ---
    else:
        raise ValueError(f"Embeddings not supported for model type: {type(config)}")
