import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pytest
from unittest.mock import patch, MagicMock

from langfabric.schema import (
    AzureOpenAIModelConfig,
    OpenAIModelConfig,
    GroqModelConfig,
    OllamaModelConfig,
)

from langfabric.fabric import build_embeddings

def test_build_azure_openai_embeddings():
    config = AzureOpenAIModelConfig(
        name="test-azure",
        provider="azure_openai",
        deployment_name="test-deployment",
        api_key="sk-test",
        endpoint="https://fake.azure.com",
        api_version="2024-05-01-preview",
    )
    with patch("langchain_openai.AzureOpenAIEmbeddings") as mock_embed:
        mock_instance = MagicMock()
        mock_embed.return_value = mock_instance

        result = build_embeddings(config, chunk_size=2)
        assert result is mock_instance
        mock_embed.assert_called_once()
        kwargs = mock_embed.call_args.kwargs
        assert kwargs["deployment"] == "test-deployment"
        assert kwargs["chunk_size"] == 2

def test_build_openai_embeddings():
    config = OpenAIModelConfig(
        name="test-openai",
        provider="openai",
        model="text-embedding-ada-002",
        api_key="sk-openai",
    )
    with patch("langchain_openai.OpenAIEmbeddings") as mock_embed:
        mock_instance = MagicMock()
        mock_embed.return_value = mock_instance

        result = build_embeddings(config)
        assert result is mock_instance
        mock_embed.assert_called_once()
        kwargs = mock_embed.call_args.kwargs
        assert kwargs["model"] == "text-embedding-ada-002"

def test_build_ollama_embeddings():
    config = OllamaModelConfig(
        name="test-ollama",
        provider="ollama",
        model="ollama-embed"
    )
    with patch("langchain_ollama.OllamaEmbeddings") as mock_embed:
        mock_instance = MagicMock()
        mock_embed.return_value = mock_instance

        result = build_embeddings(config)
        assert result is mock_instance
        mock_embed.assert_called_once()

def test_build_unsupported_embeddings():
    from langfabric.schema import AzureMLModelConfig
    config = AzureMLModelConfig(
        name="test-azureml",
        provider="azureml",
        endpoint_url="https://azureml-endpoint",
        api_key="sk-aml",
    )
    with pytest.raises(NotImplementedError):
        build_embeddings(config)
