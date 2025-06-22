def rebuild_langchain_models():
    from typing import Union  # Needed by LangChain models
    from langchain_core.caches import BaseCache
    from langchain_core.callbacks import Callbacks

    # Import all models that require rebuild
    from langchain_openai import AzureChatOpenAI
    from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint
    from langchain_community.chat_models.ollama import ChatOllama
    from langchain_groq import ChatGroq

    # Rebuild models to resolve forward refs
    AzureChatOpenAI.model_rebuild()
    AzureMLChatOnlineEndpoint.model_rebuild()
    ChatOllama.model_rebuild()
    ChatGroq.model_rebuild()
