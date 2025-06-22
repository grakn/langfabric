# Langfabric

**Langfabric** is a flexible Python framework for managing, instantiating, and caching Large Language Model (LLM) instances from YAML configuration files.  
It supports OpenAI, Azure OpenAI, Groq, Ollama, AzureML, and other providers, making LLM orchestration and deployment easy, reproducible, and robust.

---

## Features

- **Declarative YAML model configs** (with secret support via [seyaml](https://github.com/grakn/seyaml))
- **Multiple provider support:** OpenAI, Azure OpenAI, Groq, Ollama, AzureML, and more
- **Thread-safe model caching**
- **Runtime overrides:** temperature, max tokens, etc.
- **Parallel/preloaded model initialization**
- **Automatic LangChain model rebuild**

---

## Installation

```
pip install langfabric
```

Or clone the repo and install locally:

```
git clone https://github.com/grakn/langfabric.git
cd langfabric
pip install -e .
```

# Example: Model Configuration YAML

```
# models.yaml
- name: gpt4o
  provider: azure_openai
  model: gpt-4o
  deployment_name: gpt-4o-deployment
  api_key: !env AZURE_OPENAI_API_KEY
  endpoint: https://your-endpoint.openai.azure.com
  api_version: 2024-06-01-preview
  max_tokens: 4096
  temperature: 0.1
- name: ollama
  provider: ollama
  model: llama3
  max_tokens: 4096
```

# Usage

## 1. Load model configs

```
from langfabric.loader import load_model_configs

model_configs = load_model_configs("./models.yaml")
```

## 2. Build and cache models
```
from langfabric.manager import ModelManager

manager = ModelManager(model_configs)
model = manager.get("gpt4o")  # Get Azure OpenAI GPT-4o model
```

## 3. Optional preload all models in parallel (multi-threaded)
```
manager.preload()  # Warms up cache in threads for all configs
```

## 4. Use runtime parameter overrides
```
custom_model = manager.get(
    "gpt4o",
    temperature=0.5,
    max_tokens=2048,
    json_response=True,
    streaming=False,
)
```

## 5. Iterate over all models
```
for name, model in manager.items():
    print(f"{name}: {model}")
```

# Advanced

## Load model configs with secrets

```
from langfabric.loader import load_model_configs

secrets = {"api_key": "sk-..."}
model_configs = load_model_configs("./models/model.yaml", secrets)
```

Use !secrets pre-processor to point on secret names

```
# models.yaml
- name: gpt4o
  provider: azure_openai
  model: gpt-4o
  deployment_name: gpt-4o-deployment
  api_key: !secret api_key
  endpoint: https://your-endpoint.openai.azure.com
  api_version: 2024-06-01-preview
  max_tokens: 4096
  temperature: 0.1
- name: ollama
  provider: ollama
  model: llama3
  max_tokens: 4096
```

## Load multiple model config files with secrets

```
from langfabric.loader import load_model_configs

secrets = {"api_key": "sk-..."}
model_configs = load_model_configs(["./models/models1.yaml", "./models/models2.yaml"], secrets)
```

## Load multiple model config files from directories

```
from langfabric.loader import load_model_configs

secrets = {"api_key": "sk-..."}
model_configs = load_model_configs(["./models1/", "./models2/], secrets)
```
