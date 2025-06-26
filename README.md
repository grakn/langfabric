# Langfabric

![Unit Tests](https://github.com/grakn/langfabric/actions/workflows/test.yaml/badge.svg?branch=main)
[![PyPI Downloads](https://static.pepy.tech/badge/langfabric)](https://pepy.tech/projects/langfabric)

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
- **Automatic caching in ModelManager**

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

- name: llama3
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
model = manager.load("gpt4o")  # Get Azure OpenAI GPT-4o model
```

## 3. Optional preload all models in parallel (multi-threaded)
```
manager.preload_all()  # Warms up cache in threads for all configs
```

## 4. Use runtime parameter overrides
```
custom_model = manager.load(
    "gpt4o",
    temperature=0.5,
    max_tokens=2048,
    json_response=True,
    streaming=False,
)
```

## 5. Get total amount loaded models
```
manager.active()
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

## 📘 Simple Example: Using `langfabric` with OpenAI

This example demonstrates how to load a model configuration from YAML and run an asynchronous prompt chain.

---

### 1. Create a `models.yaml` file

```yaml
- name: gpt4o-mini
  provider: openai
  model: o4-mini
  api_key: !env OPENAI_API_KEY
  max_tokens: 4096
  streaming: True
```

### 2. Export Your API Key
Before running the example, make sure your OpenAI API key is available as an environment variable:

```bash
export OPENAI_API_KEY=your_openai_key_here
```

### 3. Run the Example Script
```
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langfabric import load_model_configs, build_model

async def main():
    # Load model configuration from YAML
    cfg = load_model_configs(["models.yaml"])

    # Build the model instance using the configuration
    llm = build_model(cfg["gpt4o-mini"])

    # Define a structured prompt with system and user roles
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant answering questions about the region {region_name}. Provide short and clear answers.",
        ),
        ("human", "{input}"),
    ])

    # Create a prompt-model chain
    chain = prompt | llm

    # Execute the chain with input values
    output = await chain.ainvoke({
        "region_name": "Bay Area",
        "input": "How many people are living there?",
    })

    # Print the model output
    print(output.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### Example Output
```
The Bay Area has approximately 7.7 million people.
```
