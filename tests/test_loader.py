import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langfabric.loader import load_model_configs, load_models
from langfabric.manager import ModelManager

def test_load_model_config():
    cfg = load_model_configs(["tests/models.yaml"], secrets={"gpt35_key": "sk-..."})
    assert cfg["gpt35"].provider == "azure_openai"
    assert cfg["gpt35"].deployment_name == "best_model"
    assert cfg["gpt35"].embeddings_deployment_name == "best_model_embed"
    manager = ModelManager(cfg)
    llm = manager.get("gpt35")
    assert llm != None

def test_load_model_config():
    llms = load_models("tests/models.yaml", secrets={"gpt35_key": "sk-..."})
    assert llms["gpt35"] != None
