from pathlib import Path
from typing import Optional, Dict, Iterator, Any
from pydantic import TypeAdapter
from seyaml import load_seyaml
import threading

from .schema import ModelConfig
from .fabric import build_model

class ModelManager:
    def __init__(self, model_configs: Dict[str, ModelConfig]):
        self.model_configs = model_configs
        self._cache = {}
        self._lock = threading.Lock()

    def get(self, model_name: str, *,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            max_retries: Optional[int] = None,
            json_response: bool = False,
            streaming: Optional[bool] = None) -> Any:
        key = (
            model_name,
            temperature,
            max_tokens,
            max_retries,
            json_response,
            streaming,
        )
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        config = self.model_configs.get(model_name)
        if not config:
            raise ValueError(f"Model config '{model_name}' not found")

        from .fabric import build_model    
        model = build_model(
            config,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            json_response=json_response,
            streaming=streaming,
        )

        with self._lock:
            self._cache[key] = model

        return model

    def has(self, model_name: str) -> bool:
        with self._lock:
            return model_name in self.model_configs

    def preload(self) -> None:
        """Force build and cache of all model configs using multiple threads."""
        threads = []
        for model_name in self.model_configs:
            t = threading.Thread(target=self.get, args=(model_name,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def keys(self):
        return self.model_configs.keys()

    def values(self) -> Iterator[Any]:
        for model_name in self.keys():
            yield self.get(model_name)

    def items(self) -> Iterator[tuple[str, Any]]:
        for model_name in self.keys():
            yield (model_name, self.get(model_name))
