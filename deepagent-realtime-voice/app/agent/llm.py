"""Azure OpenAI chat model, built by the org's shared ``llm_service.py``.

Loaded by file path (``LLM_SERVICE_PATH``) rather than via ``sys.path`` so the
FleetAgent2 project's other modules can't shadow ours. Its ``build_azure_llm``
authenticates with the compute's managed identity and reads
AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT / AZURE_OPENAI_API_VERSION /
AZURE_OAI_CLIENT_ID from the environment.
"""

import importlib.util
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from app import config


def build_llm() -> BaseChatModel:
    path = Path(config.LLM_SERVICE_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"LLM_SERVICE_PATH does not exist: {path}")
    spec = importlib.util.spec_from_file_location("org_llm_service", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_azure_llm()
