import os
import anthropic
import voyageai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PROVIDERS = {
    "anthropic": {
        "type": "anthropic",
        "enrich_model": "claude-haiku-4-5-20251001",
        "model_prefix": "claude",
    },
    "openai": {
        "type": "openai_compat",
        "enrich_model": "gpt-4o-mini",
        "model_prefix": "gpt",
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "type": "openai_compat",
        "enrich_model": "gemini-2.0-flash-lite",
        "model_prefix": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
    },
    "deepseek": {
        "type": "openai_compat",
        "enrich_model": "deepseek-chat",
        "model_prefix": "deepseek",
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "type": "openai_compat",
        "enrich_model": "qwen2.5-7b-instruct",
        "model_prefix": "qwen",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_env": "QWEN_API_KEY",
    },
}

_anthropic_client = anthropic.Anthropic()
voyage_client = voyageai.Client()
_openai_clients: dict[str, OpenAI] = {}


def get_client(provider: str) -> anthropic.Anthropic | OpenAI:
    if provider not in PROVIDERS:
        raise ValueError(f"Provider inválido: {provider}. Opções: {list(PROVIDERS.keys())}")
    config = PROVIDERS[provider]
    if config["type"] == "anthropic":
        return _anthropic_client
    if provider not in _openai_clients:
        _openai_clients[provider] = OpenAI(
            api_key=os.environ.get(config["key_env"]),
            base_url=config["base_url"],
        )
    return _openai_clients[provider]


def get_provider_for_model(model: str) -> str:
    for provider, config in PROVIDERS.items():
        if model.startswith(config["model_prefix"]):
            return provider
    raise ValueError(f"Provider desconhecido para o modelo: {model}")
