"""Thin wrapper around litellm with optional Ollama routing."""

import logging
from typing import Optional

import litellm

from .config import SystemConfig

logger = logging.getLogger(__name__)

_OLLAMA_HINTS = ("sqlcoder", "qwen", "coder", "deepseek")


def call_llm(config: SystemConfig, prompt: str, model_override: Optional[str] = None) -> Optional[str]:
    model = model_override or config.model_name
    use_ollama = any(k in model.lower() for k in _OLLAMA_HINTS)
    try:
        if use_ollama:
            response = litellm.completion(
                model=model,
                custom_llm_provider="ollama",
                api_base=config.ollama_base_url,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
            )
        else:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=config.api_key,
                temperature=config.temperature,
            )
        return response.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM call failed (model=%s): %s", model, exc)
        raise
