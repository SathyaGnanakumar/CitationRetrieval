"""Utility helpers for constructing LangChain chat models."""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel


def get_chat_model(model_name: str, temperature: float = 0.0, timeout: Optional[float] = None) -> BaseChatModel:
    """Create a LangChain chat model based on the provided identifier.

    Args:
        model_name: Backend model identifier (e.g., ``gpt-4o`` or ``claude-3-sonnet``).
        temperature: Sampling temperature to use for stochastic models.
        timeout: Optional per-request timeout in seconds.

    Returns:
        LangChain ``BaseChatModel`` instance.

    Raises:
        ValueError: If the backend cannot be resolved based on the model name.
    """

    normalized = model_name.lower()

    if normalized.startswith("gpt"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature, timeout=timeout)

    if "claude" in normalized:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=temperature, timeout=timeout)

    if "llama" in normalized or "meta-llama" in normalized or "mixtral" in normalized:
        from langchain_together import ChatTogether

        return ChatTogether(model=model_name, temperature=temperature, timeout=timeout)

    raise ValueError(
        "Unsupported model_name for LangChain chat model. "
        "Please supply an OpenAI, Anthropic, or Together AI identifier."
    )
