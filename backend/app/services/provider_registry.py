"""Stable provider metadata and Agent capability requirements.

Provider model catalogs change frequently. This registry describes connection
families and requirements while leaving model-specific capabilities unknown
until the selected model is explicitly tested.
"""
from __future__ import annotations

from dataclasses import dataclass


CAPABILITIES = (
    "chat_completion",
    "tool_calling",
    "structured_output",
    "json_schema",
    "streaming",
)
CAPABILITY_STATES = frozenset({"supported", "unsupported", "unknown"})
SUPPORTED_CONFIGURATION = (
    "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty",
)
_COMMON_PROVIDER_CONFIGURATION = ("temperature", "max_tokens", "top_p")
_PROVIDER_CONFIGURATION = {
    "openai": SUPPORTED_CONFIGURATION,
    "anthropic": _COMMON_PROVIDER_CONFIGURATION,
    "google_gemini": _COMMON_PROVIDER_CONFIGURATION,
    "groq": _COMMON_PROVIDER_CONFIGURATION,
    "mistral": _COMMON_PROVIDER_CONFIGURATION,
    "deepseek": _COMMON_PROVIDER_CONFIGURATION,
    "openrouter": _COMMON_PROVIDER_CONFIGURATION,
    "ollama": _COMMON_PROVIDER_CONFIGURATION,
    "vllm": _COMMON_PROVIDER_CONFIGURATION,
    "lm_studio": _COMMON_PROVIDER_CONFIGURATION,
    "custom_openai_compatible": _COMMON_PROVIDER_CONFIGURATION,
}
_LITELLM_API_BASE = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "google_gemini": "https://generativelanguage.googleapis.com",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "deepseek": "https://api.deepseek.com/beta",
    "openrouter": "https://openrouter.ai/api/v1",
}
AGENT_ROLES = ("pipeline_agent", "report_agent")
ROLE_REQUIRED_CAPABILITIES = {
    "pipeline_agent": ("chat_completion", "tool_calling"),
    "report_agent": ("chat_completion", "structured_output", "json_schema"),
}


@dataclass(frozen=True)
class ProviderDefinition:
    provider_id: str
    display_name: str
    category: str
    endpoint_mode: str
    api_key_requirement: str
    endpoint_requirement: str
    model_identifier_format: str
    litellm_provider: str | None = None
    description: str = ""

    def catalog_entry(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "category": self.category,
            "endpoint_mode": self.endpoint_mode,
            "api_key_requirement": self.api_key_requirement,
            "endpoint_requirement": self.endpoint_requirement,
            "model_identifier_format": self.model_identifier_format,
            "supported_configuration": list(supported_configuration_for(self.provider_id)),
            "description": self.description,
            "capabilities": {name: "unknown" for name in CAPABILITIES},
            "capability_note": (
                "Capabilities are model-specific and remain unknown until an "
                "explicit connection and capability test succeeds."
            ),
            "agent_requirements": {
                role: list(required)
                for role, required in ROLE_REQUIRED_CAPABILITIES.items()
            },
        }


_DEFINITIONS = (
    ProviderDefinition(
        "openai", "OpenAI", "cloud", "litellm_managed", "required", "managed",
        "Provider model ID, entered without the LiteLLM provider prefix", "openai",
    ),
    ProviderDefinition(
        "anthropic", "Anthropic", "cloud", "litellm_managed", "required", "managed",
        "Provider model ID, entered without the LiteLLM provider prefix", "anthropic",
    ),
    ProviderDefinition(
        "google_gemini", "Google Gemini", "cloud", "litellm_managed", "required", "managed",
        "Provider model ID, entered without the LiteLLM provider prefix", "gemini",
    ),
    ProviderDefinition(
        "groq", "Groq", "cloud", "litellm_managed", "required", "managed",
        "Provider model ID, entered without the LiteLLM provider prefix", "groq",
    ),
    ProviderDefinition(
        "mistral", "Mistral", "cloud", "litellm_managed", "required", "managed",
        "Provider model ID, entered without the LiteLLM provider prefix", "mistral",
    ),
    ProviderDefinition(
        "deepseek", "DeepSeek", "cloud", "litellm_managed", "required", "managed",
        "Provider model ID, entered without the LiteLLM provider prefix", "deepseek",
    ),
    ProviderDefinition(
        "openrouter", "OpenRouter", "cloud", "litellm_managed", "required", "managed",
        "OpenRouter model ID, including its provider/model portion", "openrouter",
    ),
    ProviderDefinition(
        "ollama", "Ollama", "local", "local_openai_compatible", "optional", "required",
        "Installed Ollama model identifier",
        description="Uses the OpenAI-compatible local API on a loopback endpoint.",
    ),
    ProviderDefinition(
        "vllm", "vLLM", "self_hosted", "local_openai_compatible", "optional", "required",
        "Model identifier served by this vLLM endpoint",
        description="Uses the OpenAI-compatible local API on a loopback endpoint.",
    ),
    ProviderDefinition(
        "lm_studio", "LM Studio", "local", "local_openai_compatible", "optional", "required",
        "Model identifier served by LM Studio",
        description="Uses the OpenAI-compatible local API on a loopback endpoint.",
    ),
    ProviderDefinition(
        "custom_openai_compatible", "Custom OpenAI-compatible", "custom", "custom_openai_compatible", "optional", "required",
        "Exact model identifier accepted by the configured endpoint",
        description=(
            "Remote custom endpoints must use HTTPS. DNS is resolved and pinned "
            "for each explicit connection test; private and metadata destinations are rejected."
        ),
    ),
)

PROVIDER_REGISTRY = {definition.provider_id: definition for definition in _DEFINITIONS}


def supported_configuration_for(provider_id: str) -> tuple[str, ...]:
    return _PROVIDER_CONFIGURATION.get(provider_id.strip().casefold(), ())


def litellm_api_base_for(provider_id: str) -> str | None:
    """Return the managed provider's canonical endpoint, bypassing env overrides."""
    return _LITELLM_API_BASE.get(provider_id.strip().casefold())


def get_provider(provider_id: str) -> ProviderDefinition | None:
    return PROVIDER_REGISTRY.get(provider_id.strip().casefold())


def provider_catalog() -> list[dict]:
    return [definition.catalog_entry() for definition in _DEFINITIONS]


def resolve_litellm_model(provider_id: str, model_identifier: str) -> str:
    definition = get_provider(provider_id)
    if definition is None or definition.litellm_provider is None:
        raise ValueError("This provider does not use the LiteLLM managed endpoint.")
    prefix = f"{definition.litellm_provider}/"
    if model_identifier.startswith(prefix):
        return model_identifier
    other_provider_prefixes = {
        "openai/", "anthropic/", "gemini/", "groq/", "mistral/",
        "deepseek/", "openrouter/",
    }
    if any(model_identifier.startswith(item) for item in other_provider_prefixes):
        allowed_nested_prefix = (
            (provider_id == "groq" and model_identifier.startswith("openai/"))
            or (provider_id == "openrouter" and any(
                model_identifier.startswith(item)
                for item in other_provider_prefixes - {"openrouter/"}
            ))
        )
        if not allowed_nested_prefix:
            raise ValueError("The model identifier uses a different LiteLLM provider prefix.")
    return prefix + model_identifier
