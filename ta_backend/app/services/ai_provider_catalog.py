from dataclasses import dataclass


DEFAULT_UNSUPPORTED_SCAN_REASON = "Not currently supported for image-based medical document scanning."


@dataclass(frozen=True)
class AIProviderCatalogEntry:
    provider: str
    package_name: str
    class_name: str
    credential_env_var: str
    settings_attr: str
    supports_image_input: bool
    is_scan_compatible: bool
    activation_block_reason: str | None = None


AI_PROVIDER_CATALOG: dict[str, AIProviderCatalogEntry] = {
    "google_genai": AIProviderCatalogEntry(
        provider="google_genai",
        package_name="langchain_google_genai",
        class_name="ChatGoogleGenerativeAI",
        credential_env_var="GOOGLE_API_KEY",
        settings_attr="google_api_key",
        supports_image_input=True,
        is_scan_compatible=True,
    ),
    "openrouter": AIProviderCatalogEntry(
        provider="openrouter",
        package_name="langchain_openrouter",
        class_name="ChatOpenRouter",
        credential_env_var="OPENROUTER_API_KEY",
        settings_attr="openrouter_api_key",
        supports_image_input=True,
        is_scan_compatible=True,
    ),
    "ollama": AIProviderCatalogEntry(
        provider="ollama",
        package_name="langchain_ollama",
        class_name="ChatOllama",
        credential_env_var="OLLAMA_API_KEY",
        settings_attr="ollama_api_key",
        supports_image_input=True,
        is_scan_compatible=True,
    ),
    "groq": AIProviderCatalogEntry(
        provider="groq",
        package_name="langchain_groq",
        class_name="ChatGroq",
        credential_env_var="GROQ_API_KEY",
        settings_attr="groq_api_key",
        supports_image_input=False,
        is_scan_compatible=False,
        activation_block_reason=DEFAULT_UNSUPPORTED_SCAN_REASON,
    ),
    "mistralai": AIProviderCatalogEntry(
        provider="mistralai",
        package_name="langchain_mistralai",
        class_name="ChatMistralAI",
        credential_env_var="MISTRAL_API_KEY",
        settings_attr="mistral_api_key",
        supports_image_input=False,
        is_scan_compatible=False,
        activation_block_reason=DEFAULT_UNSUPPORTED_SCAN_REASON,
    ),
    "deepseek": AIProviderCatalogEntry(
        provider="deepseek",
        package_name="langchain_deepseek",
        class_name="ChatDeepSeek",
        credential_env_var="DEEPSEEK_API_KEY",
        settings_attr="deepseek_api_key",
        supports_image_input=False,
        is_scan_compatible=False,
        activation_block_reason=DEFAULT_UNSUPPORTED_SCAN_REASON,
    ),
    "cohere": AIProviderCatalogEntry(
        provider="cohere",
        package_name="langchain_cohere",
        class_name="ChatCohere",
        credential_env_var="COHERE_API_KEY",
        settings_attr="cohere_api_key",
        supports_image_input=False,
        is_scan_compatible=False,
        activation_block_reason=DEFAULT_UNSUPPORTED_SCAN_REASON,
    ),
    "cerebras": AIProviderCatalogEntry(
        provider="cerebras",
        package_name="langchain_cerebras",
        class_name="ChatCerebras",
        credential_env_var="CEREBRAS_API_KEY",
        settings_attr="cerebras_api_key",
        supports_image_input=False,
        is_scan_compatible=False,
        activation_block_reason=DEFAULT_UNSUPPORTED_SCAN_REASON,
    ),
}


def get_ai_provider_catalog_entry(provider: str) -> AIProviderCatalogEntry | None:
    return AI_PROVIDER_CATALOG.get(provider)


def is_provider_scan_compatible(provider: str) -> bool:
    entry = get_ai_provider_catalog_entry(provider)
    return bool(entry and entry.is_scan_compatible)


def get_provider_activation_block_reason(provider: str) -> str | None:
    entry = get_ai_provider_catalog_entry(provider)
    if entry is None:
        return "Provider is not registered."
    return entry.activation_block_reason
