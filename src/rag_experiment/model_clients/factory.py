import os
from typing import Any, Optional

from rag_experiment.model_clients.profiles import PROFILES
from rag_experiment.model_clients.providers import PROVIDERS


def get_llm(provider: str, model: Optional[str] = None, **kwargs: Any):
    """Create a chat model directly from a provider name."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: '{provider}'. Available: {list(PROVIDERS)}")

    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        raise EnvironmentError(f"Missing environment variable: {cfg['api_key_env']}")

    init_kwargs: dict[str, Any] = {
        "model": model or cfg["default_model"],
        "api_key": api_key,
        **kwargs,
    }
    if "base_url" in cfg:
        init_kwargs["base_url"] = cfg["base_url"]

    return cfg["cls"](**init_kwargs)


def get_llm_profile(profile_name: str, **overrides: Any):
    """
    Create a chat model from a named experiment profile.

    Standard usage:
    model = get_llm_profile("attack_weak_model")

    Temporary override without editing the saved profile:
    model = get_llm_profile("attack_strong_model", temperature=0.3)
    """
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile: '{profile_name}'. Available: {list(PROFILES)}")

    cfg = PROFILES[profile_name].copy()
    provider = cfg.pop("provider")

    # Overrides are local to this call; the profile definition stays unchanged.
    cfg.update(overrides)

    return get_llm(provider, **cfg)
