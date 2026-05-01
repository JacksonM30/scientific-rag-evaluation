# Profiles are named experiment presets.
# Keep these stable so runs can be reproduced later.

PROFILES: dict[str, dict] = {
    "attack_weak_model": {
        "provider": "deepseek",
        "temperature": 0,
        "max_tokens": 512,
    },
    "attack_strong_model": {
        "provider": "qwen",
        "model": "qwen3.6-plus",
        "temperature": 0,
        "max_tokens": 2048,
    },
}
