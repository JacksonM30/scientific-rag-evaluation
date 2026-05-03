# Profiles are named experiment presets.
# Keep these stable so runs can be reproduced later.

PROFILES: dict[str, dict] = {
    "rag_qwen_generation_v1": {
        "provider": "qwen",
        "model": "qwen3-8b",
        "temperature": 0,
        "max_tokens": 512,
        "extra_body": {"enable_thinking": False},
    },
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
