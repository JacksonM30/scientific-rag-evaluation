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
    "rag_qwen35_flash_v3_debug": {
        "provider": "qwen",
        "model": "qwen3.5-flash",
        "temperature": 0,
        "max_tokens": 1024,
        "extra_body": {"enable_thinking": False},
    },
    "rag_qwen35_flash_thinking_v3_debug": {
        "provider": "qwen",
        "model": "qwen3.5-flash",
        "temperature": 0,
        "max_tokens": 1024,
        "extra_body": {"enable_thinking": True},
    },
    "rag_qwen3_30b_a3b_v3_report": {
        "provider": "qwen",
        "model": "qwen3-30b-a3b",
        "temperature": 0,
        "max_tokens": 1024,
        "extra_body": {"enable_thinking": False},
    },
    "rag_qwen3_30b_a3b_instruct_2507_v3_report": {
        "provider": "qwen",
        "model": "qwen3-30b-a3b-instruct-2507",
        "temperature": 0,
        "max_tokens": 1024,
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
