from langchain_openai import ChatOpenAI

DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"
DASHSCOPE_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# Chat provider registry.
# Each entry defines how to construct a LangChain chat model for that vendor.
PROVIDERS: dict[str, dict] = {
    "qwen": {
        "cls": ChatOpenAI,
        "api_key_env": DASHSCOPE_API_KEY_ENV,
        "base_url": DASHSCOPE_OPENAI_BASE_URL,
        "default_model": "qwen3-8b",
    },
    "deepseek": {
        "cls": ChatOpenAI,
        "api_key_env": DASHSCOPE_API_KEY_ENV,
        "base_url": DASHSCOPE_OPENAI_BASE_URL,
        "default_model": "deepseek-v3",
    },
    "glm": {
        "cls": ChatOpenAI,
        "api_key_env": DASHSCOPE_API_KEY_ENV,
        "base_url": DASHSCOPE_OPENAI_BASE_URL,
        "default_model": "glm-4.5",
    },
}


# Embedding provider registry.
# Keep separate from chat providers because embeddings expose a different API.
EMBEDDING_PROVIDERS: dict[str, dict] = {
    "dashscope": {
        "api_key_env": DASHSCOPE_API_KEY_ENV,
        "base_url": DASHSCOPE_OPENAI_BASE_URL,
        "default_model": "text-embedding-v2",
    },
}
