from langchain_openai import ChatOpenAI

# Provider registry.
# Each entry defines how to construct a LangChain chat model for that vendor.
PROVIDERS: dict[str, dict] = {
    "qwen": {
        "cls": ChatOpenAI,
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen3-8b",
    },
    "deepseek": {
        "cls": ChatOpenAI,
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "deepseek-v3",
    },
    "glm": {
        "cls": ChatOpenAI,
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "glm-4.5",
    },
}
