# Model Clients 配置指南

本项目将模型设置拆分为三个微型文件，以实现模块化管理：
•	providers.py：供应商注册表。负责记录如何连接各个模型厂商。
•	profiles.py：实验预设（Profile）。记录特定实验所使用的供应商、模型及参数配置。
•	factory.py：工厂函数。负责将供应商和预设配置转化为可用的 LangChain 聊天模型实例。

## 核心理念

当你需要直接选择供应商时，使用 get_llm()：

```python
from rag_experiment.model_clients import get_llm

model = get_llm("qwen", temperature=0, max_tokens=1024)
```

当你需要进行可复现的实验profile时，使用 get_llm_profile()：

```python
from rag_experiment.model_clients import get_llm_profile

model = get_llm_profile("attack_weak_model")
```

你可以在不修改 profiles.py 的情况下，临时覆盖预设参数：

```python
model = get_llm_profile("attack_weak_model", temperature=0.3)
```

## 文件协作机制

1. providers.py
定义技术层面的连接细节：
•	cls: LangChain 聊天模型类（例如 ChatOpenAI）。
•	api_key_env: 存储 API 密钥的环境变量名称。
•	base_url: 自定义的 OpenAI 兼容端点（如果需要）。
•	default_model: 当调用者未指定 model 时使用的默认模型。
2. profiles.py
定义实验层面的选择：
•	provider: 必须与 PROVIDERS 中的键匹配。
•	model: 可选。如果省略，则使用供应商的默认模型。
•	其他字段（如 temperature 和 max_tokens）将直接传递给模型构造函数。
3. factory.py
负责验证供应商/预设名称、从环境中读取 API 密钥、合并配置项，并返回初始化后的 LangChain 模型。

# 操作指南

## 添加新供应商

在 PROVIDERS 中添加新条目：

```python
"new_provider": {
    "cls": ChatOpenAI,
    "api_key_env": "NEW_PROVIDER_API_KEY",
    "base_url": "https://example.com/compatible-mode/v1",
    "default_model": "new-model-name",
}
```

在运行 Notebook 之前设置环境变量：

```bash
export NEW_PROVIDER_API_KEY="..."
```

## 添加新实验Profile

在 PROFILES 中添加新条目：

```python
"rag_eval_qwen": {
    "provider": "qwen",
    "model": "qwen3-8b",
    "temperature": 0,
    "max_tokens": 1024,
}
```
提示：请使用能够描述实验目的的名称作为 Profile 键名，而不仅仅是模型名称。

# 团队规范
•	禁止在 Notebook 或 Python 文件中硬编码 API 密钥。
•	连接细节统一存放在 providers.py。
•	实验预设统一存放在 profiles.py。
•	对于可重复的实验，优先使用 get_llm_profile()。
•	对于快速的单次探索，可以使用 get_llm()。
•	如果预设中引用了某个供应商，该供应商必须存在于 PROVIDERS 中。

embeddings
