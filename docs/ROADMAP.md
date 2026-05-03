# RAG Project Roadmap

This file tracks project direction, near-term todos, deferred ideas, and durable decisions. Keep detailed learning notes in `docs/First_RAG.md` and current system explanations in `rag_project/docs/ARCHITECTURE.md`.

## Current Focus

- Inspect the first real BM25 generation artifacts before expanding comparisons.
- Use artifact inspection to identify retrieval misses, unsupported answers, and citation behavior.
- Use answer plus cited passage IDs as the default model output shape.
- Keep the code library-first where possible, especially for standard RAG components.
- Use notebooks as learning surfaces, not as the main experiment engine.

## Near-Term Todos

- Manually inspect the 3-example BM25 generation artifact and record obvious failure patterns.
- Create or update a notebook that explains the full retrieval-to-generation path with a tiny run.
- After the BM25 generation loop is clear, run the same path with dense and hybrid retrieval.
- Add retrieval evaluation metrics such as hit@k, recall@k, and MRR.

## Later Todos

- Add answer evaluation after generation artifacts are stable.
- Add citation and grounding evaluation after answers include inspectable evidence.
- Add PubMedQA or another second dataset once the HotpotQA loop is stable.
- Compare a second generation model after the single-model path is clear.
- Use `experiment-analyzer` on saved artifacts before writing report claims.

## Deferred

- Persistent vector database or saved vector index. Use `InMemoryVectorStore` for mini experiments, then revisit FAISS, Chroma, or LanceDB after the first generation loop works.
- FAISS indexing, unless dataset scale makes `InMemoryVectorStore` too limiting.
- Larger datasets before small runs are easy to inspect.
- Self-RAG or more advanced retrieval/generation variants.
- Heavy report automation before experiment artifacts are trustworthy.

## Decisions

- Keep `rag_project/` as the implementation repo.
- Use config files to control datasets, retrievers, models, `top_k`, prompts, evaluators, and outputs.
- Prefer LangChain or other libraries for standard implementations such as BM25, dense retrieval plumbing, hybrid fusion, prompt utilities, and model adapters.
- Keep project wrappers around library components when needed for stable experiment artifacts.
- The first generation output format should include both the answer and cited passage IDs, because later grounding and citation evaluation need inspectable evidence.
- The first generation profile is `rag_qwen_generation_v1`, using DashScope/Qwen with thinking disabled for non-streaming calls.
- Keep vector storage in memory for the first mini experiments. Do not add a persistent vector database until the retrieval-to-generation loop is stable and scale requires it.
- Use DashScope OpenAI-compatible embeddings first:
  - provider: `dashscope`
  - env var: `DASHSCOPE_API_KEY`
  - base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
  - default embedding model: `text-embedding-v2`
- Keep assistant-driven automation interactive around expensive or scope-changing steps.

## Open Questions

- Should the strict JSON schema add more fields, such as `unsupported_reason` or `confidence`, after the first artifact review?
- How large should the first non-mini HotpotQA run be on the local MacBook?
- If scale requires persistent vector storage, should we use FAISS, Chroma, LanceDB, or Qdrant?
