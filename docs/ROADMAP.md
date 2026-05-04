# RAG Project Roadmap

This file tracks project direction, near-term todos, deferred ideas, and durable decisions. Keep detailed learning notes in `docs/First_RAG.md` and current system explanations in `rag_project/docs/ARCHITECTURE.md`.

## Current Focus

- Build and inspect real pooled-corpus retrieval artifacts for PubMedQA and
  SciFact.
- PubMedQA and SciFact pooled runners exist with BM25, dense, and hybrid
  options; inspect misses before scaling runs.
- Use artifact inspection to identify retrieval misses and evidence coverage
  before making report-facing claims.
- Use answer plus cited passage IDs as the default model output shape.
- Keep the code library-first where possible, especially for standard RAG components.
- Use notebooks as learning surfaces, not as the main experiment engine.
- Treat HotpotQA mini as a debug/sanity dataset, not the current report-facing
  metric path.

## Near-Term Todos

- Inspect PubMedQA/SciFact pooled BM25 artifacts and record obvious retrieval
  misses or weak query/passage patterns.
- Compare small BM25, dense, and hybrid smoke runs before scaling dense/hybrid
  because they call the embedding API.
- Decide the first report-facing sample sizes for PubMedQA and SciFact.

## Later Todos

- Connect pooled retrieval artifacts to one generation model after retrieval
  quality is inspectable.
- Add answer evaluation for generated PubMedQA yes/no/maybe outputs.
- Add citation and grounding evaluation after generated answers include stable,
  inspectable evidence references.
- Compare a second generation model after the single-model path is clear.
- Use `experiment-analyzer` on saved artifacts before writing report claims.

## Deferred

- Persistent vector database or saved vector index. Use in-memory retrieval for
  current pooled samples, then revisit FAISS, Chroma, LanceDB, or Qdrant if
  scale makes rebuilds painful.
- FAISS indexing, unless dataset scale makes in-memory retrieval too limiting.
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
- PubMedQA and SciFact are the report-facing dataset direction. HotpotQA mini
  remains useful for local pipeline debugging.
- Keep real experiment runners under `runners/`; keep dataset inspection and
  loading helpers under `data/`.
- Oracle normalized samples are learning/debug artifacts only. Report-facing
  retrieval claims should come from pooled-corpus retrieval artifacts.
- Keep vector storage in memory for the first pooled experiments. Do not add a
  persistent vector database until scale requires it.
- Use DashScope OpenAI-compatible embeddings first:
  - provider: `dashscope`
  - env var: `DASHSCOPE_API_KEY`
  - base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
  - default embedding model: `text-embedding-v2`
- Keep assistant-driven automation interactive around expensive or scope-changing steps.

## Open Questions

- How large should the first report-facing PubMedQA/SciFact pooled runs be on
  the local MacBook?
- If scale requires persistent vector storage, should we use FAISS, Chroma, LanceDB, or Qdrant?
