# RAG Project Roadmap

This file tracks project direction, near-term todos, deferred ideas, and durable decisions. Keep detailed learning notes in `docs/First_RAG.md` and current system explanations in `rag_project/docs/ARCHITECTURE.md`.

## Current Focus

- Turn PubMedQA and SciFact n=100 artifacts into report-ready retriever, model,
  and domain comparison evidence.
- Keep report-facing experiment records under `experiments/`; raw JSONL, CSV,
  and metric JSON artifacts stay under ignored `outputs/`.
- PubMedQA and SciFact pooled runners exist with BM25, dense, and hybrid
  options; dense/hybrid now default to cached DashScope `text-embedding-v4`
  vectors.
- Use dense v4 retrieval artifacts as the default generation inputs because
  dense v4 is the strongest current retriever.
- Use answer plus cited passage IDs and `evidence_summary` as the report-facing
  model output shape for generation comparisons.
- Keep the code library-first where possible, especially for standard RAG components.
- Use notebooks as learning surfaces, not as the main experiment engine.
- Treat HotpotQA mini as a debug/sanity dataset, not the current report-facing
  metric path.

## Near-Term Todos

- Use `experiments/001_report_ready_retriever_model_domain.md` as the source of
  truth for report-facing metric tables and artifact paths.
- Inspect representative success/failure examples from the report model outputs
  before writing final claims, especially PubMedQA wrong answers with valid
  gold-overlapping citations.
- Convert the experiment record into report figures/tables and a concise
  narrative: dense beats BM25, instruct-2507 is the stronger generator, and
  PubMedQA/SciFact differ by domain and label structure.

## Later Todos

- Add small tests for citation metric edge cases if the evaluator keeps growing.
- Compare a second generation model after the single-model path is clear.
- Use `experiment-analyzer` on saved artifacts before writing final report prose.

## Deferred

- Hybrid v4 underperforms dense v4 in the current n=100 pooled runs. The current
  hybrid retriever is equal-weight BM25+dense RRF, so lexical BM25 candidates can
  displace semantically stronger dense hits in top-k. Before making hybrid a
  report-facing improvement claim, inspect misses and tune `weights`,
  `candidate_k`, `rrf_k`, or try dense-first reranking.
- HotpotQA dense/hybrid dry-run configs still reference `text-embedding-v2`.
  Current embedding defaults are v4-oriented and inject `dimensions=1024`, so
  either update those configs to v4 or make dimensions model-aware before using
  them again.
- Persistent vector database. Use local JSON vector-store caches for current
  pooled samples, then revisit FAISS, Chroma, LanceDB, or Qdrant if scale makes
  JSON caches too slow or too large.
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
- Keep retrieval in memory, but save local embedding vector-store caches under
  ignored `outputs/embedding_cache/` so dense/hybrid reruns do not repeatedly
  embed the same passage pool.
- Use DashScope OpenAI-compatible embeddings first:
  - provider: `dashscope`
  - env var: `DASHSCOPE_API_KEY`
  - base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
  - default embedding model: `text-embedding-v4`
  - default dimensions: `1024`
- Keep assistant-driven automation interactive around expensive or scope-changing steps.

## Open Questions

- Do PubMedQA wrong-answer cases mostly come from prompt conservatism, label
  ambiguity, or retrieved contexts that do not fully support the gold answer?
- Why does citation-first v2 sometimes cite fewer or less gold-overlapping
  passages even when answer accuracy improves slightly on selected cases?
- For PubMedQA, does the v3 `evidence_summary` show model misreading, weak gold
  evidence, or a prompt policy problem?
- Does the selected-case PubMedQA gain from `qwen3.5-flash` hold on n=100, or is
  it specific to the failure-heavy selected subset?
- For SciFact, are the thinking-mode citation gains useful enough to justify the
  label-accuracy drop, or should we keep `qwen3-8b` v3-debug as the stronger
  citation-balanced path?
- If scale requires persistent vector storage, should we use FAISS, Chroma, LanceDB, or Qdrant?
