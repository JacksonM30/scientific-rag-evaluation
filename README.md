# CS6493 RAG Experiment Harness

This repository contains the implementation code for the CS6493 Topic 4 RAG project.

Planning notes, assignment PDFs, and Codex workflow documents live one level up in `../docs/`.
Keep this repository focused on runnable experiment code, configs, tests, and inspectable outputs.
See `docs/ARCHITECTURE.md` for the current module map and data flow.

## Current Direction

- Start with a tiny HotpotQA RAG loop.
- Save intermediate artifacts so retrieval and generation failures can be inspected.
- Use standard libraries for standard RAG components where practical.
- Keep experiment configs, artifacts, and analysis under project control.
- Use `rag_experiment.model_clients` for generation profiles.
- Expand only after one end-to-end run is understandable.

## Dataset Policy

- Commit tiny synthetic fixtures such as `data/hotpotqa_mini/sample.jsonl`.
- Keep full dataset downloads under ignored paths like `data/raw/`.
- Record sample size, split, and source path in each run config/artifact.
- Start with 5-10 examples, then scale only after retrieval and artifact traces look correct.
- Use real dataset samples, not synthetic fixtures, for report-facing results.

## Environment

Use the local conda environment:

```bash
conda activate LLM
```

or run commands without activating the shell:

```bash
conda run -n LLM python -V
```

## Dependencies

LangChain is already used for model clients and retrieval adapters. The BM25
baseline uses `langchain_community.retrievers.BM25Retriever`, which requires the
optional `rank-bm25` package. Dense retrieval uses LangChain's in-memory vector
store plus DashScope embeddings through the OpenAI-compatible API. Hybrid
retrieval uses LangChain `EnsembleRetriever` over BM25 and dense retrievers.

Core RAG dependencies are recorded in `requirements.txt`. Install or refresh
them in the `LLM` environment with:

```bash
conda activate LLM
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

If retrieval reports that `rank-bm25` is missing, install it in the `LLM`
environment:

```bash
conda activate LLM
python -m pip install rank-bm25
```

Experiment code should not install dependencies at runtime. During development,
the assistant may try setup commands; if download/install fails, it should give
the exact command for manual installation.

## Retrieval Dry Runs

Run the BM25 no-API artifact pass from this repository root:

```bash
conda run -n LLM python -m rag_experiment.runners.dry_run configs/hotpotqa_bm25_dry_run.json
```

Dense and hybrid retrieval use `DASHSCOPE_API_KEY` for embeddings:

```bash
conda run -n LLM python -m rag_experiment.runners.dry_run configs/hotpotqa_dense_dry_run.json
conda run -n LLM python -m rag_experiment.runners.dry_run configs/hotpotqa_hybrid_dry_run.json
```

This writes JSONL artifacts under `outputs/`, which is ignored by Git. The dry
run validates dataset loading, retrieval, prompt rendering, and artifact shape
before any answer-generation model call.

## Generation Smoke Run

The first real generation config uses HotpotQA mini, BM25, `top_k=3`, and the
`rag_qwen_generation_v1` model profile:

```bash
conda run -n LLM python -m rag_experiment.runners.run_generation configs/hotpotqa_bm25_generation.json
```

This calls DashScope through the OpenAI-compatible API, so `DASHSCOPE_API_KEY`
must be set. Each output row includes the question, gold answer, retrieved
passages, rendered messages, raw model answer, parsed answer, cited passage IDs,
and any per-example error.

## Generation Artifact Inspection

Inspect a saved generation artifact without making new API calls:

```bash
conda run -n LLM python -m rag_experiment.analysis.inspect_generation outputs/hotpotqa_mini_bm25_generation/results.jsonl
```

Use `--limit N` to print only the first N examples, `--show-prompt` to include
rendered messages, or `--summary-json PATH` to write aggregate counts. The raw
`results.jsonl` remains the per-example source of truth.

## Normalized Metric Fixtures

The first formal evaluator supports only normalized artifact schema `v0.1` for
PubMedQA and SciFact. It does not evaluate the older HotpotQA smoke-test
artifacts.

```bash
conda run -n LLM python -m rag_experiment.evaluation.evaluate_artifact data/evaluation_fixtures/pubmedqa_v01.jsonl --dataset pubmedqa
conda run -n LLM python -m rag_experiment.evaluation.evaluate_artifact data/evaluation_fixtures/scifact_v01.jsonl --dataset scifact
```

Build tiny normalized artifacts from real PubMedQA/SciFact rows for learning:

```bash
conda run -n LLM python -m rag_experiment.data.build_normalized_samples pubmedqa --limit 5
conda run -n LLM python -m rag_experiment.data.build_normalized_samples scifact --limit 5
```

These use oracle context/evidence and gold-label demo predictions. They are for
understanding schema and metrics, not for report-facing retrieval claims.

Run pooled-corpus PubMedQA retrieval artifacts:

```bash
conda run -n LLM python -m rag_experiment.runners.run_pubmedqa_retrieval --retriever bm25 --limit 20 --top-k 5
conda run -n LLM python -m rag_experiment.runners.run_pubmedqa_retrieval --retriever dense --limit 5 --top-k 5
conda run -n LLM python -m rag_experiment.runners.run_pubmedqa_retrieval --retriever hybrid --limit 5 --top-k 5
conda run -n LLM python -m rag_experiment.evaluation.evaluate_artifact outputs/retrieval/pubmedqa_bm25_pooled_v01.jsonl --dataset pubmedqa
```

Run pooled-corpus SciFact retrieval artifacts:

```bash
conda run -n LLM python -m rag_experiment.runners.run_scifact_retrieval --retriever bm25 --limit 20 --top-k 5
conda run -n LLM python -m rag_experiment.runners.run_scifact_retrieval --retriever dense --limit 5 --top-k 5 --corpus-doc-limit 50
conda run -n LLM python -m rag_experiment.runners.run_scifact_retrieval --retriever hybrid --limit 5 --top-k 5 --corpus-doc-limit 50
conda run -n LLM python -m rag_experiment.evaluation.evaluate_artifact outputs/retrieval/scifact_bm25_pooled_v01.jsonl --dataset scifact
```

These runs search each query over a shared passage pool, including distractors.
The prediction answer is still the gold-label demo value, so these artifacts are
retrieval-focused rather than generation-focused. Dense and hybrid runs call the
embedding API; keep limits small while developing.
