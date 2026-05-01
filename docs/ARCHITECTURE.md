# Architecture Map

This project is a small experiment harness, not a RAG platform. The structure is
meant to keep the experiment understandable while leaving room for dense
retrieval, generation, evaluation, and analysis later.

## Current Structure

```text
rag_project/
  configs/                 # Run settings: dataset, retriever, top_k, prompt, output
  data/                    # Tiny fixtures plus ignored raw/processed datasets
  src/rag_experiment/
    data/                  # Dataset schemas and loaders
    retrieval/             # Retriever adapters, starting with LangChain BM25
    generation/            # Prompt definitions now; model generation later
    runners/               # Executable experiment flows
  outputs/                 # Ignored run artifacts
```

Planning notes and assignment documents live outside this repo in `../docs/`.

## Current Data Flow

```text
config
  -> load HotpotQA-style examples
  -> flatten each example context into passages
  -> retrieve top_k passages with BM25
  -> format retrieved context
  -> render reusable prompt messages
  -> write one JSONL artifact per example
```

The current runner is dry-run only, so it stops before calling an LLM. It writes
`raw_model_answer: null` and `parsed_answer: null`.

## Module Responsibilities

- `data.hotpotqa`: owns `HotpotExample`, `Passage`, and JSONL loading.
- `retrieval.bm25`: wraps `langchain_community` BM25 and returns project-owned
  retrieval records.
- `generation.prompts`: owns named prompt definitions and context formatting.
- `runners.dry_run`: connects config, data loading, retrieval, prompt rendering,
  and artifact writing.

## What Comes Later

- `generation`: add model-client calls using the existing LLM profiles.
- `evaluation`: add retrieval, answer, citation, and grounding metrics once real
  generation artifacts exist.
- `analysis`: aggregate artifact JSONL into tables and failure patterns.

Do not create these modules before they have real behavior. Empty folders make
the project look organized but harder to understand.

## Artifact Policy

Commit code, configs, and tiny fixtures. Do not commit `outputs/`, full dataset
downloads, indexes, or caches. Every run artifact should be traceable from:

```text
question -> retrieved passages -> rendered prompt -> answer/evaluation
```
