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
    corpus/                # Dataset-specific pooled corpus builders
    retrieval/             # BM25, dense, and hybrid retriever adapters
    model_clients/         # Chat model and embedding factories
    generation/            # Prompt definitions and response parsing
    runners/               # Executable experiment flows
  outputs/                 # Ignored run artifacts
```

Planning notes and assignment documents live outside this repo in `../docs/`.

## Current Data Flow

```text
config
  -> load HotpotQA-style examples
  -> flatten each example context into passages
  -> retrieve top_k passages with the configured retriever
  -> format retrieved context
  -> render reusable prompt messages
  -> optionally call the configured generation model
  -> parse answer JSON when generation is enabled
  -> write one JSONL artifact per example
```

For report-facing PubMedQA/SciFact retrieval, the flow is:

```text
dataset rows
  -> select query rows
  -> build a pooled retrieval corpus from separate corpus limits
  -> retrieve top_k passages with BM25, dense, or hybrid retrieval
  -> write normalized v0.1 JSONL artifacts
  -> evaluate with dataset-aware metrics
```

There are two runner modes:

- `runners.dry_run`: retrieval and prompt preview only. It writes
  `raw_model_answer: null`, `parsed_answer: null`, and `cited_passage_ids: []`.
- `runners.run_generation`: calls a configured chat model profile and expects
  strict JSON with `answer` and `cited_passage_ids`.

## Module Responsibilities

- `data.hotpotqa`: owns `HotpotExample`, `Passage`, and JSONL loading.
- `corpus`: converts PubMedQA and SciFact source rows into query sets, gold
  evidence keys, and pooled passage corpora without leaking answer fields into
  retrievable text.
- `retrieval`: wraps library retrievers behind project-owned retrieval records.
  Current methods are BM25, dense vector retrieval over DashScope embeddings,
  and hybrid fusion through LangChain `EnsembleRetriever`.
- `model_clients.embeddings`: creates DashScope embeddings through the
  OpenAI-compatible API.
- `generation.prompts`: owns named prompt definitions and context formatting.
- `generation.parsing`: parses strict JSON answer artifacts.
- `runners.dry_run`: connects config, data loading, retrieval, prompt rendering,
  and artifact writing.
- `runners.run_generation`: adds model invocation and answer parsing to the same
  artifact path.
- `runners.run_pubmedqa_retrieval` and `runners.run_scifact_retrieval`: build
  normalized pooled retrieval artifacts for formal metric evaluation.

## What Comes Later

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
