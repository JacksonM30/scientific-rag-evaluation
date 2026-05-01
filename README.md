# CS6493 RAG Experiment Harness

This repository contains the implementation code for the CS6493 Topic 4 RAG project.

Planning notes, assignment PDFs, and Codex workflow documents live one level up in `../docs/`.
Keep this repository focused on runnable experiment code, configs, tests, and inspectable outputs.
See `docs/ARCHITECTURE.md` for the current module map and data flow.

## Current Direction

- Start with a tiny HotpotQA BM25 loop.
- Save intermediate artifacts so retrieval and generation failures can be inspected.
- Use standard libraries for standard RAG components where practical.
- Keep experiment configs, artifacts, and analysis under project control.
- Use `rag_experiment.model_clients` for generation profiles.
- Expand only after one end-to-end run works.

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
optional `rank-bm25` package.

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

## First Dry Run

Run the first no-API artifact pass from this repository root:

```bash
conda run -n LLM python -m rag_experiment.runners.dry_run configs/hotpotqa_bm25_dry_run.json
```

This writes JSONL artifacts under `outputs/`, which is ignored by Git. The dry
run validates dataset loading, LangChain BM25 retrieval, prompt rendering, and
artifact shape before any model API call.
