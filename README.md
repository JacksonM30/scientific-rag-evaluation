# CS6493 RAG Experiment Harness

This repository contains a reproducible experiment harness for the CS6493 Topic
4 project: Retrieval-Augmented Generation for knowledge-intensive scientific and
biomedical tasks.

The report-facing work compares BM25, dense, and hybrid retrieval on PubMedQA
and SciFact, then evaluates generated answers and citation grounding under fixed
retrieval/model settings. Raw datasets, model outputs, embedding caches, and
large generated artifacts are intentionally ignored by Git; tracked experiment
records under `experiments/` summarize the report evidence.

## Current Results

Tracked experiment records:

- `experiments/001_report_ready_retriever_model_domain.md`: BM25/hybrid/dense
  retrieval comparison, dense-only model comparison, and PubMedQA/SciFact domain
  comparison.
- `experiments/002_retriever_influence_on_generation.md`: BM25 vs dense
  generation impact under the fixed stronger report model.

Main n=100 findings:

- Dense v4 retrieval is stronger than BM25 on both datasets.
- `qwen3-30b-a3b-instruct-2507` is the stronger report generator among the two
  tested Qwen 30B profiles.
- Holding the generator fixed, dense retrieval improves both answer accuracy and
  citation-gold grounding over BM25.

## Repository Layout

- `src/rag_experiment/`: package code for datasets, retrieval, generation,
  evaluation, model clients, and analysis utilities.
- `configs/`: small HotpotQA smoke-run configs.
- `data/`: tiny committed fixtures plus ignored raw/cache directories.
- `docs/`: architecture, dataset, evaluation, and model-client notes.
- `experiments/`: tracked report-facing experiment records.
- `notebooks/`: learning notebooks for understanding retrieval, metrics, and
  generated citations.
- `tests/`: focused unit tests for parsing, generation, evaluation, and
  comparison helpers.

## Setup

The project was developed with Python 3.11 in a local conda environment named
`LLM`.

```bash
conda activate LLM
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

Commands can also be run without activating the shell:

```bash
conda run -n LLM env PYTHONPATH=src python -m unittest discover -s tests
```

## API Keys

Dense retrieval, hybrid retrieval, and generation profiles use DashScope through
OpenAI-compatible LangChain clients. Set the API key before running API-backed
commands:

```bash
export DASHSCOPE_API_KEY="..."
```

Do not commit `.env` files, raw keys, generated outputs, or embedding caches.

## Common Commands

Run tests:

```bash
conda run -n LLM env PYTHONPATH=src python -m unittest discover -s tests
conda run -n LLM env PYTHONPATH=src python -B -m compileall -q src tests
```

Evaluate a normalized artifact:

```bash
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.evaluation.evaluate_artifact outputs/retrieval/pubmedqa_bm25_pooled_v01.jsonl --dataset pubmedqa
```

Run pooled PubMedQA retrieval:

```bash
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.runners.run_pubmedqa_retrieval --retriever bm25 --limit 20 --corpus-limit 100 --top-k 5
```

Run pooled SciFact retrieval:

```bash
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.runners.run_scifact_retrieval --retriever bm25 --limit 20 --corpus-doc-limit 300 --top-k 5
```

Generate answers from an existing pooled retrieval artifact:

```bash
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.runners.run_pooled_generation --dataset pubmedqa --input outputs/retrieval/pubmedqa_dense_v4_pooled_n100_fullcorpus_v01.jsonl --output outputs/generation/pubmedqa_generation.jsonl --prompt-id pubmedqa_rag_json_v3_debug --model-profile rag_qwen3_30b_a3b_instruct_2507_v3_report --limit 3
```

Compare two generated artifacts by example ID:

```bash
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.analysis.generation_ab compare-models --dataset pubmedqa --baseline outputs/generation/baseline.jsonl --candidate outputs/generation/candidate.jsonl --baseline-label baseline --candidate-label candidate --output-csv outputs/analysis/generation_ab/comparison.csv --output-json outputs/analysis/generation_ab/comparison_summary.json
```

## Data and Artifact Policy

- Commit only tiny fixtures and human-readable experiment records.
- Keep full dataset downloads under `data/raw/`.
- Keep generated outputs, embedding caches, vector indexes, logs, and report
  exports under ignored local paths such as `outputs/`.
- Use `experiments/*.md` as the tracked source for report-ready metric tables and
  artifact paths.

## Verification Status

Latest local verification before GitHub publication:

```bash
conda run -n LLM env PYTHONPATH=src python -m unittest discover -s tests
conda run -n LLM env PYTHONPATH=src python -B -m compileall -q src tests
git diff --check
```
