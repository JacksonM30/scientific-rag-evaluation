# Experiment Records

This directory keeps tracked, report-facing experiment notes. Raw JSONL, CSV,
and metric JSON artifacts stay under ignored `outputs/` so reruns can be large
without making the Git history noisy.

## Active Experiments

- `001_report_ready_retriever_model_domain.md`: BM25 vs dense retrieval,
  dense-only model comparison, and PubMedQA/SciFact domain comparison.
- `002_retriever_influence_on_generation.md`: BM25 vs dense generation impact
  under the fixed stronger report model, focused on answer quality and citation
  grounding.
