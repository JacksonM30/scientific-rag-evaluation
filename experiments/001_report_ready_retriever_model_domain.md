# Experiment 001: Report-Ready Retriever, Model, and Domain Comparison

## Goal

Build the main report evidence chain:

1. Compare BM25 and dense retrieval on PubMedQA and SciFact.
2. Use dense retrieval for all generation experiments because it is stronger.
3. Compare `qwen3-30b-a3b` and `qwen3-30b-a3b-instruct-2507` under the same
   dense retriever, prompt family, sample size, and evaluator.
4. Compare PubMedQA and SciFact behavior using answer accuracy and citation
   grounding metrics.

## Fixed Settings

- Sample size: n=100 first pass.
- Retriever for generation: dense v4, top-k 5.
- Embedding model: DashScope `text-embedding-v4`, 1024 dimensions.
- PubMedQA generation prompt: `pubmedqa_rag_json_v3_debug`.
- SciFact generation prompt: `scifact_rag_json_v3_debug`.
- Generation model profiles:
  - `rag_qwen3_30b_a3b_v3_report`
  - `rag_qwen3_30b_a3b_instruct_2507_v3_report`
- Thinking is disabled for both report model profiles.

## Commands

Retriever summaries:

```bash
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.evaluation.evaluate_artifact outputs/retrieval/pubmedqa_bm25_pooled_n100_fullcorpus_v01.jsonl --dataset pubmedqa --summary-json outputs/evaluation/report_pubmedqa_bm25_retrieval_n100_summary.json
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.evaluation.evaluate_artifact outputs/retrieval/pubmedqa_hybrid_v4_pooled_n100_fullcorpus_v01.jsonl --dataset pubmedqa --summary-json outputs/evaluation/report_pubmedqa_hybrid_v4_retrieval_n100_summary.json
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.evaluation.evaluate_artifact outputs/retrieval/pubmedqa_dense_v4_pooled_n100_fullcorpus_v01.jsonl --dataset pubmedqa --summary-json outputs/evaluation/report_pubmedqa_dense_v4_retrieval_n100_summary.json
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.evaluation.evaluate_artifact outputs/retrieval/scifact_bm25_pooled_n100_corpus1000_v01.jsonl --dataset scifact --summary-json outputs/evaluation/report_scifact_bm25_retrieval_n100_summary.json
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.evaluation.evaluate_artifact outputs/retrieval/scifact_hybrid_v4_pooled_n100_corpus1000_v01.jsonl --dataset scifact --summary-json outputs/evaluation/report_scifact_hybrid_v4_retrieval_n100_summary.json
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.evaluation.evaluate_artifact outputs/retrieval/scifact_dense_v4_pooled_n100_corpus1000_v01.jsonl --dataset scifact --summary-json outputs/evaluation/report_scifact_dense_v4_retrieval_n100_summary.json
```

Smoke generation commands use the same full-run inputs and outputs with
`--limit 3`. Full n=100 generation commands remove `--limit 3`.

## Artifact Paths

Retrieval inputs:

- `outputs/retrieval/pubmedqa_bm25_pooled_n100_fullcorpus_v01.jsonl`
- `outputs/retrieval/pubmedqa_hybrid_v4_pooled_n100_fullcorpus_v01.jsonl`
- `outputs/retrieval/pubmedqa_dense_v4_pooled_n100_fullcorpus_v01.jsonl`
- `outputs/retrieval/scifact_bm25_pooled_n100_corpus1000_v01.jsonl`
- `outputs/retrieval/scifact_hybrid_v4_pooled_n100_corpus1000_v01.jsonl`
- `outputs/retrieval/scifact_dense_v4_pooled_n100_corpus1000_v01.jsonl`

Report generation outputs:

- `outputs/generation/report_pubmedqa_dense_v4_qwen3_30b_a3b_v3_n100_v01.jsonl`
- `outputs/generation/report_pubmedqa_dense_v4_qwen3_30b_a3b_instruct_2507_v3_n100_v01.jsonl`
- `outputs/generation/report_scifact_dense_v4_qwen3_30b_a3b_v3_n100_v01.jsonl`
- `outputs/generation/report_scifact_dense_v4_qwen3_30b_a3b_instruct_2507_v3_n100_v01.jsonl`

Report model comparison outputs:

- `outputs/analysis/generation_ab/report_pubmedqa_qwen3_30b_a3b_vs_instruct_2507_v3_n100_summary.json`
- `outputs/analysis/generation_ab/report_pubmedqa_qwen3_30b_a3b_vs_instruct_2507_v3_n100.csv`
- `outputs/analysis/generation_ab/report_scifact_qwen3_30b_a3b_vs_instruct_2507_v3_n100_summary.json`
- `outputs/analysis/generation_ab/report_scifact_qwen3_30b_a3b_vs_instruct_2507_v3_n100.csv`

## Retriever Results

Status: saved report summaries.

| Dataset | Retriever | Main retrieval metric | Recall metric | Note |
| --- | --- | ---: | ---: | --- |
| PubMedQA | BM25 | context hit `0.90` | context recall `0.553571` | Baseline lexical retrieval. |
| PubMedQA | hybrid v4 | context hit `0.97` | context recall `0.762500` | Improves over BM25 but trails dense. |
| PubMedQA | dense v4 | context hit `0.99` | context recall `0.900952` | Stronger on both metrics. |
| SciFact | BM25 | sentence hit `0.52` | evidence recall `0.365111` | Baseline lexical retrieval. |
| SciFact | hybrid v4 | sentence hit `0.83` | evidence recall `0.624611` | Improves over BM25 but trails dense. |
| SciFact | dense v4 | sentence hit `0.95` | evidence recall `0.822190` | Stronger on both metrics. |

## Model Results

Status: smoke and full generation completed.

| Dataset | Model profile | Accuracy metric | Citation valid | Citation gold hit | Generation errors |
| --- | --- | ---: | ---: | ---: | ---: |
| PubMedQA | `rag_qwen3_30b_a3b_v3_report` | answer accuracy `0.62` | `0.98` | `0.99` | `0` |
| PubMedQA | `rag_qwen3_30b_a3b_instruct_2507_v3_report` | answer accuracy `0.68` | `0.99` | `0.98` | `0` |
| SciFact | `rag_qwen3_30b_a3b_v3_report` | label accuracy `0.86` | `1.00` | `0.92` | `0` |
| SciFact | `rag_qwen3_30b_a3b_instruct_2507_v3_report` | label accuracy `0.90` | `1.00` | `0.93` | `0` |

## Model Comparison Results

| Dataset | Better answer model | Accuracy gain | Answer fixes/regressions | Citation-gold change |
| --- | --- | ---: | ---: | ---: |
| PubMedQA | `qwen3-30b-a3b-instruct-2507` | `+0.06` | `13 / 7` | `-0.01` |
| SciFact | `qwen3-30b-a3b-instruct-2507` | `+0.04` | `5 / 1` | `+0.01` |

## Domain Results

| Model profile | PubMedQA accuracy | SciFact accuracy | PubMedQA citation gold hit | SciFact citation gold hit |
| --- | ---: | ---: | ---: | ---: |
| `rag_qwen3_30b_a3b_v3_report` | `0.62` | `0.86` | `0.99` | `0.92` |
| `rag_qwen3_30b_a3b_instruct_2507_v3_report` | `0.68` | `0.90` | `0.98` | `0.93` |

## Interpretation Notes

- Retrieval comparison should ignore placeholder `gold_label_demo` answer
  accuracy and cite retrieval-only metrics.
- Model comparison is fair only because retriever, prompt, sample size, top-k,
  temperature, token limit, and thinking setting are fixed.
- PubMedQA and SciFact metrics are related but not identical. Use citation
  metrics as the clearest cross-domain grounding bridge.
- Hybrid retrieval improves over BM25, but dense retrieval remains the strongest
  generation input: PubMedQA context recall is `0.553571` for BM25, `0.762500`
  for hybrid, and `0.900952` for dense; SciFact evidence recall is `0.365111`
  for BM25, `0.624611` for hybrid, and `0.822190` for dense.
- `qwen3-30b-a3b-instruct-2507` is the stronger report model on answer/label
  accuracy for both datasets, with similar citation grounding.
- SciFact appears easier for label accuracy than PubMedQA in this n=100 setup,
  even though both datasets have strong citation grounding under dense retrieval.

## Status

- [x] Plan locked.
  - [x] Experiment tracking files created.
- [x] Report model profiles added and tested.
- [x] Retriever summaries saved.
- [x] Smoke generation completed.
- [x] Full n=100 generation completed.
- [x] Generation summaries saved.
- [x] Report-ready conclusions written.
