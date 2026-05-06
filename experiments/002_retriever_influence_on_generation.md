# Experiment 002: Retriever Influence on Generation

## Goal

Measure whether stronger retrieval improves downstream generation quality and
citation grounding.

This experiment fixes the generator and prompt, then compares BM25 against dense
v4 retrieval on the same n=100 PubMedQA and SciFact examples. Dense generation
artifacts are reused from Experiment 001; only BM25 generation is newly run.

## Fixed Settings

- Sample size: n=100.
- Compared retrievers: BM25 and dense v4, top-k 5.
- Fixed generation model profile: `rag_qwen3_30b_a3b_instruct_2507_v3_report`.
- PubMedQA prompt: `pubmedqa_rag_json_v3_debug`.
- SciFact prompt: `scifact_rag_json_v3_debug`.
- Thinking is disabled.

## Commands

Smoke BM25 generation:

```bash
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.runners.run_pooled_generation --dataset pubmedqa --input outputs/retrieval/pubmedqa_bm25_pooled_n100_fullcorpus_v01.jsonl --output outputs/generation/report_smoke_pubmedqa_bm25_qwen3_30b_a3b_instruct_2507_v3_n3_v01.jsonl --prompt-id pubmedqa_rag_json_v3_debug --model-profile rag_qwen3_30b_a3b_instruct_2507_v3_report --limit 3
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.runners.run_pooled_generation --dataset scifact --input outputs/retrieval/scifact_bm25_pooled_n100_corpus1000_v01.jsonl --output outputs/generation/report_smoke_scifact_bm25_qwen3_30b_a3b_instruct_2507_v3_n3_v01.jsonl --prompt-id scifact_rag_json_v3_debug --model-profile rag_qwen3_30b_a3b_instruct_2507_v3_report --limit 3
```

Full BM25 generation:

```bash
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.runners.run_pooled_generation --dataset pubmedqa --input outputs/retrieval/pubmedqa_bm25_pooled_n100_fullcorpus_v01.jsonl --output outputs/generation/report_pubmedqa_bm25_qwen3_30b_a3b_instruct_2507_v3_n100_v01.jsonl --prompt-id pubmedqa_rag_json_v3_debug --model-profile rag_qwen3_30b_a3b_instruct_2507_v3_report
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.runners.run_pooled_generation --dataset scifact --input outputs/retrieval/scifact_bm25_pooled_n100_corpus1000_v01.jsonl --output outputs/generation/report_scifact_bm25_qwen3_30b_a3b_instruct_2507_v3_n100_v01.jsonl --prompt-id scifact_rag_json_v3_debug --model-profile rag_qwen3_30b_a3b_instruct_2507_v3_report
```

Evaluation and comparison:

```bash
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.evaluation.evaluate_artifact outputs/generation/report_pubmedqa_bm25_qwen3_30b_a3b_instruct_2507_v3_n100_v01.jsonl --dataset pubmedqa --summary-json outputs/evaluation/report_pubmedqa_bm25_qwen3_30b_a3b_instruct_2507_v3_n100_summary.json
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.evaluation.evaluate_artifact outputs/generation/report_scifact_bm25_qwen3_30b_a3b_instruct_2507_v3_n100_v01.jsonl --dataset scifact --summary-json outputs/evaluation/report_scifact_bm25_qwen3_30b_a3b_instruct_2507_v3_n100_summary.json
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.analysis.generation_ab compare-models --dataset pubmedqa --baseline outputs/generation/report_pubmedqa_bm25_qwen3_30b_a3b_instruct_2507_v3_n100_v01.jsonl --candidate outputs/generation/report_pubmedqa_dense_v4_qwen3_30b_a3b_instruct_2507_v3_n100_v01.jsonl --baseline-label bm25 --candidate-label dense-v4 --output-csv outputs/analysis/generation_ab/report_pubmedqa_bm25_vs_dense_v4_generation_qwen3_30b_a3b_instruct_2507_v3_n100.csv --output-json outputs/analysis/generation_ab/report_pubmedqa_bm25_vs_dense_v4_generation_qwen3_30b_a3b_instruct_2507_v3_n100_summary.json
conda run -n LLM env PYTHONPATH=src python -m rag_experiment.analysis.generation_ab compare-models --dataset scifact --baseline outputs/generation/report_scifact_bm25_qwen3_30b_a3b_instruct_2507_v3_n100_v01.jsonl --candidate outputs/generation/report_scifact_dense_v4_qwen3_30b_a3b_instruct_2507_v3_n100_v01.jsonl --baseline-label bm25 --candidate-label dense-v4 --output-csv outputs/analysis/generation_ab/report_scifact_bm25_vs_dense_v4_generation_qwen3_30b_a3b_instruct_2507_v3_n100.csv --output-json outputs/analysis/generation_ab/report_scifact_bm25_vs_dense_v4_generation_qwen3_30b_a3b_instruct_2507_v3_n100_summary.json
```

## Artifact Paths

BM25 generation outputs:

- `outputs/generation/report_pubmedqa_bm25_qwen3_30b_a3b_instruct_2507_v3_n100_v01.jsonl`
- `outputs/generation/report_scifact_bm25_qwen3_30b_a3b_instruct_2507_v3_n100_v01.jsonl`

Reused dense generation outputs:

- `outputs/generation/report_pubmedqa_dense_v4_qwen3_30b_a3b_instruct_2507_v3_n100_v01.jsonl`
- `outputs/generation/report_scifact_dense_v4_qwen3_30b_a3b_instruct_2507_v3_n100_v01.jsonl`

Evaluation outputs:

- `outputs/evaluation/report_pubmedqa_bm25_qwen3_30b_a3b_instruct_2507_v3_n100_summary.json`
- `outputs/evaluation/report_scifact_bm25_qwen3_30b_a3b_instruct_2507_v3_n100_summary.json`

Comparison outputs:

- `outputs/analysis/generation_ab/report_pubmedqa_bm25_vs_dense_v4_generation_qwen3_30b_a3b_instruct_2507_v3_n100_summary.json`
- `outputs/analysis/generation_ab/report_pubmedqa_bm25_vs_dense_v4_generation_qwen3_30b_a3b_instruct_2507_v3_n100.csv`
- `outputs/analysis/generation_ab/report_scifact_bm25_vs_dense_v4_generation_qwen3_30b_a3b_instruct_2507_v3_n100_summary.json`
- `outputs/analysis/generation_ab/report_scifact_bm25_vs_dense_v4_generation_qwen3_30b_a3b_instruct_2507_v3_n100.csv`

## Results

Status: BM25 smoke/full generation, evaluation, and dense comparison completed.

| Dataset | Retriever | Retrieval recall | Accuracy metric | Citation valid | Citation gold hit | Citation gold recall | No citation | Generation errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PubMedQA | BM25 | context recall `0.553571` | answer accuracy `0.51` | `0.99` | `0.86` | `0.412381` | `0.11` | `0` |
| PubMedQA | dense v4 | context recall `0.900952` | answer accuracy `0.68` | `0.99` | `0.98` | `0.619048` | `0.00` | `0` |
| SciFact | BM25 | evidence recall `0.365111` | label accuracy `0.71` | `1.00` | `0.50` | `0.339000` | `0.23` | `0` |
| SciFact | dense v4 | evidence recall `0.822190` | label accuracy `0.90` | `1.00` | `0.93` | `0.778690` | `0.01` | `0` |

## Paired Comparison Results

| Dataset | Better retriever | Accuracy gain | Answer fixes/regressions | Citation-gold gain | Citation-gold fixes/regressions |
| --- | --- | ---: | ---: | ---: | ---: |
| PubMedQA | dense v4 | `+0.17` | `22 / 5` | `+0.12` | `12 / 0` |
| SciFact | dense v4 | `+0.19` | `23 / 4` | `+0.43` | `45 / 2` |

## Interpretation Notes

- This is the missing bridge between retrieval-only comparison and dense-only
  generation comparison.
- Dense retrieval improves both answer quality and citation grounding when the
  generator is held fixed. The effect is clear on PubMedQA and much larger on
  SciFact citation grounding.
- Dense retrieval also sharply reduces no-citation cases: PubMedQA goes from
  `0.11` to `0.00`; SciFact goes from `0.23` to `0.01`.
- Hybrid remains useful in Experiment 001 retrieval comparison, but is excluded
  here to keep the generation-impact story simple.
- Citation validity alone is not enough to claim grounding quality. BM25 already
  has high valid-citation rates because citations mostly refer to retrieved
  passage IDs, but its citations overlap gold evidence much less often.

## Status

- [x] Plan locked.
- [x] Experiment tracking file created.
- [x] BM25 smoke generation completed.
- [x] Full BM25 generation completed.
- [x] BM25 generation summaries saved.
- [x] BM25 vs dense comparisons saved.
- [x] Results and interpretation written.
