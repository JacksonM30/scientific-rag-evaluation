# Evaluation

The first formal metric layer supports only normalized artifact schema `v0.1`.
It does not evaluate older HotpotQA smoke-test artifacts. Converters can be
added later if needed.

## Normalized Artifact v0.1

Each JSONL row must contain:

```json
{
  "schema_version": "v0.1",
  "dataset": "pubmedqa_or_scifact",
  "run": {
    "name": "run name",
    "retriever": "bm25",
    "top_k": 3,
    "model_profile": "optional"
  },
  "example": {
    "id": "example id",
    "query": "question or claim",
    "gold_answer": "yes/no/maybe or SUPPORT/CONTRADICT/NOT_ENOUGH_INFO",
    "gold_evidence": []
  },
  "retrieved_passages": [],
  "prediction": {
    "answer": "model answer or label",
    "cited_passage_ids": []
  },
  "error": null
}
```

Passage metadata must carry enough information for matching:

- PubMedQA: `metadata.pubid` and `metadata.context_idx`
- SciFact: `metadata.doc_id` and `metadata.sentence_index`

Gold evidence uses the same key fields:

- PubMedQA: `{"pubid": 21645374, "context_idx": 0}`
- SciFact: `{"doc_id": 12345, "sentence_index": 2}`

## Metrics

PubMedQA:

- `answer_accuracy`: normalized prediction equals `yes`, `no`, or `maybe`.
- `context_hit_at_k`: at least one gold PubMedQA context passage retrieved.
- `context_recall_at_k`: matched gold contexts divided by total gold contexts.

These retrieval metrics are meaningful only when retrieval is over a pooled
corpus, not only the current question's own attached context.

SciFact:

- `label_accuracy`: normalized prediction equals the gold claim-verification
  label.
- `evidence_doc_hit_at_k`: at least one gold evidence document retrieved.
- `evidence_sentence_hit_at_k`: at least one gold rationale sentence retrieved.
- `evidence_recall_at_k`: matched rationale sentences divided by total gold
  rationale sentences.
- `evidence_all_hit_at_k`: all gold rationale sentences retrieved.

SciFact records without usable gold evidence are skipped for evidence metrics,
not counted as failures.

Citation precision and unsupported-answer rate are intentionally deferred until
generation prompts and citation outputs are stable.

## Commands

Evaluate PubMedQA:

```bash
conda run -n LLM python -m rag_experiment.evaluation.evaluate_artifact data/evaluation_fixtures/pubmedqa_v01.jsonl --dataset pubmedqa
```

Evaluate SciFact:

```bash
conda run -n LLM python -m rag_experiment.evaluation.evaluate_artifact data/evaluation_fixtures/scifact_v01.jsonl --dataset scifact
```

Write an aggregate JSON summary:

```bash
conda run -n LLM python -m rag_experiment.evaluation.evaluate_artifact data/evaluation_fixtures/scifact_v01.jsonl --dataset scifact --summary-json /tmp/scifact_metrics.json
```

