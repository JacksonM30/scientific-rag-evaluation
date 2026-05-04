# Report Notes

## Retrieval Evidence Checkpoint

Current retrieval artifacts support only retrieval claims. The `prediction`
field still uses `gold_label_demo`, so PubMedQA answer accuracy and SciFact
label accuracy are sanity checks, not model performance.

### Available Artifacts

- PubMedQA BM25 pilot:
  `outputs/retrieval/pubmedqa_bm25_pooled_v01.jsonl`
  - 20 queries, 20 corpus rows, 64 passages, `top_k=5`
  - `context_hit_at_k=0.95`, `context_recall_at_k=0.670833`
- SciFact BM25 pilot:
  `outputs/retrieval/scifact_bm25_pooled_v01.jsonl`
  - 20 claims, 300 corpus docs, 2609 passages, `top_k=5`
  - `evidence_doc_hit_at_k=0.95`, `evidence_sentence_hit_at_k=0.75`,
    `evidence_all_hit_at_k=0.45`, `evidence_recall_at_k=0.545`
- PubMedQA dense/hybrid smoke:
  `outputs/retrieval/pubmedqa_dense_pooled_v01.jsonl`,
  `outputs/retrieval/pubmedqa_hybrid_pooled_v01.jsonl`
  - 5-query API-backed smoke runs only; useful for sanity, not report claims.
- SciFact dense/hybrid smoke:
  `outputs/retrieval/scifact_dense_pooled_v01.jsonl`,
  `outputs/retrieval/scifact_hybrid_pooled_v01.jsonl`
  - 5-claim API-backed smoke runs only; useful for sanity, not report claims.
- PubMedQA BM25 larger run:
  `outputs/retrieval/pubmedqa_bm25_pooled_n100_fullcorpus_v01.jsonl`
  - 100 queries, full PubMedQA corpus rows, 3358 passages, `top_k=5`
  - `context_hit_at_k=0.9`, `context_recall_at_k=0.553571`
- SciFact BM25 larger run:
  `outputs/retrieval/scifact_bm25_pooled_n100_corpus1000_v01.jsonl`
  - 100 claims, 1000 corpus docs, 8572 passages, `top_k=5`
  - `evidence_doc_hit_at_k=0.8`, `evidence_sentence_hit_at_k=0.52`,
    `evidence_all_hit_at_k=0.25`, `evidence_recall_at_k=0.365111`

### Failure Patterns

- PubMedQA BM25, 100-query run:
  - 25 examples retrieved every gold context section.
  - 65 examples retrieved only part of the gold context.
  - 10 examples retrieved no gold context.
  - Common pattern: the gold article appears for many questions, but `top_k=5`
    misses some abstract sections because PubMedQA often marks several sections
    as gold evidence.
- SciFact BM25, 100-claim run:
  - 25 examples retrieved all gold evidence sentences.
  - 27 examples retrieved only part of the gold sentence evidence.
  - 28 examples retrieved the gold document but the wrong sentence.
  - 20 examples missed the gold document entirely.
  - Common pattern: BM25 often finds the right paper but not the exact evidence
    sentence, which makes sentence-level evidence metrics much stricter than
    document-level metrics.

### Report Readiness

- Supported now: BM25 pooled-corpus retrieval metrics for PubMedQA and SciFact.
- Weak now: dense/hybrid comparisons, because current dense/hybrid artifacts are
  only 5-row smoke runs.
- Not supported yet: generated answer quality, citation precision, hallucination
  reduction, or model-vs-model comparisons.

Next evidence step: either scale dense/hybrid to the same sample sizes if API
budget is acceptable, or connect BM25 retrieval to one generation profile and
measure answer/citation behavior on a small sample.

## Future Idea: Context Expansion After Retrieval

Do not add neighboring context to `retrieved_passages`. Keep retrieval metrics
pure: `retrieved_passages` should mean only the original top-k passages returned
by BM25, dense, or hybrid retrieval.

Later, for generation, add a separate `context_passages` field:

```text
retriever top-k
-> expand each hit with a small same-document window
-> deduplicate
-> pass expanded context to the LLM
-> evaluate retrieval using only original top-k
```

This is more useful for SciFact than PubMedQA because SciFact passages are
individual sentences, while PubMedQA passages are already abstract sections.

Possible future artifact shape:

```json
{
  "retrieved_passages": ["original top-k only"],
  "context_passages": ["top-k plus neighboring passages"],
  "context_expansion": {
    "enabled": true,
    "window": 1,
    "scope": "same_doc"
  }
}
```

Distill: keep retrieval metrics clean now. Add neighboring context later only as
a generation-time input expansion, especially for SciFact.
