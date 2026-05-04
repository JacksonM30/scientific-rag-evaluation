# Dataset Dissection Notes

This project is moving toward evidence-grounded RAG over scientific and
biomedical text tasks. HotpotQA remains useful for debugging the local pipeline,
but the next report-facing datasets are PubMedQA and SciFact.

Full/raw datasets should stay under ignored local paths such as `data/raw/`.
Only tiny hand-made fixtures should be committed.

## Inspection Command

Inspect PubMedQA:

```bash
conda run -n LLM python -m rag_experiment.data.inspect_datasets pubmedqa
```

Inspect SciFact:

```bash
conda run -n LLM python -m rag_experiment.data.inspect_datasets scifact
```

Inspect both and save a local JSON summary:

```bash
conda run -n LLM python -m rag_experiment.data.inspect_datasets all --save-json data/raw/dataset_inspection.json
```

The JSON summary is intentionally written under `data/raw/` by default because
it can include copied example rows from the source datasets.

## PubMedQA

Initial source: Hugging Face `qiaojin/PubMedQA`, config `pqa_labeled`, split
`train`.

Observed shape:

- Rows: 1000 labeled examples.
- Label counts from the current inspection: `yes=552`, `no=338`,
  `maybe=110`.
- Main fields: `pubid`, `question`, `context`, `long_answer`,
  `final_decision`.
- `final_decision` is the gold yes/no/maybe answer.
- `context` contains lists such as `contexts`, `labels`, `meshes`,
  `reasoning_required_pred`, and `reasoning_free_pred`.
- The abstract/context text is available, but there is no explicit gold
  evidence sentence id like HotpotQA supporting facts.

Likely future mapping:

- query: `question`
- gold answer: `final_decision`
- passages: entries from `context.contexts`
- metadata: `pubid`, context labels, MeSH terms, and `long_answer`

Main risk:

- Retrieval evidence evaluation will probably be context-level or section-level,
  not exact rationale-sentence matching.

## SciFact

Initial source: official SciFact release tarball:
`https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`

The Hugging Face `allenai/scifact` builder is script-based and is blocked by the
current `datasets` package, so the project inspector downloads the official
tarball directly into `data/raw/scifact/`.

Observed shape:

- Corpus file: `corpus.jsonl`.
- Claim files: `claims_train.jsonl`, `claims_dev.jsonl`, `claims_test.jsonl`.
- Current row counts: `corpus=5183`, `claims_train=809`,
  `claims_dev=300`, `claims_test=300`.
- Corpus rows include `doc_id`, `title`, `abstract`, and `structured`.
- Train/dev claim rows include `id`, `claim`, `evidence`, and usually
  `cited_doc_ids`.
- Evidence entries include a label such as `SUPPORT` or `CONTRADICT` and
  rationale sentence indices.
- Current train/dev evidence label counts: `SUPPORT=832`,
  `CONTRADICT=463`.
- Test claims are unlabeled and should not be used for first local metrics.

Likely future mapping:

- query: `claim`
- gold answer: evidence label, usually support/refute style
- passages: abstract sentences from `corpus.jsonl`
- gold evidence: evidence document id plus rationale sentence indices
- metadata: claim id, cited document ids, corpus doc title, structured flag

Main risk:

- SciFact is claim verification rather than ordinary QA, so generation prompts
  and answer accuracy should be label-based rather than free-form answer based.

## Why These Datasets Are Valuable

PubMedQA and SciFact are valuable together because they test different parts of
an evidence-grounded RAG system without dragging the project into PDF/table
engineering.

PubMedQA is biomedical yes/no/maybe QA over PubMed abstracts:

- It is good for answer accuracy because the gold answer space is simple:
  `yes`, `no`, or `maybe`.
- It is less clean for precise retrieval evaluation because it does not provide
  exact gold rationale sentence ids.
- It fits the project as a biomedical QA task where the model should answer from
  retrieved abstract context.

SciFact is scientific claim verification:

- It is good for grounding analysis because it has evidence labels and rationale
  sentence indices.
- It is less like normal QA because the model should classify a claim as
  support/refute-style rather than produce a free-form answer.
- It fits the project as a scientific evidence retrieval and verification task.

Together they support this project framing:

```text
Evidence-grounded RAG for scientific and biomedical text tasks:
  PubMedQA tests biomedical yes/no/maybe QA.
  SciFact tests scientific claim verification and evidence grounding.
```

Practical interpretation:

- PubMedQA is the easier dataset for answer accuracy.
- SciFact is the stronger dataset for retrieval evidence, citation, and
  grounding analysis.
- This pair is more focused and lower-risk than FinanceBench for the current
  project scope.

## Current Decision

The dataset dissection step has established the real PubMedQA and SciFact
schemas. Runtime sampling and pooled-corpus dataset runners are still deferred,
but the formal metric layer now has a fixed normalized artifact contract.

## Evaluation Contract

Metric code uses normalized artifact schema `v0.1`, documented in
`docs/EVALUATION.md`. PubMedQA and SciFact share the same artifact shape, but
their reliable metrics differ:

- PubMedQA is answer-accuracy-first, with coarse pooled-context retrieval
  checks.
- SciFact is evidence-retrieval-first, with rationale sentence evidence checks.

Older HotpotQA smoke-test artifacts are not part of this formal metric layer.
