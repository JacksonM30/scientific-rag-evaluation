# HotpotQA Mini Synthetic Fixture

Tiny synthetic HotpotQA-style JSONL fixture for the first inspectable RAG loop.
These examples were written manually for pipeline smoke tests; they are not
sampled from the official HotpotQA dataset.

Format: one example per line with these fields:

- `id`
- `question`
- `answer`
- `type`
- `level`
- `supporting_facts`: `[title, sentence_index]` pairs
- `context`: `[title, [sentences...]]` pairs

The sample is intentionally small and checked into Git. Use it for parser,
retriever, prompt, and artifact smoke tests. Do not use it for report-facing
dataset claims. Put larger HotpotQA exports under `../raw/` or generated caches
under `../cache/`.
