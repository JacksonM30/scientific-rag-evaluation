# Data Layout

This repo should commit only tiny fixtures and configuration-friendly samples.

- `hotpotqa_mini/`: checked-in synthetic HotpotQA-style JSONL for smoke tests.
- `raw/`: ignored local downloads or full datasets.
- `cache/`: ignored processed dataset caches.

Do not commit full HotpotQA dumps or generated indexes. Keep large data in `data/raw/`
and record the source, split, and sampling seed in run configs/artifacts. Do not
use synthetic fixtures for report-facing dataset claims.
