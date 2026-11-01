# Contributing

Use the README for installation and [COMMUNITY_EVALUATION.md](COMMUNITY_EVALUATION.md)
for reproducible model comparisons. The original pipeline source is licensed
under [Apache-2.0](LICENSE). Unless explicitly stated otherwise, contributions
intentionally submitted for inclusion are provided under Apache-2.0, consistent
with Section 5 of that license. Model weights and third-party assets retain their
separate terms; see [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md).

Useful contributions include English/Urdu reference datasets, language-switch
cases, short replies, translation corrections, latency profiling, memory/CPU
improvements, acoustic echo tests, speaker consistency and platform ports.

For bug reports, include hardware/OS, model revisions, engine/settings, a minimal
input, expected versus actual behavior, cold/warm state, stage timings and an
anonymized error/report. Keep model accuracy reports separate from software test
results. Include evidence for claims such as faster inference or better Urdu.

Before submitting a source change:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
node --test tests/live-conversation.test.cjs
python tools/evaluate.py examples/eval-text.jsonl --dry-run
```

Add focused regression cases for changes to language routing, state/history,
cancellation, streaming or evaluation. Avoid tests that download weights or use a
physical microphone by default. Keep expensive real-model evaluations opt-in.
Preserve the no-Whisper requirement and isolate model-specific dependencies.

Do not include `models/`, `native/`, `.venv/`, private recordings, generated outputs,
secrets or caches in a source contribution. Retain upstream notices. Describe
new model formats, runtime requirements, parameter counts and license sources.
Do not claim production phone-call support or real-time performance without
measurements on the stated hardware and acoustic path.
