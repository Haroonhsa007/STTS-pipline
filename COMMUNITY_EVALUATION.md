# Community model testing and evaluation

Start from the repository README. Download the pinned assets once, start the
server yourself with `python web.py`, and keep it running while evaluating.
Stop it afterward with Ctrl+C in that terminal. The evaluation tools do not
start a background server or automatically install/download candidate models.

## 1. Establish a baseline

Record OS, CPU/GPU, RAM, Python version, engine, model revisions, quantization,
thread count, context/reply budgets, voice settings and whether the first run
loaded models. Save `python tools/model_inventory.py` output. Keep other workloads
consistent and include swap/memory observations when comparing latency.

Use `examples/eval-text.jsonl` for a no-audio smoke test. Use
`examples/eval-english.jsonl` to compare the Native English and bilingual presets
on identical English prompts. It intentionally excludes Native's unsupported
verbatim `speak` operation.

## 2. Add a dataset

Use UTF-8 JSONL: one JSON object per line. Available fields:

| Field | Meaning |
| --- | --- |
| `id` | Unique case identifier |
| `mode` | `transcribe`, `ask`, `speak`, or `run` |
| `language` | `en`, `ur`, or `auto`; Native English requires `en` |
| `text` | Text input for `ask`, `speak`, or `run` |
| `audio` | Audio file path relative to the manifest; required for `transcribe` |
| `reference` | Exact human transcript for audio WER/CER, if available |

Provide exactly one of `audio` or `text`. Recordings must fit the server's
30-second / 12 MiB upload limit. Use WAV, FLAC or AIFF supported by the audio
reader. Do not turn an inferred transcript into its own ground truth.

```json
{"id":"ur-001","mode":"transcribe","language":"auto","audio":"fixtures/urdu.wav","reference":"آپ کیسے ہیں؟"}
```

Copy and edit `examples/eval-audio.jsonl.example`, then validate without loading
models or contacting a server:

```bash
python tools/evaluate.py examples/my-audio.jsonl --dry-run
```

Dry-run checks the cases and local file paths. Web-setting values are validated
by the server when an actual evaluation is submitted.

Suggested slices: English; Urdu; English→Urdu→English turns; short yes/no answers;
names/numbers/dates; within-sentence code switches; background noise; long history;
and interruptions. Keep synthetic and human recordings as separate datasets.
Check data provenance/permission before sharing recordings or transcripts.

## 3. Run stage and conversation tests

```bash
python tools/evaluate.py examples/my-audio.jsonl --engine bilingual \
  --settings examples/eval-settings.json --output outputs/audio-evaluation.json
```

Cases use separate session histories by default. For a conversation sequence,
put turns in order and add `--conversation`. Recognition-only and TTS-only cases
do not create user/assistant reasoning history.

```bash
python tools/evaluate.py examples/my-audio.jsonl --engine bilingual \
  --conversation --output outputs/conversation-evaluation.json
```

The report saves per-case results and grouped p50/p95 statistics. It keeps
language, mode, input kind, cold/warm state, full model settings and verification
options separate. A group with one sample is not a useful latency distribution.
Model/server errors count as failures. Transport errors/timeouts stop the batch
rather than submitting further jobs into uncertain server state. Interrupted or
timed-out evaluator jobs receive a cancellation request; the server remains yours
to stop. Generated audio follows the server's normal retention rules.

Settings files accept the web API's supported settings, for example device,
reply token budget, synthesis steps, speed and `verify_tts`. The engine CLI flag
and each case's language take precedence. Inference is offline by default;
install assets before running. A model ID is not a valid web settings field.

For TTS agreement checks, set `"verify_tts": true` in a copy of the settings file
and compare separately from unverified latency. For reference-voice comparisons,
use the web UI's reference upload and exact transcript, or the legacy CLI's
`--reference-audio` / `--reference-text` options; the file evaluator does not
upload reference voices.

## 4. Interpret the measurements

| Stage | Useful evidence | What it does not establish |
| --- | --- | --- |
| STT | WER/CER against human references, language routing, substitutions on names/numbers, RTF | Overall Urdu accuracy from a few clean/synthetic clips |
| LLM | First-token delay, text tokens/s, expected facts, omissions, labeled factual errors | Correctness from speed, fluent wording or substring matches |
| Translation | Bilingual review of preserved meaning, names, negation and numbers | Accuracy from fluent Urdu alone |
| TTS | First emitted PCM, RTF, blind listening, identity consistency, intelligibility | MOS/emotion adherence from ASR agreement |
| Full session | First server audio p50/p95, endpoint delay, audible start, interruption behavior, failures, memory/swap | Acoustic latency from an HTTP timestamp |

The generic evaluator polls server metrics; its first-audio value is the server's
emission timestamp. It does not measure actual speaker output. The separate
`conversation_test.py` observes HTTP chunks but also excludes acoustic playback.
Overlapping LLM/TTS stage durations can sum to more than total turn time.
Normalization and Latin/Urdu transliterations can materially change strict WER.
Inspect error examples alongside the numeric score.

For subjective ratings, use the same anonymized sentences and blind listening.
Rate naturalness, intelligibility, speaker consistency and the intended emotion
separately. Leave unknown scores blank. For factual answers, define the expected
facts before running the model and record both wrong facts and missing facts.

## 5. Test or integrate another model

Preset A/B tests need no code changes: run the same English cases with
`--engine native` and `--engine bilingual`. Keep separate output files and compare
like-for-like warm runs. Synthesized input tests are useful for regression, not a
substitute for real speakers.

The optimized adapters are architecture-specific:

- `stts/native.py`: llama.cpp model/projector loading and native speech/text.
- `stts/bilingual.py`: ONNX Flex recognition, CT2 translation and audio.cpp voice.
- `stts/providers.py`: legacy Transformer/Ollama LLMs and Python speech adapters.
- `stts/pipeline.py`: legacy orchestration; `stts/web.py`: web presets and sessions.

To add a new architecture, implement its adapter and map it into a deliberate
preset. Verify tokenizer/sample-rate handling, language support, cancellation,
completion/error events, stage metrics and cleanup. Merely renaming a weight file
or editing a model ID does not establish compatibility. Preserve the no-Whisper
policy; do not add a hidden fallback. Keep the original under-3B-per-model target
explicit when submitting alternatives; quantization does not change parameter count.

For a compatible text model on the experimental Python path:

```bash
python STTS.py ask --llm-model YOUR_COMPATIBLE_CHECKPOINT \
  --text "Explain one benefit of local inference."
```

Use `python STTS.py --help` for current CLI options. New checkpoints can require
additional packages, custom code, publisher access or downloads. Keep those
experiments separate from the pinned native/bilingual baseline.

## Results submission template

```text
Hardware / OS / Python:
Code revision or source archive hash:
Engine / model IDs / model revisions / quantization:
Dataset provenance, language slice and number of cases:
Cold/warm counts; token/context/voice settings:
First server audio p50/p95; microphone endpoint and playback measurement method:
STT WER/CER and normalization:
TTS RTF and human listening method:
Expected facts, wrong facts and missing facts:
Interruptions, errors and incomplete replies:
Peak RAM / swap / competing workloads:
Exact command and anonymized report:
Known limitations:
```

Submit this with the smallest reproducer and a source patch if applicable. Do not
publish raw private voices or transcripts without permission. Third-party model
licenses remain separate from the pipeline source license.
