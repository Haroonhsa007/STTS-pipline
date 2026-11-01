# STTS Pipeline — local English ↔ Urdu conversation

An experimental Python/FastAPI pipeline for speech recognition, language-model
replies and speech synthesis on local hardware. Built and tested on an Apple M4
Mac mini with 16 GB RAM. The source is being prepared for community release.

Press **Start conversation** once, allow microphone access, and speak. Pauses send
your turn, replies play automatically, and the microphone stays open. Speaking
over a reply interrupts it. English and Urdu can change between utterances while
conversation history is retained. **End conversation** closes the microphone.

No Whisper backend or hidden Whisper fallback is used. This is automatic
turn-taking after a pause, not simultaneous word-by-word streaming recognition.
Latency, Urdu translation, factual accuracy and acoustic echo handling still need
improvement. See [measured results and limits](VALIDATION.md).

## Quick start

Clone/download this repository and open a terminal in its root directory. Use
native Apple Silicon Python 3.13. Allow roughly 10 GiB for dependencies, models,
installation and working files; the installer checks for 5 GiB free after the
Python dependencies are installed.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-bilingual.txt
python setup_bilingual.py
python tools/model_inventory.py
python web.py
```

Open **http://127.0.0.1:8765/**. Leave the engine on **Live English ↔ Urdu** and the
language on **Auto**. Start conversation prepares models automatically. Startup
is slower than subsequent turns. Headphones help avoid speaker-to-microphone
feedback during interruption testing.

If this checkout already has its environment and downloaded assets:

```bash
source .venv/bin/activate
python web.py
```

To stop the web server, press **Ctrl+C in the terminal running `web.py`**. It also
closes its owned inference processes. Ending the conversation in the page closes
capture but leaves the web server running.

The bilingual installer is tested on macOS arm64. Linux/Windows bilingual setup
and the 4 CPU / 8 GB server are not validated. The separate Native English
installer includes Linux artifacts; this does not establish bilingual Linux
support or equivalent performance.

## Pipeline

```text
Microphone → echo cancellation → speech/pause gate → utterance audio
                                                        │
                                              English/Urdu ASR
                                                        │
                                    Urdu → English translation, if needed
                                                        │
                                         LLM + conversation history
                                                        │
                                    English → Urdu translation, if needed
                                                        │
                                                  TTS → playback
                                                        │
                  speak over reply → cancel ────────────┘

                  recognition / translation / LLM / TTS → live metrics
```

The lightweight browser gate suppresses quiet/short input and DC offset; it is
not a trained speech classifier. Physical echo and noisy-room behavior depend
on the actual microphone and room. History retains 16 recent turns, not permanent
personal memory. Automatic language selection is restricted to English/Urdu.
Typed Roman Urdu is not automatically identified as Urdu.

## Downloaded models and engines

The following weights are present in the development checkout. A fresh clone
contains source and installers, not these weights. Sizes are logical GiB of the
weight files, not parameter counts.

| Asset | Role | Downloaded weights | Size |
| --- | --- | --- | ---: |
| LiquidAI LFM3-Audio-2B | Bilingual text backbone; separate Native English speech/text engine | Q4_K_M backbone + Q8_0 projector | 1.307 GiB |
| Bodhan Indic-Transcribe Flex | English/Urdu recognition | ONNX int8 encoder/decoder and external tensor files | 1.588 GiB |
| NLLB-200 distilled 600M | Urdu ↔ English translation | CTranslate2 int8 | 0.580 GiB |
| OmniVoice | English/Urdu speech synthesis | Q8_0 GGUF, including audio components | 1.258 GiB |
| **Total weight files** | | | **4.733 GiB** |

See [model inventory, source links and revisions](MODEL_INVENTORY.md). Run
`python tools/model_inventory.py` to check the files in your own checkout without
loading models or accessing the network. Native runtime distributions add about
127 MiB, and tokenizer/config files add some more storage.

Each selected model is below 3B parameters individually. The combined pipeline
is larger than 3B. The OmniVoice speech backbone is approximately 0.6B, with
additional audio components in its GGUF. Byte sizes do not prove parameter counts.

| Engine | Language fit | What to test |
| --- | --- | --- |
| `bilingual` / Live English ↔ Urdu | English and Urdu, per-turn detection | All four evaluation modes; recommended default |
| `native` / Native English | English only | Transcription, text replies and generated conversational speech; no verbatim `speak` mode |
| `python` / Python adapters | Depends on selected checkpoint/adapter | Experimental compatible models; extra dependencies or publisher access may be required |

Small direct Urdu LLM candidates produced poor results in local testing. The
current bilingual path translates Urdu around English reasoning. This improves
the tested responses but adds latency and can still introduce incorrect wording
or facts. The faster Native English option does not support Urdu.

## Test each stage

In the web page, choose the response mode:

| Mode | Input | Result |
| --- | --- | --- |
| Transcribe audio only (`transcribe`) | Recorded/uploaded audio | Transcript; optional reference-based WER/CER |
| Text reply (`ask`) | Typed text | LLM reply; Urdu includes translation |
| Read my text aloud (`speak`) | Typed text | TTS without LLM generation; bilingual or supported Python voice |
| Reply with voice (`run`) | Typed text or audio | Full reply; audio input includes recognition |

For a simple community evaluation from files, start `python web.py` in one
terminal, then use a second terminal:

```bash
python -m pip install -r requirements-dev.txt
python tools/evaluate.py examples/eval-text.jsonl --dry-run
python tools/evaluate.py examples/eval-text.jsonl --engine bilingual \
  --settings examples/eval-settings.json --output outputs/evaluation.json
```

The sample tests English/Urdu text replies, TTS and text-to-spoken-reply behavior.
It does **not** measure speech recognition. Copy
`examples/eval-audio.jsonl.example`, add your recordings and exact reference
transcripts, then evaluate those audio cases. Paths are relative to the manifest.

```bash
python tools/evaluate.py examples/my-audio.jsonl --engine bilingual \
  --conversation --output outputs/audio-evaluation.json
```

Use `--conversation` to retain history across cases, for example an English
introduction followed by an Urdu recall question. Without it, cases use isolated
sessions. The evaluator never starts the web server, opens a microphone or
downloads models. `--dry-run` only validates inputs; real evaluation requires the
server and installed assets. On timeout it cancels its own job and stops the batch.

For an English preset comparison, use the same cases on each engine:

```bash
python tools/evaluate.py examples/eval-english.jsonl --engine bilingual \
  --output outputs/english-bilingual.json
python tools/evaluate.py examples/eval-english.jsonl --engine native \
  --output outputs/english-native.json
```

For the existing synthetic spoken conversation stress test on macOS:

```bash
python tools/conversation_test.py --engine bilingual --min-tokens 1000 \
  --output outputs/conversation-test.json
```

That replay synthesizes English user speech with macOS Samantha and runs real
STT/LLM/TTS. It is not a human, Urdu or phone-call accuracy benchmark.

Read [COMMUNITY_EVALUATION.md](COMMUNITY_EVALUATION.md) for dataset format, settings,
model integration, fair comparisons and a results submission template.

## Metrics and current results

The dashboard reports recognition, translation, LLM and TTS stages; first emitted
audio; CPU; Python/child-process RSS; available RAM; swap; and free disk space.
Exports and `outputs/metrics.jsonl` retain per-turn evidence. Evaluation reports
separate operation, language, input type, cold/warm state, configuration and
verification options. Overlapped stage times must not be added as total latency.

A development replay completed **28 consecutive turns / 1,244 generated text
tokens**, with **5.57 s median / 10.78 s p95** until the first received HTTP audio
chunk and no token-limit cutoffs. These timings exclude actual playback and the
microphone's pause endpoint. The strict mean input WER was **2.19% on synthetic
English speech**, not general English/Urdu accuracy. See [VALIDATION.md](VALIDATION.md)
for the exact scope and the final long-reply regression fix.

WER/CER require human reference transcripts. TTS re-transcription is only an ASR
agreement proxy and adds latency. It does not measure naturalness. LLM factual
correctness and emotion adherence need labeled evaluation; those scores are not
fabricated. Voice identity, expression and reference audio are described in
[VOICE_CONTROLS.md](VOICE_CONTROLS.md); detailed UI metrics are in [WEB.md](WEB.md).

## Try other models and contribute

The optimized native and bilingual engines use pinned asset formats and adapters;
the web settings do not accept arbitrary Hugging Face model IDs. For compatible
experimental text checkpoints, the older CLI supports explicit overrides:

```bash
python STTS.py ask --llm-model YOUR_COMPATIBLE_CHECKPOINT \
  --text "Explain one benefit of local inference."
```

That command uses the **legacy Python adapter**, may download weights when not
in offline mode, and is not a replacement for the optimized bilingual engine.
Architecture compatibility and the parameter limit must be checked for each
candidate. See the community guide for adapter extension points.

Useful contributions include real English/Urdu references, within-sentence
language switches, noisy-room/echo tests, lower first-audio latency, consistent
voices, translation fixes, CPU/Linux ports and reproducible model comparisons.
Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting results or code.

## Developer checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
node --test tests/live-conversation.test.cjs
```

The software suite currently has 79 Python tests and 5 Node audio-control tests.
These checks do not establish pretrained-model accuracy. Model tests must be
reported separately with real weights, settings, hardware and data provenance.

## Files, privacy and publication

- `stts/`: pipeline, providers, metrics and web API.
- `web/`: plain HTML/CSS/JS, microphone capture and streaming playback.
- `tools/`: inventory, file-based evaluation and synthetic conversation replay.
- `examples/`: community evaluation manifests and settings.
- `models/`, `native/`: local downloaded assets; excluded from Git/source archives.
- `outputs/`: generated speech and reports; may contain private transcripts.

Uploaded raw audio/reference clips are temporary. App-generated UUID reply WAVs
are retained up to 32; named recordings and metrics are preserved. Do not publish
private transcripts, reference voices, credentials, caches or model weights with
benchmark results. The optional online font has a system-font fallback.

**Source-code license: [Apache-2.0](LICENSE).** Original pipeline source code and
documentation are licensed under the Apache License, Version 2.0. See [NOTICE](NOTICE)
for project attribution. Third-party model weights, codecs, converted checkpoints,
native runtimes and dependencies retain their own licenses, including custom and
noncommercial model terms. The code license does not replace those terms. See
[THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md).
