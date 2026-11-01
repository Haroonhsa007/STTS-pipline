# Validation and remaining limits

Tested locally on the user's M4 Mac mini, 16 GB, Python 3.13, on October 31, 2026.
The workstation was shared with other active applications and used swap. These
are measured development runs, not isolated laboratory benchmarks or promises.

## Software verification

- 70 Python tests pass, including session isolation, serialization, input validation,
  initialization recovery, partial-stream rejection, cancellation during a blocked
  HTTP read, language routing, greeting scope, output retention and preparation
  without a typed message. Web tests use temporary output directories.
- 5 Node tests pass: continuous two-turn segmentation, noise/impulse rejection,
  acquired-track release during startup, permission/hide cancellation, and queued
  audio isolation across End→Start.
- Ruff checks and format checks pass; Python compilation and JavaScript syntax
  checks pass; `pip check` reports no broken dependencies.
- Astra High performed the final review. Its material findings were fixed.
  Physical acoustic echo behavior remains unverified.
- Real-model cancellation released the worker in 0.058 s (LLM) and 0.113 s (speech).
  The following spoken reply succeeded with first audio at 4.36 s and no model reload.
  Playback cancellation is handled immediately in the browser; these are server
  cancellation-to-completion timings, not acoustic interruption measurements.

## Real-model spoken replays

All recognition, reasoning and generated speech below used installed model
weights. User-side English speech was synthesized with macOS Samantha; no human
microphone or actual phone call is represented by these accuracy figures.
First HTTP audio receipt excludes actual speaker playback and the preceding
microphone endpoint pause.

| Replay | Completed turns | Generated text tokens | First HTTP audio p50 / p95 | Token-limit cutoffs |
| --- | ---: | ---: | --- | ---: |
| Native English |24|2028|2.78s /3.42s|0|
| Bilingual, before spoken-format fix |22|3553|7.05s /17.63s|0|
| Bilingual, sentence-format constraint |28|1244|5.57s /10.78s|0|

The unconstrained baseline was stopped after exceeding its token target; its
unfinished next turn is excluded. The28 turn replay's strict mean input WER was
2.19% on synthetic English. Warm recognition median RTF was 0.152. Warm reasoning
first-token median was 0.667 s. The cold first turn emitted audio after 32.90 s;
model preparation is now automatic when a fresh interactive session starts.
Another no-message preparation test took 41.79 s under host load. Startup and
warmed inference must be compared separately.

The28 turn replay exposed a final300 word semicolon-linked sentence: a sentence
count alone did not keep it brief. The final grammar also disallows semicolon
chains. Replaying that exact prompt and the preceding16 turns against the real
reasoning model produced three complete sentences/69 words in 5.84 s. There is no
hard character cutoff that would force punctuation in the middle of an idea.
The full replay figures above precede this last guard; they are not a fresh
28 turn benchmark of that guard.

Earlier user baseline: 78.66 s total, first waveform 43.67 s, STT 5.45 s, LLM 19.78 s,
TTS 53.10 s,128 token truncated answer. These are distinct workloads/configurations,
so their ratio is not a controlled hardware speedup claim.

## English ↔ Urdu and quality

A real-model sequence kept the same history through English introduction → Urdu
name/city question → English follow-up → spoken Urdu greeting → Urdu tea question.
It correctly recalled Ali and Regina and identified the spoken Urdu greeting.
Warmed first emitted audio in those cases was about 1.87–5.69 s, excluding startup.
The conventional full greeting's generated speech re-transcribed exactly in the
ASR proxy. Urdu translations still included awkward wording in the tea answer.

Three natural Urdu FLEURS test recordings routed to Urdu and produced strict WER
0.159, 0.350 and 0.143. The second includes substantial number/transliteration
normalization differences. These three clips do not establish overall Urdu
accuracy, within-sentence code-switch accuracy, or noisy-call accuracy. The
binary language score is conditional on English/Urdu, not a calibrated confidence
that speech is correct or belongs to either language.

The small direct Urdu LLM candidates produced incoherent answers in testing and
were rejected. The selected path uses NLLB translation around English reasoning.
That fixed the tested nonsensical Urdu responses, but does not eliminate
hallucinations or translation mistakes. In the long replay the assistant recalled
budget/desk width and the updated budget/reserve, but omitted the user's job from
a three part memory question and made unverified product/ergonomics suggestions.
There is no claimed LLM factual-accuracy score or hallucination-free certification.

Naturalness, emotion adherence, cross-language speaker identity and real human
intelligibility have not been scored. MOS/emotion values stay null. Optional TTS
ASR agreement measures text agreement only and adds latency.

## Interactive browser verification

Controlled PCM plus mocked HTTP/SSE in Chrome verified two automatically submitted
turns, automatic cancellation of the first, stopped queued audio, and continued
listening after the second, without per-turn button clicks. A separate pending-
POST interruption test sent deferred cancellation and prevented final-WAV autoplay.
These are control-flow tests, not physical speaker/microphone acoustic tests.

The browser requests echo cancellation and suppresses quiet/short noise and DC
input. Its lightweight gate is not a trained speech classifier. Real room noise
or speaker echo may still cause false onsets. Test with the intended microphone;
headphones avoid the simplest acoustic feedback path.

## What to improve next

- Measure real English/Urdu caller recordings, including brief replies, interruptions,
  names, numbers and language changes; use human reference transcripts.
- Evaluate factual answers against task-specific expected facts. Give the assistant
  verified business information before using it to answer business calls.
- Compare voices with blind listening and rate naturalness/emotion independently.
- Profile ASR, first speech chunk and swap separately. Current Urdu conversation
  is usable for experimentation but is not consistently sub-second real time.

Phone transport, telephony echo handling, external caller sessions and the4 CPU/
8 GB server have not been integrated or validated in this browser-only delivery.

## Community evaluation tooling update

The documentation/community release preparation adds nine evaluator regression
cases, bringing the software total to 79 Python tests plus 5 Node audio-control
tests. The new cases cover UTF-8 Urdu references, relative audio paths, manifest
validation, duplicate IDs, polling without exporting PCM, owned-job cancellation,
and separate language/cold/verification statistics. These use a mocked HTTP
transport; no new pretrained-model accuracy benchmark is claimed. Inventory and
evaluator dry runs do not load models, download weights or start the web server.
