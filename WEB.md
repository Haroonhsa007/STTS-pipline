# Interactive conversation

Run `python web.py`, open http://127.0.0.1:8765, leave the engine on **Live English
↔ Urdu**, and press **Start conversation**. The browser needs one initial click
for microphone/audio permission. You then talk without pressing Send, Stop or
Interrupt for each turn. Speaking during a reply automatically cancels it.
End conversation closes capture. The optional microphone check loads no model.

```text
                         ONE ACTIVE SESSION
Microphone ── echo cancellation ── speech onset / pause detector
      ▲                                      │
      │ stays open                           │ utterance WAV
      │                                      ▼
      │                          Flex ASR + English/Urdu choice
      │                                      │
      │                          Urdu? → English translation
      │                                      │
      │                            shared conversation history
      │                                      │
      │                            LFM3 short spoken answer
      │                                      │
      │                          Urdu? → Urdu translation
      │                                      │
      │                              OmniVoice chunks
      │                                      │
      └── you speak → cancel reply ◄── streamed playback

                    Each stage → live metrics
```

Capture is continuous while the session is active; ASR processes completed
utterances. There is a0.75 second silence endpoint. Speech onset is confirmed over
roughly0.12 seconds, or0.3 seconds while assistant audio is playing, with preroll
preserving the beginning. A single utterance is capped at29 seconds. No-speech
input is discarded. Model inference remains one job at a time on the16 GB Mac.

Browser echo cancellation is requested. Speaker-to-microphone behavior depends
on the actual device/room and needs a physical test; synthetic tests cannot prove
that speaker echo never causes an interruption. Headphones are the simplest way
to avoid that acoustic feedback during testing. Phone-call integration is not
included in this local browser implementation.

## Language and voice

Auto chooses English or Urdu anew on each utterance. It can preserve English
words inside Urdu transcriptions, but arbitrary within-sentence code-switching
has not been exhaustively evaluated. The router does not claim general language
identification outside these two languages. Typed Arabic-script text selects
Urdu; Roman Urdu typed using only Latin letters is treated as English.

History is isolated by browser session and retains16 recent turns. Long-term
personal memory is not implemented. Urdu input/replies pass through translation;
model facts and translations can still be wrong. Conventional complete Urdu
greetings have a narrow direct reply; a greeting followed by a question is sent
to the reasoning stage.

Voice description and optional reference audio condition the bilingual voice.
Reference transcription is never guessed automatically. Laughter/sigh controls
are hidden for the native bilingual adapter because it does not implement them.
The default fixed seed does not prove identical voice identity across languages.

## Metrics

- **Turn latency:** server job elapsed time through completion, including optional
  preparation/verification and any overlapped work. It excludes actual speaker playback.
- **LLM first token:** reasoning-stage delay to first text token. Urdu translation
  has its own stages. It is not the time until the user hears speech.
- **First waveform:** elapsed turn time to the first emitted PCM chunk.
- **Browser scheduled first audio:** estimated scheduled WebAudio start relative
  to submission. It is not a measured acoustic or DAC timestamp.
- **Stage times:** recognition, translation, reasoning and synthesis. Speech can
  overlap generation, so adding stage times can exceed total turn time.
- **RTF:** stage compute elapsed divided by relevant audio duration; below 1 is
  faster than the audio duration. This alone does not prove low turn latency.
- **Tokens:** bilingual reasoning text tokens; Native English output steps can
  include audio and must not be compared directly with text tokens/second.
- **Accuracy:** input WER/CER require the optional exact reference transcript.
  TTS verification chunks its ASR pass into25 second pieces and measures text
  agreement, not human naturalness or emotion. Spelling/transliteration affects WER.
- **System:** all-core CPU; Python server RSS; owned child-process RSS; whole-system
  available RAM/swap; free disk. Stage CPU excludes native child CPU unless stated.

Exports hold up to 32 turns in the current page. Pipeline records append to
`outputs/metrics.jsonl`; `python STTS.py stats` separates config, detected language,
input type, verification options, device and cold/warm runs.

On interruption, queued sound is stopped and late chunks are ignored. A queued
microphone utterance belongs to its microphone-session generation; it cannot
spill into a newly started session. Cancellation first disconnects the request socket while retaining model weights.
Only if a blocked reader fails to release is its owned inference process restarted;
that fallback adds load latency to the next response.

Raw uploads are temporary. Only this application's UUID-named reply WAVs are
pruned to 32 on restart; named user recordings are not removed. Metrics and manual
experiments are retained. The page uses plain HTML/CSS/JS with no frontend build.
