# Voice identity, expression and accuracy

The default bilingual voice is OmniVoice through native audio.cpp. It supports
English/Urdu, a voice description, speed, inference steps and an optional reference
clip with its exact transcript. The default 16 steps is a latency/quality compromise.
More steps or reference conditioning can increase latency. Try a short clean
reference spoken in the style you want, then listen to both languages.

The reference changes speaker identity/delivery; it does not establish reliable
named emotion control. This implementation does not claim validated happy/sad/
angry adherence in Urdu. Its live emotion and MOS values deliberately stay null.
The native bilingual adapter hides laughter/sigh controls it does not implement.

The optional legacy Python adapters include OmniVoice vocal events and Indic-Mio
named styles. Their accepted tags are not a benchmark of emotional reliability.
They are not selected by the recommended interactive preset and have additional
runtime/asset requirements. Liquid's Native English conversation uses its default
voice; voice customization there is more limited.

To compare voices, keep the text/reference/seed fixed, test numbers, names,
questions, Urdu and language switches, and listen blind. Record intelligibility,
identity consistency, naturalness 1–5 and intended emotion separately from latency.
ASR re-transcription is only a pronunciation/intelligibility proxy. It cannot
prove realistic delivery or reliable emotion control.

Primary references:
- https://huggingface.co/k2-fsa/OmniVoice
- https://github.com/0xShug0/audio.cpp/blob/main/docs/models/omnivoice.md
- https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF
- https://huggingface.co/SPRINGLab/Indic-Mio
