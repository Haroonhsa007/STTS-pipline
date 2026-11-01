#!/usr/bin/env python3
"""Local speech -> text -> LLM -> speech. Run ``python STTS.py --help``.

The liquid profile uses the two requested LiquidAI checkpoints. The Urdu
profile uses Indic-Transcribe, Qwen3.5 and OmniVoice because LFM3 does not
document Urdu support. No Whisper backend or automatic voice transcription.
"""

from stts.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
