"""Text completion checks shared by streaming and batch adapters."""

import re


def complete_prefix(text):
    """Keep complete sentences if a token cap cuts an answer mid-sentence."""
    text = text.strip()
    matches = list(re.finditer(r"[.!?۔؟](?:[\"\'”’)]*)(?=\s|$)", text))
    if matches:
        return text[: matches[-1].end()].strip()
    return ""


def speech_text(text):
    """Remove visual Markdown without discarding the spoken content."""
    text = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", "", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()
