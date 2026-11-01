import io
import json
import threading
import time
from urllib.request import Request

import pytest

from stts.bilingual import (
    BilingualPipeline,
    VoiceRuntime,
    greeting_reply,
    text_language,
)
from stts.config import Config, PipelineError
from stts.metrics import Metrics
from stts.transport import event_lines
from stts.web import prune_generated_audio


def test_language_is_reconsidered_each_utterance():
    assert text_language("Hello") == "en"
    assert text_language("آپ کیسے ہیں؟") == "ur"
    assert text_language("Back to English", "ur") == "en"
    assert text_language("120", "ur") == "ur"
    assert text_language("اس کی price کیا ہے؟") == "ur"


def test_greeting_does_not_swallow_a_substantive_question():
    assert greeting_reply("السلام علیکم۔", "ur").startswith("وعلیکم")
    assert greeting_reply("السلام علیکم آپ کیسے ہیں؟", "ur")
    assert greeting_reply("السلام علیکم، چائے کیسے بناتے ہیں؟", "ur") is None
    assert greeting_reply("Hello", "en") is None


def test_partial_llm_stream_is_rejected(monkeypatch):
    pipeline = BilingualPipeline(Config(device="cpu"))
    pipeline.runtime.url = "http://127.0.0.1:1"
    fragment = {
        "choices": [
            {"delta": {"content": "An incomplete answer"}, "finish_reason": None}
        ]
    }
    monkeypatch.setattr(
        "stts.bilingual.event_lines",
        lambda *args: iter([b"data: " + json.dumps(fragment).encode()]),
    )
    with pytest.raises(PipelineError, match="before completion"):
        pipeline._answer("Hi", "en", Metrics("ask", pipeline.config))
    assert pipeline.history == []


def test_partial_voice_stream_is_rejected(monkeypatch):
    voice = VoiceRuntime(Config(device="cpu"))
    voice.url = "http://127.0.0.1:1"
    monkeypatch.setattr("stts.bilingual.event_lines", lambda *args: iter([]))
    with pytest.raises(PipelineError, match="before completion"):
        list(voice.synthesize("Hello", "en"))


def test_failed_initialization_can_retry(monkeypatch, tmp_path):
    config = Config(device="cpu", metrics_file=str(tmp_path / "metrics.jsonl"))
    pipeline = BilingualPipeline(config)
    monkeypatch.setattr("stts.bilingual.FlexRecognizer", lambda *args: object())

    def fail(*args):
        raise PipelineError("Missing translation weights")

    monkeypatch.setattr("stts.bilingual.TranslationBridge", fail)
    with pytest.raises(PipelineError, match="Missing translation"):
        pipeline.turn("warmup")
    assert pipeline.asr is None
    monkeypatch.setattr("stts.bilingual.TranslationBridge", lambda *args: object())
    monkeypatch.setattr(pipeline.runtime, "start", lambda: None)
    monkeypatch.setattr(pipeline.voice, "start", lambda: None)
    assert pipeline.turn("warmup")["status"] == "ok"
    assert pipeline.translator is not None


@pytest.mark.parametrize("terminal", [False, True])
def test_cancel_or_terminal_completion_releases_stalled_http_read(
    monkeypatch, terminal
):
    release = threading.Event()
    cancelled = threading.Event()

    class Response(io.BytesIO):
        status = 200

        def __iter__(self):
            if terminal:
                yield b"data: [DONE]\n"
            release.wait(3)

    class Wire:
        def shutdown(self, how):
            release.set()

    class Connection:
        sock = Wire()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(
        "stts.transport.http.client.HTTPConnection",
        lambda *args, **kwargs: Connection(),
    )
    aborted = []

    def check():
        if cancelled.is_set():
            raise PipelineError("Interrupted")

    timer = threading.Timer(0.05, cancelled.set)
    if not terminal:
        timer.start()
    started = time.monotonic()
    try:
        stream = event_lines(
            Request("http://127.0.0.1:1/test"), check, lambda: aborted.append(True)
        )
        if terminal:
            assert next(stream) == b"data: [DONE]\n"
            stream.close()
        else:
            with pytest.raises(PipelineError, match="Interrupted"):
                list(stream)
        assert time.monotonic() - started < 0.8
        assert not aborted  # The request closes; the loaded model stays resident.
    finally:
        release.set()
        if not terminal:
            timer.join()


def test_output_retention_preserves_named_files(tmp_path):
    import uuid

    for _ in range(5):
        (tmp_path / f"{uuid.uuid4()}.wav").write_bytes(b"audio")
    named = tmp_path / "my-recording.wav"
    named.write_bytes(b"user recording")
    prune_generated_audio(tmp_path, keep=2)
    assert named.read_bytes() == b"user recording"
    assert len(list(tmp_path.glob("*.wav"))) == 3
