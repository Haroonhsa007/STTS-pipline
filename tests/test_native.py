import json
import threading
import time
from types import SimpleNamespace

import pytest

from stts.config import Config, PipelineError
from stts.metrics import Metrics
from stts.native import NativePipeline, NativeRuntime
from stts.providers import CancellationCheck
from stts.text import complete_prefix, speech_text
from stts.web import Job


@pytest.mark.parametrize(
    "system,machine,requested,expected",
    [
        ("Darwin", "arm64", "auto", "metal"),
        ("Darwin", "arm64", "cpu", "cpu"),
        ("Linux", "x86_64", "auto", "cpu"),
    ],
)
def test_device_reporting(monkeypatch, system, machine, requested, expected):
    monkeypatch.setattr("stts.native.platform.system", lambda: system)
    monkeypatch.setattr("stts.native.platform.machine", lambda: machine)
    assert NativeRuntime(Config(device=requested)).device == expected


def test_unsupported_native_accelerator_rejected():
    with pytest.raises(PipelineError, match="supports Apple Metal"):
        NativeRuntime(Config(device="cuda"))


def test_cutoff_keeps_complete_sentences_and_spoken_markdown_clean():
    text = "Python is useful. **Cons:**\n- It can be slow.\n- Some features may feel"
    assert complete_prefix(text).endswith("It can be slow.")
    assert "Some features" not in complete_prefix(text)
    assert speech_text("**Pros:**\n- Easy to read.") == "Pros:\nEasy to read."
    assert complete_prefix("unfinished sentence") == ""
    assert complete_prefix("آپ کیسے ہیں؟ ادھورا") == "آپ کیسے ہیں؟"


def test_streaming_audio_cursor_does_not_repeat_chunks():
    job = Job("id", "owner")
    job.event({"kind": "audio_chunk", "pcm": "AAAA", "sample_rate": 24000})
    assert len(job.snapshot()["deltas"]) == 1
    assert job.snapshot(after=1)["deltas"] == []
    job.event({"kind": "audio_chunk", "pcm": "BBBB", "sample_rate": 24000})
    assert [d["seq"] for d in job.snapshot(after=1)["deltas"]] == [2]


def test_cancel_checked_before_waiting_on_native_socket():
    pipeline = NativePipeline(Config(device="cpu"))
    pipeline.cancel_event = threading.Event()
    pipeline.cancel_event.set()
    sent = []
    connection = SimpleNamespace(send=lambda raw: sent.append(json.loads(raw)))
    with pytest.raises(PipelineError, match="interrupted"):
        pipeline._receive(connection, time.monotonic() + 1)
    assert sent == [{"type": "response.cancel"}]


def test_python_generation_streamer_observes_interrupt():
    event = threading.Event()
    check = CancellationCheck(SimpleNamespace(cancel_event=event))
    check.put(None)
    event.set()
    with pytest.raises(PipelineError, match="interrupted"):
        check.put(None)


def test_failed_partial_native_response_is_not_success(monkeypatch):
    pipeline = NativePipeline(Config(device="cpu"))
    events = iter(
        [
            {"type": "response.output_text.delta", "delta": "Partial answer"},
            {
                "type": "response.done",
                "response": {
                    "status": "failed",
                    "status_details": {"error": "failure"},
                },
            },
        ]
    )

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def send(self, value):
            pass

    monkeypatch.setattr(pipeline, "_session", lambda: (Connection(), {}))
    monkeypatch.setattr(pipeline, "_receive", lambda *args: next(events))
    with pytest.raises(PipelineError, match="response failed"):
        pipeline._conversation("Hello", Metrics("run", pipeline.config), False)


def test_full_disk_does_not_discard_successful_native_result(monkeypatch):
    pipeline = NativePipeline(Config(device="cpu"))
    pipeline.runtime.process = SimpleNamespace(poll=lambda: None)

    def disk_full(*args):
        raise OSError("disk full")

    monkeypatch.setattr(Metrics, "append", disk_full)
    assert pipeline.turn("warmup")["status"] == "ok"
