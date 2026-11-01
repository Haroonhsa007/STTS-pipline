import io
import json
import threading
import time
import uuid
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from stts.audio import save_wave
from stts.metrics import Metrics
from stts.web import create_app


class FakePipeline:
    instances: ClassVar[list] = []
    entered = threading.Event()
    release = threading.Event()
    block = False
    fail = False

    def __init__(self, config, event_sink=None):
        self.config, self.event_sink, self.history = config, event_sink, []
        self.instances.append(self)

    def close(self):
        pass

    def turn(
        self,
        mode,
        input_path=None,
        text=None,
        output=None,
        verify_tts=False,
        reference=None,
    ):
        metrics = Metrics(mode, self.config, self.event_sink)
        with metrics.stage("llm") as row:
            row.update(ttft_s=0.01, output_tokens=4, tokens_per_s=40)
            self.entered.set()
            if self.block:
                assert self.release.wait(4)
            if self.fail:
                raise RuntimeError("Test model failure")
        transcript = "Audio transcript" if input_path else text
        reply = f"Previous turns: {len(self.history) // 2}"
        metrics.emit("transcript", text=transcript)
        metrics.emit("reply", text=reply)
        self.history.extend(
            [
                {"role": "user", "content": transcript},
                {"role": "assistant", "content": reply},
            ]
        )
        metrics.data.update(transcript=transcript, reply=reply, device="cpu")
        if mode not in ("ask", "transcribe"):
            save_wave(
                output, np.sin(np.arange(1600) * 0.1).astype("float32") * 0.2, 16000
            )
            metrics.data["output"] = str(output)
        return metrics.finish()


@pytest.fixture(autouse=True)
def isolated_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr("stts.web.OUTPUTS", tmp_path / "outputs")


@pytest.fixture
def client():
    FakePipeline.instances = []
    FakePipeline.entered.clear()
    FakePipeline.release.clear()
    FakePipeline.block = False
    FakePipeline.fail = False
    with TestClient(create_app(FakePipeline)) as client:
        yield client


def submit(client, owner, **changes):
    payload = {"session": owner, "text": "Hello", "mode": "run", "settings": "{}"}
    payload.update(changes)
    return client.post("/api/turn", data=payload)


def wait_job(client, identifier, owner):
    for _ in range(100):
        response = client.get(f"/api/jobs/{identifier}", params={"session": owner})
        assert response.status_code == 200
        result = response.json()
        if result["status"] != "running":
            return result
        time.sleep(0.01)
    raise AssertionError("Job did not complete")


def clip(seconds=0.1):
    output = io.BytesIO()
    sf.write(
        output,
        np.sin(np.arange(round(seconds * 16000)) * 0.1) * 0.2,
        16000,
        format="WAV",
    )
    return output.getvalue()


def test_static_lazy_initialization_and_system(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/recorder-worklet.js").status_code == 200
    assert not FakePipeline.instances
    assert client.get("/api/system").json()["system_total_mb"] > 0
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400
    assert (
        client.post("/api/turn", headers={"Origin": "https://evil.example"}).status_code
        == 403
    )


def test_turn_events_audio_and_session_isolation(client):
    owner = str(uuid.uuid4())
    response = submit(client, owner)
    assert response.status_code == 202
    identifier = response.json()["id"]
    result = wait_job(client, identifier, owner)
    assert result["status"] == "ok"
    assert result["stages"]["llm"]["output_tokens"] == 4
    events = client.get(f"/api/jobs/{identifier}/events", params={"session": owner})
    snapshot = json.loads(events.text.strip().removeprefix("data: "))
    assert snapshot["reply"] == "Previous turns: 0"
    assert "system" in snapshot
    audio = client.get(result["audio_url"])
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
    assert (
        client.get(
            f"/api/jobs/{identifier}", params={"session": str(uuid.uuid4())}
        ).status_code
        == 404
    )
    Path(result["output"]).unlink()


def test_models_reused_and_history_isolated(client):
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    for owner, expected in ((first, 0), (first, 1), (second, 0), (first, 2)):
        response = submit(client, owner, mode="ask")
        result = wait_job(client, response.json()["id"], owner)
        assert result["reply"] == f"Previous turns: {expected}"
    assert len(FakePipeline.instances) == 1


def test_busy_job_progress_and_failure_recovery(client):
    owner = str(uuid.uuid4())
    FakePipeline.block = True
    identifier = submit(client, owner, mode="ask").json()["id"]
    assert FakePipeline.entered.wait(2)
    assert submit(client, owner).status_code == 409
    time.sleep(0.45)
    running = client.get(f"/api/jobs/{identifier}", params={"session": owner}).json()
    assert running["status"] == "running" and running["stages"]["llm"]["wall_s"] >= 0.3
    FakePipeline.release.set()
    assert wait_job(client, identifier, owner)["status"] == "ok"
    FakePipeline.block = False
    FakePipeline.fail = True
    failed = submit(client, owner, mode="ask").json()["id"]
    assert wait_job(client, failed, owner)["error"] == "Test model failure"
    FakePipeline.fail = False
    recovered = submit(client, owner, mode="ask").json()["id"]
    assert wait_job(client, recovered, owner)["status"] == "ok"


def test_audio_upload_validation_and_unicode(client):
    owner = str(uuid.uuid4())
    valid = client.post(
        "/api/turn",
        data={"session": owner, "mode": "transcribe"},
        files={"audio": ("voice.wav", clip(), "audio/wav")},
    )
    assert valid.status_code == 202
    assert (
        wait_job(client, valid.json()["id"], owner)["transcript"] == "Audio transcript"
    )
    for contents in (b"garbage", clip(30.1), b""):
        invalid = client.post(
            "/api/turn",
            data={"session": owner},
            files={"audio": ("x.wav", contents, "audio/wav")},
        )
        assert invalid.status_code == 422
    response = submit(client, owner, text="آپ کیسے ہیں؟", mode="ask")
    assert (
        wait_job(client, response.json()["id"], owner)["transcript"] == "آپ کیسے ہیں؟"
    )


@pytest.mark.parametrize(
    "settings",
    [
        '{"engine":"native","language":"ur"}',
        '{"engine":"native","tts_backend":"liquid","speed":1.2}',
        '{"max_new_tokens":9999}',
        '{"reference_audio":"/etc/passwd"}',
        '{"device":"bad"}',
    ],
)
def test_invalid_options_rejected_before_loading(client, settings):
    assert submit(client, str(uuid.uuid4()), settings=settings).status_code == 422
    assert not FakePipeline.instances


def test_no_input_or_ambiguous_input_rejected(client):
    owner = str(uuid.uuid4())
    assert submit(client, owner, text="").status_code == 422
    assert (
        client.post(
            "/api/turn",
            data={"session": owner, "text": "Hello"},
            files={"audio": ("x.wav", clip(), "audio/wav")},
        ).status_code
        == 422
    )
    assert submit(client, "bad-session").status_code == 422


def test_failed_device_change_can_return_to_previous_device():
    def factory(config, event_sink=None):
        if config.device == "cuda":
            raise RuntimeError("CUDA unavailable")
        return FakePipeline(config, event_sink)

    with TestClient(create_app(factory)) as client:
        owner = str(uuid.uuid4())
        for device, status in (("cpu", "ok"), ("cuda", "error"), ("cpu", "ok")):
            response = submit(
                client, owner, mode="ask", settings=json.dumps({"device": device})
            )
            assert wait_job(client, response.json()["id"], owner)["status"] == status


def test_text_only_does_not_require_reference_voice(client):
    owner = str(uuid.uuid4())
    response = submit(
        client,
        owner,
        mode="ask",
        settings='{"tts_backend":"indic-mio","emotion":"happy"}',
    )
    assert response.status_code == 202
    assert wait_job(client, response.json()["id"], owner)["status"] == "ok"


def test_reference_identity_stable_and_audio_ready_failure_metrics(tmp_path):
    from stts.web import Job, Settings

    for name in ("first.wav", "second.wav"):
        (tmp_path / name).write_bytes(clip())
    options = Settings(
        engine="python", language="en", tts_backend="omnivoice", reference_text="Hello"
    )
    first = options.config(tmp_path / "first.wav")
    second = options.config(tmp_path / "second.wav")
    assert first.public_dict() == second.public_dict()
    assert first.public_dict()["reference_audio"].startswith("sha256:")
    job = Job(str(uuid.uuid4()), str(uuid.uuid4()))
    metrics = Metrics("run", first, job.event)
    metrics.data.update(device="cpu", quality={"asr": {"wer": 0.5}})
    metrics.emit("audio_ready", response_first_audio_s=1.2)
    assert job.snapshot()["response_first_audio_s"] == 1.2
    metrics.finish("error", RuntimeError("Test failure"))
    result = job.snapshot()
    assert result["config"] == first.public_dict()
    assert result["device"] == "cpu" and result["quality"]["asr"]["wer"] == 0.5


def test_prepare_models_needs_no_message_or_microphone(client):
    owner = str(uuid.uuid4())
    response = client.post("/api/turn", data={"session": owner, "mode": "warmup"})
    assert response.status_code == 202
    assert wait_job(client, response.json()["id"], owner)["status"] == "ok"
