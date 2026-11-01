import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from stts import audio
from stts.cli import main, parser
from stts.config import Config, PipelineError
from stts.metrics import Metrics, error_rates, summarize
from stts.providers import (
    LiquidAudio,
    MioTTS,
    OmniTTS,
    Runtime,
    offline_torch_downloads,
    speech_chunks,
    spoken_text,
)


def tone(seconds=1, rate=16000):
    return (
        np.sin(np.arange(int(seconds * rate)) * 2 * np.pi * 220 / rate) * 0.2
    ).astype("float32")


@pytest.mark.parametrize(
    "argv",
    [
        ["--profile", "urdu", "doctor"],
        ["doctor", "--profile", "urdu"],
        ["doctor", "--language", "ur"],
    ],
)
def test_urdu_profile_and_option_positions(argv):
    config = Config.from_args(parser().parse_args(argv))
    assert (config.profile, config.language, config.llm_model, config.tts_backend) == (
        "urdu",
        "ur",
        "Qwen/Qwen3.5-2B",
        "omnivoice",
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"language": "ur"},
        {"expression": "sigh"},
        {"emotion": "happy"},
        {"speed": 0.8},
        {"max_new_tokens": 0},
        {"silence_threshold": 0},
        {"max_input_seconds": 31},
        {"mixed_script": True},
    ],
)
def test_invalid_config_rejected(changes):
    with pytest.raises(PipelineError):
        replace(Config(), **changes).validate()


def test_reference_transcript_required_for_omnivoice(tmp_path):
    clip = tmp_path / "voice.wav"
    sf.write(clip, tone(), 16000)
    with pytest.raises(PipelineError, match="exact --reference-text"):
        Config(tts_backend="omnivoice", reference_audio=str(clip)).validate()
    Config(
        tts_backend="omnivoice", reference_audio=str(clip), reference_text="Hello."
    ).validate()


def test_mio_emotion_language_validation(tmp_path):
    clip = tmp_path / "voice.wav"
    sf.write(clip, tone(), 16000)
    config = Config(
        profile="urdu",
        language="ur",
        tts_backend="indic-mio",
        reference_audio=str(clip),
        emotion="happy",
    )
    config.validate()
    with pytest.raises(PipelineError, match="supports these"):
        replace(config, emotion="whisper").validate()


def test_error_rates_english_and_urdu():
    assert error_rates("Hello, WORLD!", "hello world")["wer"] == 0
    assert error_rates("آپ کیسے ہیں؟", "آپ کیسے ہیں")["cer"] == 0
    assert error_rates("a b", "a x")["wer"] == 0.5
    assert error_rates("a", "a b c")["wer"] == 2
    with pytest.raises(ValueError):
        error_rates(" !!! ", "word")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<think>private</think>Hello", "Hello"),
        ("<think>unfinished", ""),
        ("private</think>Hello", "Hello"),
        ("Hello", "Hello"),
    ],
)
def test_reasoning_never_spoken(text, expected):
    assert spoken_text(text) == expected


def test_chunking_preserves_multilingual_text():
    original = ("آپ کیسے ہیں؟ آج موسم اچھا ہے۔ Hello, how are you? " * 8).strip()
    parts = speech_chunks(original, 80)
    assert all(0 < len(part) <= 80 for part in parts)
    assert " ".join(parts) == original
    assert "".join(speech_chunks("x" * 1000, 100)) == "x" * 1000


def test_audio_stereo_resampling_and_no_clobber(tmp_path):
    source = tmp_path / "stereo.wav"
    stereo = np.column_stack([tone(rate=48000), tone(rate=48000)])
    sf.write(source, stereo, 48000)
    samples, rate = audio.read_audio(source)
    assert samples.shape == (16000,)
    assert rate == 16000
    target = audio.save_wave(tmp_path / "output.wav", samples, rate)
    before = target.read_bytes()
    with pytest.raises(PipelineError, match="exists"):
        audio.save_wave(target, samples * 0.1, rate)
    assert target.read_bytes() == before
    audio.save_wave(target, samples * 0.1, rate, force=True)
    assert target.read_bytes() != before


def test_audio_rejects_silence_long_and_invalid_output(tmp_path):
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(16000), 16000)
    with pytest.raises(PipelineError, match="silent"):
        audio.read_audio(silent)
    long = tmp_path / "long.wav"
    sf.write(long, tone(seconds=2), 16000)
    with pytest.raises(PipelineError, match="seconds"):
        audio.read_audio(long, max_seconds=1)
    with pytest.raises(PipelineError, match="invalid"):
        audio.save_wave(tmp_path / "bad.wav", [float("nan")], 16000)


def test_repeated_metric_stages_are_preserved():
    metrics = Metrics("run", Config())
    with metrics.stage("load_audio"):
        pass
    with metrics.stage("load_audio"):
        pass
    assert set(metrics.data["stages"]) == {"load_audio", "load_audio__2"}


def test_stats_keep_device_cold_and_verification_separate(tmp_path):
    path = tmp_path / "metrics.jsonl"
    for index, (device, cold, verify) in enumerate(
        [
            ("cpu", True, False),
            ("cpu", False, False),
            ("mps", False, False),
            ("mps", False, True),
            ("mps", False, True),
        ]
    ):
        record = Metrics("run", Config())
        record.data.update(
            device=device,
            dtype="float32",
            cold_in_process=cold,
            run_options={"verify_tts": verify},
        )
        record.finish()
        record.data["total_wall_s"] = index + 1
        record.append(path)
    report = summarize(path)
    assert len(report["groups"]) == 4
    repeated = next(g for g in report["groups"] if g["runs"] == 2)
    assert repeated["statistics"]["total_wall_s"]["mean"] == 4.5
    assert repeated["statistics"]["total_wall_s"]["p95"] == pytest.approx(4.95)


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    from stts import pipeline

    class Speech:
        def __init__(self, runtime):
            self.rt = runtime

        def load(self, metrics):
            return self.rt.get("liquid_audio", lambda: "test speech model", metrics)

        def transcribe(self, path, metrics):
            self.load(metrics)
            with metrics.stage("asr", self.rt):
                samples, _ = audio.read_audio(path)
                assert samples.size
                return "Hello there."

        def synthesize(self, text, metrics):
            self.load(metrics)
            return tone(0.2, 24000), 24000

    class LLM:
        def __init__(self, runtime):
            self.rt = runtime

        def answer(self, messages, metrics):
            self.rt.get("llm", lambda: "test text model", metrics)
            with metrics.stage("llm", self.rt) as row:
                row.update(ttft_s=0.01, tokens_per_s=100, output_tokens=5)
                return "Hello there."

    monkeypatch.setattr(pipeline, "LiquidAudio", Speech)
    monkeypatch.setattr(pipeline, "TransformerLLM", LLM)
    config = Config(device="cpu", metrics_file=str(tmp_path / "metrics.jsonl"))
    return pipeline.VoicePipeline(config), tmp_path


def test_file_pipeline_warm_turns_and_quality(fake_pipeline):
    pipe, directory = fake_pipeline
    source = directory / "input.wav"
    sf.write(source, tone(), 16000)
    first = pipe.turn(
        "run",
        input_path=source,
        output=directory / "one.wav",
        reference="Hello there.",
        verify_tts=True,
    )
    second = pipe.turn("run", input_path=source, output=directory / "two.wav")
    assert first["quality"]["asr"]["wer"] == 0
    assert first["quality"]["tts_asr_proxy"]["wer"] == 0
    assert first["verification"]["stages"]["asr"]["status"] == "ok"
    assert first["cold_in_process"] is True
    assert second["cold_in_process"] is False
    assert first["response_first_audio_s"] > 0
    assert first["stages"]["tts"]["model_inference_s"] >= 0
    assert sf.info(first["output"]).samplerate == 24000
    assert len(pipe.history) == 4
    assert len(Path(pipe.config.metrics_file).read_text().splitlines()) == 2
    pipe.close()


def test_release_mode_preserves_both_audio_loads(fake_pipeline):
    pipe, directory = fake_pipeline
    pipe.config.release_between_stages = True
    source = directory / "input.wav"
    sf.write(source, tone(), 16000)
    record = pipe.turn("run", input_path=source, output=directory / "reply.wav")
    assert "load_liquid_audio" in record["stages"]
    assert "load_liquid_audio__2" in record["stages"]
    pipe.close()


def test_failed_turn_logged_without_fake_success(fake_pipeline):
    pipe, directory = fake_pipeline
    with pytest.raises(PipelineError, match="silent"):
        source = directory / "silent.wav"
        sf.write(source, np.zeros(16000), 16000)
        pipe.turn("run", input_path=source)
    record = json.loads(Path(pipe.config.metrics_file).read_text())
    assert record["status"] == "error"
    assert record["stages"]["input_prepare"]["status"] == "error"
    assert "output" not in record
    pipe.close()


def test_mio_adapter_uses_1d_codes_and_actual_sample_rate(tmp_path):
    from transformers import BatchEncoding

    config = Config(
        device="cpu", tts_backend="indic-mio", reference_audio="unused", emotion="happy"
    )
    runtime = Runtime(config)
    runtime.cache["mio_voice"] = torch.zeros(4)
    seen = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen["text"] = messages[0]["content"]
            return BatchEncoding({"input_ids": torch.tensor([[10, 11]])})

    class Model:
        def generate(self, **kwargs):
            return torch.tensor([[10, 11, 151670, 151671, 3]])

    class Codec:
        config = SimpleNamespace(sample_rate=24000)

        def parameters(self):
            yield torch.zeros(1)

        def decode(self, content_token_indices, global_embedding):
            assert content_token_indices.ndim == 1
            assert content_token_indices.tolist() == [1, 2]
            return torch.ones(100)

    backend = MioTTS(runtime)
    backend.load = lambda metrics: (Model(), Tokenizer(), Codec())
    wave, rate = backend.synthesize("Hello", Metrics("speak", config))
    assert seen["text"] == "Hello <happy>"
    assert rate == 24000
    assert wave.shape == (100,)
    runtime.release()


def test_cli_rejects_input_output_collision_before_model_load(tmp_path, capsys):
    source = tmp_path / "same.wav"
    sf.write(source, tone(), 16000)
    assert (
        main(["run", "--input", str(source), "--output", str(source), "--force"]) == 1
    )
    assert "different" in capsys.readouterr().err


def test_offline_blocks_auxiliary_torch_download_and_restores_hook():
    runtime = Runtime(Config(device="cpu", offline=True))
    original = torch.hub.download_url_to_file
    with (
        offline_torch_downloads(runtime),
        pytest.raises(PipelineError, match="auxiliary Torch"),
    ):
        torch.hub.download_url_to_file("https://example.invalid/model", "unused")
    assert torch.hub.download_url_to_file is original
    runtime.release()


def test_liquid_adapter_prompt_slice_codec_shape_and_shared_load(monkeypatch):
    import transformers
    from transformers import BatchEncoding

    calls = []

    class Codec:
        def to(self, **kwargs):
            return self

        def eval(self):
            return self

        def decode(self, codes):
            assert tuple(codes.shape) == (1, 2, 3)
            return SimpleNamespace(audio_values=torch.ones(1, 100))

    class Processor:
        audio_tokenizer = Codec()

        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return BatchEncoding({"input_ids": torch.tensor([[10, 11]])})

        def decode(self, tokens, **kwargs):
            assert tokens.tolist() == [12]
            return "Hello."

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            if kwargs.get("suppress_audio_out"):
                return SimpleNamespace(sequences=torch.tensor([[10, 11, 12]]))
            return SimpleNamespace(
                sequences=torch.tensor([[10, 11, 12]]),
                audio_codes=torch.zeros(1, 2, 5, dtype=torch.long),
                audio_lengths=torch.tensor([3]),
            )

    loads = []
    monkeypatch.setattr(
        transformers.AutoModelForMultimodalLM,
        "from_pretrained",
        lambda *a, **k: loads.append(k) or Model(),
    )
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *a, **k: Processor()
    )
    runtime = Runtime(Config(device="cpu"))
    adapter = LiquidAudio(runtime)
    metrics = Metrics("run", runtime.config)
    assert adapter.transcribe("unused.wav", metrics) == "Hello."
    wave, rate = adapter.synthesize("Hello.", metrics)
    assert len(loads) == 1
    assert loads[0]["trust_remote_code"] is True
    assert calls[0][0][0]["content"] == "Perform ASR in English."
    assert calls[1][0][-1]["content"] == "<|audio_start|>"
    assert calls[1][1]["continue_final_message"] is True
    assert rate == 24000 and len(wave) == 100
    runtime.release()


def test_omni_never_auto_transcribes_reference(monkeypatch, tmp_path):
    import sys

    reference = tmp_path / "voice.wav"
    sf.write(reference, tone(), 16000)
    config = Config(
        device="cpu",
        tts_backend="omnivoice",
        reference_audio=str(reference),
        reference_text="The exact words.",
    )
    runtime = Runtime(config)
    seen = {}

    class Model:
        sampling_rate = 24000

        def create_voice_clone_prompt(self, **kwargs):
            seen["reference"] = kwargs
            return "cached prompt"

        def generate(self, **kwargs):
            seen["generation"] = kwargs
            return [np.ones(100, dtype="float32")]

    class OmniVoice:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            seen["loading"] = kwargs
            return Model()

    monkeypatch.setitem(sys.modules, "omnivoice", SimpleNamespace(OmniVoice=OmniVoice))
    adapter = OmniTTS(runtime)
    adapter.synthesize("Hello.", Metrics("speak", config))
    assert seen["loading"]["load_asr"] is False
    assert seen["reference"]["ref_text"] == "The exact words."
    assert seen["generation"]["voice_clone_prompt"] == "cached prompt"
    assert "ref_audio" not in seen["generation"]
    runtime.release()


@pytest.fixture
def fake_microphone(monkeypatch):
    import sounddevice as sd

    level = {"value": 0.0}

    class Stream:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            if "on_open" in level:
                level["on_open"]()
            return self

        def __exit__(self, *args):
            pass

        def read(self, frames):
            return np.full((frames, 1), level["value"], dtype="float32"), False

    monkeypatch.setattr(
        sd,
        "query_devices",
        lambda *args, **kwargs: {
            "name": "Test USB microphone",
            "max_input_channels": 1,
        },
    )
    monkeypatch.setattr(sd, "InputStream", Stream)
    return level


def test_capture_silence_has_actionable_error_and_metrics(fake_microphone):
    details = {}
    with pytest.raises(PipelineError, match="Microphone returned silence"):
        audio.record(Config(max_input_seconds=1), details)
    assert details["max_rms"] == 0
    assert details["input_name"] == "Test USB microphone"
    assert details["speech_detected"] is False


def test_capture_quiet_signal_distinguished_from_silence(fake_microphone):
    fake_microphone["value"] = 0.002
    details = {}
    with pytest.raises(PipelineError, match="below the speech threshold"):
        audio.record(Config(max_input_seconds=1), details)
    assert details["max_rms"] == pytest.approx(0.002)


def test_capture_wait_starts_after_device_opens(fake_microphone, monkeypatch):
    clock = {"now": 0.0}
    fake_microphone["on_open"] = lambda: clock.update(now=100.0)
    monkeypatch.setattr(audio.time, "perf_counter", lambda: clock["now"])
    details = {}
    with pytest.raises(PipelineError):
        audio.record(Config(max_input_seconds=1), details)
    assert details["captured_seconds"] >= 1


def test_mic_suggestion_is_below_even_a_very_weak_signal(fake_microphone):
    fake_microphone["value"] = 0.00002
    report = audio.microphone_test(Config(), seconds=1)
    assert 0 < report["suggested_threshold"] < report["max_rms"]


def test_microphone_test_without_models_and_optional_wav(fake_microphone, tmp_path):
    fake_microphone["value"] = 0.002
    report = audio.microphone_test(Config(), seconds=1, output=tmp_path / "check.wav")
    assert report["seconds"] == 1
    assert report["threshold_crossed"] is False
    assert report["suggested_threshold"] < report["max_rms"]
    assert sf.info(report["output"]).frames == 16000
    fake_microphone["value"] = 0.02
    assert audio.microphone_test(Config(), seconds=1)["threshold_crossed"] is True


def test_listen_uses_typed_text_instead_of_recording(monkeypatch, capsys):
    from stts import pipeline

    calls = []

    class FakePipeline:
        def __init__(self, config):
            self.history = []

        def turn(self, operation, **kwargs):
            calls.append(kwargs)
            return {
                "transcript": kwargs["text"],
                "reply": "Hello!",
                "output": "test.wav",
            }

        def close(self):
            pass

    replies = iter(["hello", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(replies))
    monkeypatch.setattr(pipeline, "VoicePipeline", FakePipeline)
    assert main(["listen", "--play"]) == 0
    assert calls[0]["text"] == "hello"
    assert calls[0]["microphone"] is False
    assert "Assistant: Hello!" in capsys.readouterr().out
