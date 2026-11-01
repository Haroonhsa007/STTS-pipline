"""Quantized LFM3-Audio via a private, project-managed llama.cpp process.

One 2B model supplies recognition and interleaved text/speech. PCM leaves the
model as it is generated; it is never held until the complete WAV is ready.
"""

import base64
import json
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .audio import read_audio, save_wave
from .config import PipelineError
from .metrics import Metrics, error_rates

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "native/llama-b11960/llama-server"
MODEL = ROOT / "models/native/LFM3-Audio-2B-Q4_K_M.gguf"
PROJECTOR = ROOT / "models/native/mmproj-LFM3-Audio-2B-Q8_0.gguf"


def available():
    return all(path.is_file() for path in (BINARY, MODEL, PROJECTOR))


class NativeRuntime:
    def __init__(self, config, model_path=MODEL, projector_path=PROJECTOR):
        self.config = config
        self.model_path, self.projector_path = (
            Path(model_path),
            Path(projector_path) if projector_path else None,
        )
        self.process = None
        self.log = None
        self.url = None
        apple = platform.system() == "Darwin" and platform.machine() == "arm64"
        if config.device == "cuda" or (config.device == "mps" and not apple):
            raise PipelineError(
                "This native build supports Apple Metal on M-series Macs or CPU. Choose auto or cpu."
            )
        self.device = "metal" if apple and config.device != "cpu" else "cpu"

    def sync(self):
        pass  # The synchronous API includes device completion in its response.

    def memory(self):
        if not self.process or self.process.poll() is not None:
            return {}
        import psutil

        try:
            return {
                "native_process_rss_mb": psutil.Process(self.process.pid)
                .memory_info()
                .rss
                / 2**20
            }
        except psutil.Error:
            return {}

    def start(self):
        if self.process and self.process.poll() is None:
            return False
        if (
            not BINARY.is_file()
            or not self.model_path.is_file()
            or (self.projector_path and not self.projector_path.is_file())
        ):
            raise PipelineError(
                "Native runtime is not installed. Run python setup_native.py, or select the Python engine."
            )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        folder = ROOT / "outputs"
        folder.mkdir(exist_ok=True)
        self.log = (folder / "native-runtime.log").open("w")
        command = [
            str(BINARY),
            "-m",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            "4096",
            "-np",
            "1",
            "-t",
            str(self.config.threads),
            "-ngl",
            "0" if self.device == "cpu" else "99",
            "--reasoning",
            "off",
            "--temp",
            "0",
            "--no-webui",
        ]
        if self.projector_path:
            command.extend(["-mm", str(self.projector_path)])
        self.process = subprocess.Popen(
            command, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise PipelineError(
                    "Native runtime failed to start. See outputs/native-runtime.log."
                )
            try:
                with urllib.request.urlopen(
                    self.url + "/health", timeout=1
                ) as response:
                    if response.status == 200:
                        return True
            except (OSError, urllib.error.URLError):
                time.sleep(0.15)
        self.close()
        raise PipelineError("Native runtime did not become ready within 60 seconds.")

    def abort(self):
        self.close(grace_seconds=0.2)

    def close(self, grace_seconds=5):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log:
            self.log.close()
        self.process = None


class NativePipeline:
    def __init__(self, config, event_sink=None):
        if config.language != "en" or config.tts_backend != "liquid":
            raise PipelineError(
                "Native conversation currently supports English with the model's own voice. Choose Python for Urdu or reference/emotion voices."
            )
        self.config, self.event_sink = config, event_sink
        self.runtime = NativeRuntime(config)
        self.history = []
        self.cancel_event = None

    def close(self):
        self.runtime.close()

    def _receive(self, connection, deadline):
        while True:
            if self.cancel_event and self.cancel_event.is_set():
                try:
                    connection.send(json.dumps({"type": "response.cancel"}))
                finally:
                    raise PipelineError("Reply interrupted.")
            if time.monotonic() >= deadline:
                raise PipelineError(
                    "Native response timed out. Interrupt and try a shorter turn."
                )
            try:
                event = json.loads(connection.recv(timeout=0.3))
                if event.get("type") == "error":
                    raise PipelineError(
                        event.get("error", {}).get(
                            "message", "Native inference failed."
                        )
                    )
                return event
            except TimeoutError:
                continue

    def _session(self, transcription=False):
        from websockets.sync.client import connect

        url = self.runtime.url.replace("http://", "ws://") + "/v1/realtime"
        if transcription:
            url += "?intent=transcription"
        connection = connect(url, open_timeout=10, max_size=8 * 1024 * 1024, proxy=None)
        try:
            event = self._receive(connection, time.monotonic() + 15)
            if event.get("type") != "session.created":
                raise PipelineError("Native runtime returned no realtime session.")
        except BaseException:
            connection.close()
            raise
        return connection, event["session"]

    def _transcribe(self, path, metrics):
        import numpy as np

        samples, rate = read_audio(path, self.config.max_input_seconds)
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        result = ""
        with metrics.stage("asr", self.runtime) as row:
            connection, _session_info = self._session(True)
            with connection:
                connection.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "transcription",
                                "audio": {
                                    "input": {"transcription": {"language": "en"}}
                                },
                            },
                        }
                    )
                )
                for offset in range(0, len(pcm), rate * 2):
                    connection.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(
                                    pcm[offset : offset + rate * 2]
                                ).decode(),
                            }
                        )
                    )
                connection.send(
                    json.dumps({"type": "input_audio_buffer.commit", "final": True})
                )
                deadline = time.monotonic() + 90
                while True:
                    event = self._receive(connection, deadline)
                    if (
                        event["type"]
                        == "conversation.item.input_audio_transcription.delta"
                    ):
                        result += event.get("delta", "")
                        metrics.emit("transcript", text=result)
                    elif (
                        event["type"]
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        result = event.get("transcript", result)
                        break
            row["audio_seconds"] = len(samples) / rate
        metrics.data["input_audio_seconds"] = len(samples) / rate
        row["rtf"] = row["wall_s"] / (len(samples) / rate)
        return result.strip()

    def _conversation(self, text, metrics, voice):
        import numpy as np

        instructions = (
            "Respond with interleaved text and audio. "
            if voice
            else "Respond in text. "
        )
        instructions += "Talk naturally with the person in English. Match the length to their question: a brief acknowledgement can be a few words; explanations usually need one to three short, complete sentences. Use contractions and a friendly, direct tone. Don't repeat the question, keep introducing yourself, or end every answer by offering help. No markdown, bullets, headings or asterisks. Finish the last sentence."
        if self.history:
            instructions += "\nRecent conversation for context:\n" + json.dumps(
                self.history[-self.config.history_turns * 2 :], ensure_ascii=False
            )
        chunks, reply, rate = [], "", 24000
        stage = "speech_to_speech" if voice else "llm"
        with metrics.stage(stage, self.runtime) as row:
            connection, session = self._session()
            with connection:
                rate = (
                    session.get("audio", {})
                    .get("output", {})
                    .get("format", {})
                    .get("rate", 24000)
                )
                connection.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "realtime",
                                "instructions": instructions,
                                "output_modalities": ["text", "audio"]
                                if voice
                                else ["text"],
                                "max_output_tokens": 2048
                                if voice
                                else self.config.max_new_tokens,
                            },
                        }
                    )
                )
                connection.send(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": text}],
                            },
                        }
                    )
                )
                began = time.perf_counter()
                connection.send(json.dumps({"type": "response.create"}))
                deadline = time.monotonic() + 120
                while True:
                    event = self._receive(connection, deadline)
                    kind = event["type"]
                    if kind == "response.output_text.delta":
                        if "ttft_s" not in row:
                            row["ttft_s"] = time.perf_counter() - began
                        reply += event.get("delta", "")
                        metrics.emit("reply_delta", text=reply)
                    elif kind == "response.output_audio.delta":
                        raw = base64.b64decode(event["delta"])
                        if not raw:
                            continue
                        chunks.append(
                            np.frombuffer(raw, dtype="<i2").astype("float32") / 32768
                        )
                        if "first_audio_s" not in row:
                            row["first_audio_s"] = time.perf_counter() - began
                            metrics.data["response_first_audio_s"] = (
                                time.perf_counter() - metrics.started
                            )
                            metrics.emit(
                                "audio_ready",
                                response_first_audio_s=metrics.data[
                                    "response_first_audio_s"
                                ],
                            )
                        metrics.emit(
                            "audio_chunk", pcm=event["delta"], sample_rate=rate
                        )
                    elif kind == "response.done":
                        response = event.get("response", {})
                        usage = response.get("usage", {})
                        row["output_tokens"] = usage.get("output_tokens")
                        row["token_scope"] = (
                            "Native combined text/audio steps; not comparable to text-only tokens/s"
                            if voice
                            else "Text tokens"
                        )
                        row["finish_status"] = response.get("status")
                        row["finish_details"] = response.get("status_details")
                        if response.get("status") in ("failed", "cancelled"):
                            raise PipelineError(
                                f"Native response {response['status']}: {response.get('status_details') or 'generation stopped'}"
                            )
                        metrics.data["reply_truncated"] = (
                            response.get("status") == "incomplete"
                        )
                        break
            if chunks:
                row["audio_seconds"] = sum(len(chunk) for chunk in chunks) / rate
        if chunks:
            row["rtf"] = row["wall_s"] / row["audio_seconds"]
            metrics.data["quality"]["emotion_target"] = "model default voice"
        if not reply.strip():
            raise PipelineError("Native model returned no reply text.")
        request = urllib.request.Request(
            self.runtime.url + "/tokenize",
            data=json.dumps({"content": reply, "add_special": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            metrics.data["text_output_tokens"] = len(json.load(response)["tokens"])
        return reply.strip(), np.concatenate(chunks) if chunks else None, rate

    def turn(
        self,
        operation,
        input_path=None,
        text=None,
        output=None,
        reference=None,
        verify_tts=False,
        **kwargs,
    ):
        metrics = Metrics(operation, self.config, self.event_sink)
        metrics.data.update(
            engine="llama.cpp b11960",
            device=self.runtime.device,
            quantization="Q4_K_M backbone + Q8_0 projector",
            model="LiquidAI/LFM3-Audio-2B-GGUF",
            quality={"naturalness_mos": None, "emotion_score": None},
        )
        try:
            if not self.runtime.process or self.runtime.process.poll() is not None:
                with metrics.stage("load_native", self.runtime):
                    self.runtime.start()
            if operation == "warmup":
                if self.cancel_event and self.cancel_event.is_set():
                    raise PipelineError("Model preparation interrupted.")
                metrics.finish()
                return metrics.data
            if input_path:
                text = self._transcribe(input_path, metrics)
                if reference:
                    metrics.data["quality"]["asr"] = error_rates(reference, text)
            text = (text or "").strip()
            if not text:
                raise PipelineError("No speech/text received.")
            metrics.data["transcript"] = text
            metrics.emit("transcript", text=text)
            if operation == "transcribe":
                metrics.data["result"] = text
            else:
                if operation == "speak":
                    raise PipelineError(
                        "Read-aloud mode uses the Python engine for now. Native conversation streams the model's answer."
                    )
                reply, waveform, rate = self._conversation(
                    text, metrics, operation != "ask"
                )
                metrics.data["reply"] = reply
                metrics.emit("reply", text=reply)
                if waveform is not None:
                    save_wave(output, waveform, rate)
                    metrics.data["output"] = str(Path(output).resolve())
                if verify_tts and waveform is not None:
                    verification_metrics = Metrics("tts_verification", self.config)
                    with metrics.stage("tts_verification", self.runtime):
                        verification = self._transcribe(output, verification_metrics)
                        verification_metrics.finish()
                        metrics.data["verification"] = verification_metrics.data
                        metrics.data["quality"]["tts_asr_proxy"] = {
                            **error_rates(reply, verification),
                            "transcript": verification,
                        }
                self.history.extend(
                    [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": reply},
                    ]
                )
                self.history = (
                    self.history[-self.config.history_turns * 2 :]
                    if self.config.history_turns
                    else []
                )
            metrics.finish()
            return metrics.data
        except BaseException as exc:
            metrics.finish("error", exc)
            raise
        finally:
            metrics.data["cold_in_process"] = "load_native" in metrics.data["stages"]
            try:
                metrics.append(self.config.metrics_file)
            except (OSError, ValueError) as exc:
                print(f"Could not save metrics: {exc}", file=sys.stderr)
