import sys
import tempfile
import time
import uuid
from pathlib import Path

from . import audio
from .config import PipelineError
from .metrics import Metrics, error_rates
from .providers import (
    IndicASR,
    LiquidAudio,
    MioTTS,
    OllamaLLM,
    OmniTTS,
    Runtime,
    TransformerLLM,
    speech_chunks,
)


class VoicePipeline:
    def __init__(self, config, event_sink=None):
        self.config = config
        self.event_sink = event_sink
        self.runtime = Runtime(config)
        self.liquid = LiquidAudio(self.runtime)
        self.asr = self.liquid if config.profile == "liquid" else IndicASR(self.runtime)
        self.llm = (
            OllamaLLM(config)
            if config.llm_backend == "ollama"
            else TransformerLLM(self.runtime)
        )
        self.tts = {
            "liquid": lambda: self.liquid,
            "omnivoice": lambda: OmniTTS(self.runtime),
            "indic-mio": lambda: MioTTS(self.runtime),
        }[config.tts_backend]()
        self.history = []
        self.cancel_event = None

    def _check_cancel(self):
        if self.cancel_event and self.cancel_event.is_set():
            raise PipelineError("Reply interrupted.")

    def close(self):
        self.runtime.release()

    def _messages(self, text):
        language = "Urdu in Urdu script" if self.config.language == "ur" else "English"
        system = f"You are a helpful voice assistant. Reply in {language}. Answer in two or three complete sentences, at most 60 words, unless the user explicitly asks for more detail. Finish the last sentence. Use natural speech. Do not use headings, markdown, bullet lists, code or hidden reasoning. Do not read emotion tags aloud."
        history = (
            self.history[-self.config.history_turns * 2 :]
            if self.config.history_turns
            else []
        )
        return [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": text},
        ]

    def _release_if_requested(self):
        if self.config.release_between_stages:
            self.runtime.release()

    def _speak(self, text, metrics, response_start):
        import numpy as np

        # Loading and voice conditioning are not counted as model inference.
        if isinstance(self.tts, (OmniTTS, MioTTS)):
            self.tts.prepare_voice(metrics)
        else:
            self.tts.load(metrics)
        waveforms, rate, inference, playback = [], None, 0.0, 0.0
        chunks = speech_chunks(text, self.config.chunk_chars)
        with metrics.stage("tts", self.runtime) as row:
            stage_start = time.perf_counter()
            row["first_audio_s"] = None
            for index, chunk in enumerate(chunks):
                self._check_cancel()
                from .text import speech_text

                rendered = speech_text(chunk)
                if index == 0 and self.config.expression != "neutral":
                    rendered = f"[{self.config.expression}] {chunk}"
                self.runtime.sync()
                started = time.perf_counter()
                waveform, chunk_rate = self.tts.synthesize(rendered, metrics)
                self._check_cancel()
                self.runtime.sync()
                inference += time.perf_counter() - started
                waveform = np.asarray(waveform, dtype="float32").reshape(-1)
                if not waveform.size or not np.isfinite(waveform).all():
                    raise PipelineError("TTS returned empty or invalid audio.")
                if float(np.max(np.abs(waveform))) < 1e-6:
                    raise PipelineError("TTS produced a silent waveform.")
                if rate is not None and rate != chunk_rate:
                    raise PipelineError("TTS sample rate changed between chunks.")
                rate = chunk_rate
                if row["first_audio_s"] is None:
                    row["first_audio_s"] = time.perf_counter() - stage_start
                    metrics.data["response_first_audio_s"] = (
                        time.perf_counter() - response_start
                    )
                    metrics.emit(
                        "audio_ready",
                        response_first_audio_s=metrics.data["response_first_audio_s"],
                    )
                    metrics.data["first_audio_definition"] = (
                        "First complete sentence waveform ready; excludes hardware/DAC delay. Playback starts here only with --play."
                    )
                if waveforms:
                    waveforms.append(np.zeros(round(rate * 0.08), dtype="float32"))
                waveforms.append(waveform)
                if metrics.event_sink:
                    import base64

                    pcm = (np.clip(waveform, -1, 1) * 32767).astype("<i2").tobytes()
                    metrics.emit(
                        "audio_chunk",
                        pcm=base64.b64encode(pcm).decode(),
                        sample_rate=rate,
                    )
                if self.config.play:
                    began_playback = time.perf_counter()
                    audio.play(waveform, rate, self.config.output_device)
                    playback += time.perf_counter() - began_playback
            combined = np.concatenate(waveforms)
            duration = len(combined) / rate
            row.update(
                chunks=len(chunks),
                audio_seconds=duration,
                model_inference_s=inference,
                blocking_playback_s=playback,
                rtf=inference / duration,
                realtime_multiple=duration / inference if inference else None,
                sample_rate=rate,
            )
            metrics.data["audio_levels"] = {
                "peak": float(np.max(np.abs(combined))),
                "rms": float(np.sqrt(np.mean(combined**2))),
                "clipping_fraction_before_save": float(
                    np.mean(np.abs(combined) >= 1.0)
                ),
            }
            metrics.data["quality"].update(
                emotion_target=self.config.emotion,
                vocal_event=self.config.expression,
                emotion_score=None,
                naturalness_mos=None,
                note="Emotion adherence, naturalness and speaker similarity require listening tests; no automatic scores are inferred from latency.",
            )
        return combined, rate

    def turn(
        self,
        operation,
        input_path=None,
        text=None,
        output=None,
        reference=None,
        microphone=False,
        verify_tts=False,
        force=False,
    ):
        metrics = Metrics(operation, self.config, self.event_sink)
        metrics.cancel_event = self.cancel_event
        self.runtime.torch.manual_seed(self.config.seed)
        metrics.data["seed_scope"] = "reseeded at the start of each turn"
        metrics.data.update(
            device=self.runtime.device,
            dtype=str(self.runtime.dtype),
            run_options={
                "verify_tts": verify_tts,
                "input_kind": "microphone"
                if microphone
                else "file"
                if input_path
                else "text",
            },
        )
        response_start = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="stts-") as scratch:
                scratch = Path(scratch)
                if microphone or input_path:
                    if microphone:
                        with metrics.stage("capture") as capture:
                            samples, rate = audio.record(
                                self.config, diagnostics=capture
                            )
                        response_start = time.perf_counter()
                    else:
                        with metrics.stage("input_prepare"):
                            samples, rate = audio.read_audio(
                                input_path, self.config.max_input_seconds
                            )
                    duration = len(samples) / rate
                    source = audio.save_wave(scratch / "input.wav", samples, rate)
                    metrics.data["input_audio_seconds"] = duration
                    text = self.asr.transcribe(source, metrics)
                    self._check_cancel()
                    metrics.data["stages"]["asr"]["rtf"] = (
                        metrics.data["stages"]["asr"]["wall_s"] / duration
                    )
                    if not text:
                        raise PipelineError("ASR produced an empty transcript.")
                    if reference is not None:
                        metrics.data["quality"]["asr"] = error_rates(reference, text)
                    self._release_if_requested()
                text = (text or "").strip()
                if not text:
                    raise PipelineError("Provide nonempty text or an audio recording.")
                metrics.data["transcript" if operation != "speak" else "text"] = text
                metrics.emit("transcript", text=text)
                if operation == "transcribe":
                    metrics.data["result"] = text
                else:
                    self._check_cancel()
                    reply = (
                        text
                        if operation == "speak"
                        else self.llm.answer(self._messages(text), metrics)
                    )
                    self._check_cancel()
                    if not reply:
                        raise PipelineError(
                            "LLM returned no speakable answer. Increase --max-new-tokens if reasoning used the budget."
                        )
                    metrics.data["reply"] = reply
                    metrics.emit("reply", text=reply)
                    if operation != "speak":
                        metrics.data["quality"]["llm"] = {
                            "correctness_score": None,
                            "note": "Token throughput and latency are measured. Factual correctness and relevance require a task-specific evaluation or human review.",
                        }
                    self._release_if_requested()
                    if operation != "ask":
                        waveform, rate = self._speak(reply, metrics, response_start)
                        target = (
                            Path(output)
                            if output
                            else Path("outputs") / f"reply-{uuid.uuid4().hex[:12]}.wav"
                        )
                        with metrics.stage("save_audio"):
                            audio.save_wave(target, waveform, rate, force)
                        metrics.data["output"] = str(target.resolve())
                        if verify_tts:
                            self._verify(waveform, rate, reply, scratch, metrics)
                    if operation not in ("speak", "transcribe"):
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
            metrics.data["cold_in_process"] = any(
                k.startswith("load_") for k in metrics.data["stages"]
            ) or metrics.data.get("verification", {}).get("cold_in_process", False)
            try:
                metrics.append(self.config.metrics_file)
            except (OSError, ValueError) as exc:
                print(f"Could not save metrics: {exc}", file=sys.stderr)

    def _verify(self, waveform, rate, expected, scratch, metrics):
        # Opt-in: this is an ASR-based proxy, not an independent TTS quality score.
        verification = Metrics("tts_verification", self.config)
        self._release_if_requested()
        try:
            with metrics.stage("tts_verification", self.runtime):
                path = audio.save_wave(scratch / "generated.wav", waveform, rate)
                samples, sample_rate = audio.read_audio(path, 30)
                normalized = audio.save_wave(
                    scratch / "generated-16k.wav", samples, sample_rate
                )
                transcript = self.asr.transcribe(normalized, verification)
                rates = error_rates(expected, transcript)
            verification.finish()
        except BaseException as exc:
            verification.finish("error", exc)
            raise
        finally:
            verification.data["cold_in_process"] = any(
                k.startswith("load_") for k in verification.data["stages"]
            )
            metrics.data["verification"] = verification.data
        metrics.data["quality"]["tts_asr_proxy"] = {
            **rates,
            "transcript": transcript,
            "asr_profile": self.config.profile,
            "note": "Includes recognizer errors; vocal events can also increase this proxy. Verification supports outputs up to 30 seconds.",
        }
