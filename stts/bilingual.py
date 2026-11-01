"""English/Urdu conversation: language-conditioned ASR, shared text model and voice."""

import base64
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .audio import read_audio, save_wave
from .config import PipelineError
from .metrics import Metrics, error_rates
from .native import MODEL, ROOT, NativeRuntime
from .providers import speech_chunks
from .text import complete_prefix, speech_text
from .transport import event_lines

BASE = ROOT / "models/bilingual"
TEXT_MODEL = MODEL
VOICE_MODEL = BASE / "tts/OmniVoice-GGUF/omnivoice-q8_0.gguf"
TRANSLATOR_MODEL = BASE / "translator-ct2/model.bin"
VOICE_BINARY = ROOT / "native/audio-v0.9.3/audiocpp_server"
SPOKEN_REPLY_GRAMMAR = r"""root ::= sentence (" " sentence){0,2}
sentence ::= [A-Za-z0-9] [^.!?\n\r*#;]* [.!?]"""


def paths():
    tokenizers = list((BASE / "asr-flex").glob("**/tokenizer.json"))
    if not tokenizers:
        return None
    folder = tokenizers[0].parent
    encoders = list(folder.glob("encoder*.onnx"))
    decoders = list(folder.glob("decoder*.onnx"))
    if not encoders or not decoders:
        return None
    if not all(
        Path(str(path) + ".data").is_file() for path in (encoders[0], decoders[0])
    ):
        return None
    return folder, encoders[0], decoders[0]


def available():
    return bool(paths()) and all(
        p.is_file()
        for p in (
            TEXT_MODEL,
            VOICE_MODEL,
            VOICE_BINARY,
            TRANSLATOR_MODEL,
            BASE / "translator/tokenizer.json",
        )
    )


def text_language(text, fallback="en"):
    arabic = sum("\u0600" <= char <= "\u06ff" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    return "ur" if arabic else "en" if latin else fallback


def greeting_reply(text, language):
    """Answer only an entire conventional greeting; never swallow a question."""
    normalized = "".join(char for char in text if char.isalpha())
    if language == "ur" and normalized in {
        "السلامعلیکم",
        "اسلامعلیکم",
        "سلامعلیکم",
        "السلامعلیکمآپکیسےہیں",
        "اسلامعلیکمآپکیسےہیں",
    }:
        return "وعلیکم السلام، میں حاضر ہوں۔ بتائیے۔"
    return None


class FlexRecognizer:
    """Limit language identification to the two languages requested by the user."""

    def __init__(self, threads=4):
        import onnxruntime as ort

        layout = paths()
        if not layout:
            raise PipelineError(
                "The bilingual speech recognizer is not installed. Run python setup_bilingual.py."
            )
        folder, encoder, decoder = layout
        self.tokens = json.loads((folder / "tokenizer.json").read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(
            str(encoder), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            str(decoder), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def transcribe(self, samples, language="auto", cancel=None):
        if len(samples) < 16000:
            padding = 16000 - len(samples)
            samples = np.pad(samples, (padding // 2, padding - padding // 2))
        cross = self.encoder.run(
            None, {"audio": samples.reshape(1, -1).astype(np.float32)}
        )
        cross_names = [
            value.name
            for value in self.decoder.get_inputs()
            if value.name.startswith("cross_")
        ]
        cross_values = dict(zip(cross_names, cross))
        past_k = np.empty((0, 24, 8, 128), dtype=np.float32)
        past_v = past_k.copy()

        def step(ids):
            nonlocal past_k, past_v
            if cancel and cancel.is_set():
                raise PipelineError("Reply interrupted.")
            result = self.decoder.run(
                None,
                {
                    "input_ids": np.asarray([ids], dtype=np.int64),
                    "past_k": past_k,
                    "past_v": past_v,
                    **cross_values,
                },
            )
            logits, new_k, new_v = result
            past_k = np.concatenate((past_k, new_k), axis=0)
            past_v = np.concatenate((past_v, new_v), axis=0)
            return logits[0]

        confidence = None
        if language == "auto":
            scores = step([7, 4, 18])
            choices = {
                lang: self.tokens["language_tokens"][lang] for lang in ("en", "ur")
            }
            language = max(choices, key=lambda key: float(scores[choices[key]]))
            pair = np.asarray(
                [scores[choices[key]] for key in choices], dtype=np.float64
            )
            pair = np.exp(pair - pair.max())
            confidence = float(pair.max() / pair.sum())
            prompt = list(self.tokens["prompts"][language])
            prompt[6] = self.tokens["itn_id"]
            logits = step(prompt[3:])
        else:
            prompt = list(self.tokens["prompts"][language])
            prompt[6] = self.tokens["itn_id"]
            logits = step(prompt)
        output = []
        for _ in range(512):
            token = int(np.argmax(logits))
            if token in (self.tokens["eos_id"], self.tokens["pad_id"]):
                break
            output.append(token)
            logits = step([token])
        else:
            raise PipelineError(
                "Speech recognition reached its token limit. Try a shorter utterance."
            )
        text = (
            "".join(
                self.tokens["pieces"][token]
                for token in output
                if token >= self.tokens["special_count"]
            )
            .replace("▁", " ")
            .strip()
        )
        return text, language, confidence


class TranslationBridge:
    def __init__(self, threads=4):
        import ctranslate2
        from transformers import AutoTokenizer

        self.model = ctranslate2.Translator(
            str(BASE / "translator-ct2"),
            device="cpu",
            compute_type="int8",
            inter_threads=1,
            intra_threads=threads,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE / "translator", local_files_only=True
        )

    def translate(self, text, source, target, cancel=None):
        self.tokenizer.src_lang = "urd_Arab" if source == "ur" else "eng_Latn"
        target = "urd_Arab" if target == "ur" else "eng_Latn"
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        if len(tokens) > 512:
            raise PipelineError(
                "This utterance is too long to translate reliably. Please use a shorter turn."
            )
        results = self.model.translate_batch(
            [tokens],
            target_prefix=[[target]],
            beam_size=1,
            max_decoding_length=256,
            callback=lambda step: bool(cancel and cancel.is_set()),
        )
        if cancel and cancel.is_set():
            raise PipelineError("Reply interrupted.")
        output = results[0].hypotheses[0][1:]
        if len(output) >= 255:
            raise PipelineError(
                "Translation reached its limit; please use a shorter turn."
            )
        result = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(output), skip_special_tokens=True
        ).strip()
        if not result:
            raise PipelineError("Translation returned no text. Please repeat.")
        return result


class VoiceRuntime:
    def __init__(self, config):
        self.config = config
        self.device = NativeRuntime(config).device
        self.process = None
        self.url = None
        self.log = None

    def start(self):
        if self.process and self.process.poll() is None:
            return
        if not VOICE_BINARY.is_file() or not VOICE_MODEL.is_file():
            raise PipelineError("The bilingual voice runtime is not installed.")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        spec = ROOT / "outputs/bilingual-voice-server.json"
        spec.parent.mkdir(exist_ok=True)
        spec.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "id": "voice",
                            "family": "omnivoice",
                            "path": str(VOICE_MODEL),
                            "task": "tts",
                            "mode": "streaming",
                        }
                    ]
                }
            )
        )
        self.log = (ROOT / "outputs/bilingual-voice.log").open("w")
        self.process = subprocess.Popen(
            [
                str(VOICE_BINARY),
                "--config",
                str(spec),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--backend",
                self.device,
                "--threads",
                str(self.config.threads),
                "--no-ui",
            ],
            cwd=VOICE_BINARY.parent,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise PipelineError(
                    "Bilingual voice failed to start. See outputs/bilingual-voice.log."
                )
            try:
                if (
                    urllib.request.urlopen(self.url + "/health", timeout=1).status
                    == 200
                ):
                    return
            except OSError:
                time.sleep(0.1)
        self.close()
        raise PipelineError("Bilingual voice startup timed out.")

    def synthesize(self, text, language, cancel=None):
        data = {
            "model": "voice",
            "input": speech_text(text),
            "response_format": "pcm",
            "stream_format": "sse",
            "options": {
                "language": "Urdu" if language == "ur" else "English",
                "num_inference_steps": self.config.omni_steps,
                "speed": self.config.speed,
                "seed": 42,
            },
        }
        if self.config.reference_audio:
            data.update(
                voice_ref={"type": "path", "path": self.config.reference_audio},
                reference_text=self.config.reference_text,
            )
        elif self.config.voice_description:
            data["options"]["instruct"] = self.config.voice_description
        request = urllib.request.Request(
            self.url + "/v1/audio/speech",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )

        def check():
            if cancel and cancel.is_set():
                raise PipelineError("Reply interrupted.")

        completed = False
        for line in event_lines(request, check, self.abort):
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                completed = True
                break
            event = json.loads(payload)
            if event.get("type") == "speech.audio.delta":
                yield base64.b64decode(event["audio"])
            elif "error" in event:
                raise PipelineError(str(event["error"]))
        if not completed:
            raise PipelineError("Speech stream ended before completion. Please retry.")

    def abort(self):
        self.close(grace_seconds=0.2)

    def close(self, grace_seconds=5):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log:
            self.log.close()
        self.process = None


class BilingualPipeline:
    def __init__(self, config, event_sink=None):
        self.config = config
        self.event_sink = event_sink
        self.history = []
        self.cancel_event = None
        self.last_language = "en"
        self.runtime = NativeRuntime(config, model_path=TEXT_MODEL, projector_path=None)
        self.voice = VoiceRuntime(config)
        self.asr = None
        self.translator = None

    def close(self):
        self.runtime.close()
        self.voice.close()
        self.asr = None
        self.translator = None

    def _check(self):
        if getattr(self, "_speech_error", None):
            raise PipelineError(f"Speech synthesis failed: {self._speech_error}")
        if self.cancel_event and self.cancel_event.is_set():
            raise PipelineError("Reply interrupted.")

    def _answer(self, text, language, metrics, sentence_sink=None, stream_visible=True):
        system = "Have a natural conversation. Always produce English in this internal stage, even if the user requests another language. A separate translation stage handles Urdu. Do not write Urdu here. If asked who you are, say you are a local voice assistant. The user's name and personal facts belong to the user, not to you. Acknowledge introductions; never repeat their introduction as your own identity. Answer directly in one or two brief complete sentences, ideally under40words total. No introductions, markdown or bullet lists. If something is unclear, ask rather than inventing details. Don't invent personal facts. Don't repeat the question or routinely offer more help."
        messages = [
            {"role": "system", "content": system},
            *self.history,
            {"role": "user", "content": text},
        ]
        request = urllib.request.Request(
            self.runtime.url + "/v1/chat/completions",
            data=json.dumps(
                {
                    "messages": messages,
                    "stream": True,
                    "max_tokens": self.config.max_new_tokens,
                    "temperature": self.config.temperature,
                    "seed": self.config.seed,
                    "grammar": SPOKEN_REPLY_GRAMMAR,
                    "top_p": 0.8,
                    "top_k": 20,
                    "min_p": 0,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        reply = ""
        reason = None
        spoken_end = 0
        with metrics.stage("llm", self.runtime) as row:
            started = time.perf_counter()
            for line in event_lines(request, self._check, self.runtime.abort):
                self._check()
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                data = json.loads(payload)
                if "error" in data:
                    raise PipelineError(str(data["error"]))
                usage = data.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    row["output_tokens"] = usage["completion_tokens"]
                if not data.get("choices"):
                    continue
                choice = data["choices"][0]
                delta = choice.get("delta", {}).get("content", "")
                if delta:
                    if "ttft_s" not in row:
                        row["ttft_s"] = time.perf_counter() - started
                    reply += delta
                    if stream_visible:
                        metrics.emit("reply_delta", text=reply)
                    if sentence_sink:
                        ready = complete_prefix(reply[spoken_end:])
                        if ready and len(ready) >= 32:
                            for sentence in speech_chunks(ready, 140):
                                sentence_sink(sentence)
                            spoken_end = reply.rfind(ready) + len(ready)
                reason = choice.get("finish_reason") or reason
                usage = data.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    row["output_tokens"] = usage["completion_tokens"]
            if reason not in ("stop", "length"):
                raise PipelineError(
                    "Reply stream ended before completion. Please retry."
                )
            row["finish_reason"] = reason
            if reason == "length":
                metrics.data["reply_truncated"] = True
                reply = complete_prefix(reply)
            row["tokens_per_s"] = (
                row["output_tokens"] / (time.perf_counter() - started)
                if row.get("output_tokens")
                else None
            )
        if not reply.strip():
            raise PipelineError(
                "No complete reply was generated. Please repeat or increase the token budget."
            )
        if sentence_sink and reply[spoken_end:].strip():
            sentence_sink(reply[spoken_end:].strip())
        metrics.data["text_output_tokens"] = row.get("output_tokens")
        return reply.strip()

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
        import sys

        self._speech_error = None
        metrics = Metrics(operation, self.config, self.event_sink)
        metrics.data.update(
            engine="bilingual local",
            device=self.runtime.device,
            providers={
                "asr": {
                    "backend": "onnxruntime",
                    "device": "cpu",
                    "quantization": "int8",
                    "model": "indic-transcribe-flex",
                    "parameters": 1222553584,
                },
                "translation": {
                    "backend": "ctranslate2",
                    "device": "cpu",
                    "quantization": "int8",
                    "model": "nllb-200-distilled-600M",
                },
                "llm": {
                    "backend": "llama.cpp",
                    "device": self.runtime.device,
                    "quantization": "Q4_K_M",
                    "model": "LFM3-Audio-2B text backbone",
                    "reply_format": "one to three complete sentences; no character cap",
                },
                "tts": {
                    "backend": "audio.cpp",
                    "device": self.voice.device,
                    "quantization": "Q8_0",
                    "model": "OmniVoice",
                },
            },
            quality={"naturalness_mos": None, "emotion_score": None},
            run_options={
                "input_kind": "audio" if input_path else "text",
                "verify_tts": verify_tts,
            },
        )
        try:
            if (
                self.asr is None
                or self.translator is None
                or not self.runtime.process
                or self.runtime.process.poll() is not None
                or not self.voice.process
                or self.voice.process.poll() is not None
            ):
                try:
                    with metrics.stage("load_bilingual", self.runtime):
                        if self.asr is None:
                            self.asr = FlexRecognizer(self.config.threads)
                        if self.translator is None:
                            self.translator = TranslationBridge(self.config.threads)
                        self.runtime.start()
                        self.voice.start()
                except BaseException:
                    self.close()
                    raise
            self._check()
            if operation == "warmup":
                metrics.finish()
                return metrics.data
            language = self.config.language
            if input_path:
                samples, rate = read_audio(input_path, self.config.max_input_seconds)
                metrics.data["input_audio_seconds"] = len(samples) / rate
                with metrics.stage("asr", self.runtime) as row:
                    text, language, confidence = self.asr.transcribe(
                        samples, language, self.cancel_event
                    )
                    row["language_confidence_two_candidates"] = confidence
                row["rtf"] = row["wall_s"] / (len(samples) / rate)
                if reference:
                    metrics.data["quality"]["asr"] = error_rates(reference, text)
            elif language == "auto":
                language = text_language(text, self.last_language)
            if not (text or "").strip():
                raise PipelineError("No speech detected. Please repeat.")
            self.last_language = language
            metrics.data.update(transcript=text, detected_language=language)
            metrics.emit("transcript", text=text)
            if operation == "transcribe":
                metrics.data["result"] = text
            else:
                llm_input = text
                if language == "ur" and operation != "speak":
                    with metrics.stage("translate_input", self.runtime):
                        llm_input = self.translator.translate(
                            text, "ur", "en", self.cancel_event
                        )
                    metrics.data["llm_input_translation"] = llm_input
                audio_parts = []
                inference = []
                futures = []
                localized = []

                direct_reply = (
                    greeting_reply(text, language) if operation != "speak" else None
                )

                def render(sentence):
                    self._check()
                    if language == "ur" and operation != "speak" and not direct_reply:
                        with metrics.stage("translate_reply", self.runtime):
                            sentence = self.translator.translate(
                                sentence, "en", "ur", self.cancel_event
                            )
                    localized.append(sentence)
                    metrics.emit("reply_delta", text=" ".join(localized))
                    began = time.perf_counter()
                    with metrics.stage("tts_chunk", self.runtime):
                        for pcm in self.voice.synthesize(
                            sentence, language, self.cancel_event
                        ):
                            self._check()
                            if not pcm:
                                continue
                            if "response_first_audio_s" not in metrics.data:
                                metrics.data["response_first_audio_s"] = (
                                    time.perf_counter() - metrics.started
                                )
                                metrics.emit(
                                    "audio_ready",
                                    response_first_audio_s=metrics.data[
                                        "response_first_audio_s"
                                    ],
                                )
                            audio_parts.append(pcm)
                            metrics.emit(
                                "audio_chunk",
                                pcm=base64.b64encode(pcm).decode(),
                                sample_rate=24000,
                            )
                    inference.append(time.perf_counter() - began)

                with ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="bilingual-speech"
                ) as executor:

                    def failed(future):
                        if future.exception():
                            self._speech_error = future.exception()

                    def enqueue(sentence):
                        future = executor.submit(render, sentence)
                        future.add_done_callback(failed)
                        futures.append(future)

                    if direct_reply:
                        reply = direct_reply
                        metrics.data["reply_source"] = "conventional_greeting"
                        if operation != "ask":
                            enqueue(reply)
                    elif operation == "speak":
                        reply = text
                        for sentence in speech_chunks(reply, 140):
                            enqueue(sentence)
                    else:
                        reply = self._answer(
                            llm_input,
                            "en",
                            metrics,
                            enqueue if operation != "ask" else None,
                            stream_visible=language != "ur",
                        )
                    english_reply = (
                        "Hello. I'm here, go ahead." if direct_reply else reply
                    )
                    for future in futures:
                        future.result()
                    if language == "ur":
                        if operation == "ask" and not direct_reply:
                            with metrics.stage("translate_reply", self.runtime):
                                reply = self.translator.translate(
                                    reply, "en", "ur", self.cancel_event
                                )
                        elif operation != "ask":
                            reply = " ".join(localized)
                    metrics.data["reply"] = reply
                    metrics.emit("reply", text=reply)
                if operation != "ask":
                    if not audio_parts:
                        raise PipelineError("No speech was generated.")
                    pcm = b"".join(audio_parts)
                    waveform = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768
                    row = {
                        "status": "ok",
                        "stage_name": "tts",
                        "wall_s": sum(inference),
                        "model_inference_s": sum(inference),
                        "chunks": len(inference),
                        "audio_seconds": len(waveform) / 24000,
                        "rtf": sum(inference) / (len(waveform) / 24000),
                    }
                    metrics.data["stages"]["tts"] = row
                    metrics.emit("stage", name="tts", row=dict(row))
                    save_wave(output, waveform, 24000)
                    metrics.data["output"] = str(Path(output).resolve())
                    if verify_tts:
                        # Verification is a separate transcription, so it cannot
                        # overwrite the caller's transcript or input duration.
                        with metrics.stage("verify_tts", self.runtime):
                            samples, _ = read_audio(output, max_seconds=120)
                            transcripts = []
                            for offset in range(0, len(samples), 25 * 16000):
                                part, _, _ = self.asr.transcribe(
                                    samples[offset : offset + 25 * 16000],
                                    language,
                                    self.cancel_event,
                                )
                                transcripts.append(part)
                            verified = " ".join(transcripts)
                        metrics.data["quality"]["tts_transcription"] = verified
                        metrics.data["quality"]["tts_asr_proxy"] = error_rates(
                            speech_text(reply), verified
                        )
                if operation not in ("speak", "transcribe"):
                    self.history.extend(
                        [
                            {"role": "user", "content": llm_input},
                            {"role": "assistant", "content": english_reply},
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
            metrics.data["cold_in_process"] = "load_bilingual" in metrics.data["stages"]
            try:
                metrics.append(self.config.metrics_file)
            except (OSError, ValueError) as exc:
                print(f"Could not save metrics: {exc}", file=sys.stderr)
