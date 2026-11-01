"""Model adapters. Heavy imports and model downloads happen only on first use."""

import gc
import importlib.util
import inspect
import json
import re
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from .config import PipelineError


@contextmanager
def offline_torch_downloads(runtime):
    """Torch audio frontends have an auxiliary cache separate from Hugging Face."""
    if not runtime.config.offline:
        yield
        return
    original = runtime.torch.hub.download_url_to_file

    def blocked(*args, **kwargs):
        raise PipelineError(
            "Offline mode: an auxiliary Torch audio checkpoint is not cached. Run this backend once online to populate the Torch cache."
        )

    runtime.torch.hub.download_url_to_file = blocked
    try:
        yield
    finally:
        runtime.torch.hub.download_url_to_file = original


def spoken_text(text):
    """Never send hidden reasoning (including a truncated reasoning span) to TTS."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


def speech_chunks(text, limit):
    """Keep sentence boundaries where possible; never drop text at hard limits."""
    sentences = re.split(r"(?<=[.!?۔؟])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit + 1)
            if cut <= 0:
                cut = limit
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if len(current) + len(sentence) + bool(current) > limit:
            chunks.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


class Runtime:
    def __init__(self, config):
        import torch

        self.torch, self.config, self.cache = torch, config, {}
        torch.set_num_threads(config.threads)
        torch.manual_seed(config.seed)
        if config.device == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = config.device
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise PipelineError(
                "MPS is unavailable. Use native arm64 Python on your M4, or --device cpu."
            )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise PipelineError("CUDA is unavailable. Use --device auto or cpu.")
        dtype = config.dtype
        if dtype == "auto":
            dtype = (
                "float16"
                if self.device == "mps"
                else "bfloat16"
                if self.device == "cuda" and torch.cuda.is_bf16_supported()
                else "float32"
            )
        self.dtype = getattr(torch, dtype)

    def sync(self):
        if self.device == "mps":
            self.torch.mps.synchronize()
        elif self.device == "cuda":
            self.torch.cuda.synchronize()

    def memory(self):
        if self.device == "mps":
            return {
                "mps_allocated_end_mb": self.torch.mps.current_allocated_memory()
                / 2**20,
                "mps_driver_end_mb": self.torch.mps.driver_allocated_memory() / 2**20,
            }
        if self.device == "cuda":
            return {
                "cuda_allocated_end_mb": self.torch.cuda.memory_allocated() / 2**20,
                "cuda_peak_allocated_mb": self.torch.cuda.max_memory_allocated()
                / 2**20,
            }
        return {}

    def get(self, key, loader, metrics):
        if key in self.cache:
            metrics.data.setdefault("cache_hits", []).append(key)
            return self.cache[key]
        print(
            f"Loading {key} on {self.device} ({str(self.dtype).split('.')[-1]})…",
            file=sys.stderr,
            flush=True,
        )
        with metrics.stage(f"load_{key}", self) as row:
            value = loader()
            self.cache[key] = value
            row["cold_in_process"] = True
            model = value[0] if isinstance(value, tuple) else value
            if not hasattr(model, "parameters") and hasattr(model, "model"):
                model = model.model
            if hasattr(model, "parameters"):
                row["loaded_parameters"] = sum(p.numel() for p in model.parameters())
        return value

    def release(self):
        self.cache.clear()
        gc.collect()
        if self.device == "mps":
            self.torch.mps.empty_cache()
        elif self.device == "cuda":
            self.torch.cuda.empty_cache()

    def hub_options(self):
        return {"trust_remote_code": True, "local_files_only": self.config.offline}


class CancellationCheck:
    def __init__(self, metrics):
        self.metrics = metrics

    def put(self, value):
        event = getattr(self.metrics, "cancel_event", None)
        if event and event.is_set():
            raise PipelineError("Reply interrupted.")

    def end(self):
        pass


class TokenClock:
    """Transformers streamer protocol, measuring raw-token latency, not speech."""

    def __init__(self, runtime, row=None, metrics=None):
        self.runtime = runtime
        self.row = row
        self.tokens = 0
        self.metrics = metrics
        self.started = time.perf_counter()
        self.prompt_seen = False
        self.first = None

    def put(self, value):
        if self.metrics:
            CancellationCheck(self.metrics).put(value)
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        if self.first is None and value.numel():
            self.runtime.sync()
            self.first = time.perf_counter() - self.started
        self.tokens += value.numel()
        if self.row is not None:
            elapsed = time.perf_counter() - self.started
            self.row.update(
                ttft_s=self.first,
                output_tokens=self.tokens,
                tokens_per_s=self.tokens / elapsed if elapsed else None,
            )

    def end(self):
        pass


class LiquidAudio:
    def __init__(self, runtime):
        self.rt, self.cfg = runtime, runtime.config

    def load(self, metrics):
        def loader():
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            model = (
                AutoModelForMultimodalLM.from_pretrained(
                    self.cfg.audio_model, dtype=self.rt.dtype, **self.rt.hub_options()
                )
                .to(self.rt.device)
                .eval()
            )
            processor = AutoProcessor.from_pretrained(
                self.cfg.audio_model, **self.rt.hub_options()
            )
            # The codec is a separate model owned by the processor. Keep its
            # default precision, as in the publisher's example.
            processor.audio_tokenizer.to(device=self.rt.device).eval()
            return model, processor

        return self.rt.get("liquid_audio", loader, metrics)

    def transcribe(self, path, metrics):
        model, processor = self.load(metrics)
        with metrics.stage("asr", self.rt) as row:
            messages = [
                {"role": "system", "content": "Perform ASR in English."},
                {"role": "user", "content": [{"type": "audio", "audio": str(path)}]},
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.rt.device)
            with self.rt.torch.inference_mode():
                result = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    suppress_mixed=True,
                    suppress_audio_out=True,
                    return_dict_in_generate=True,
                    streamer=CancellationCheck(metrics),
                )
            generated = result.sequences[0, inputs["input_ids"].shape[1] :]
            row["generated_tokens"] = int(generated.numel())
            if generated.numel() >= 1024:
                raise PipelineError(
                    "ASR hit its token limit; retry a shorter recording."
                )
            return processor.decode(generated, skip_special_tokens=True).strip()

    def synthesize(self, text, metrics):
        model, processor = self.load(metrics)
        messages = [
            {
                "role": "system",
                "content": f"Perform TTS in {self.cfg.locale} using a {self.cfg.gender} voice.",
            },
            {"role": "user", "content": text},
            {"role": "assistant", "content": "<|audio_start|>"},
        ]
        inputs = processor.apply_chat_template(
            messages,
            continue_final_message=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.rt.device)
        with self.rt.torch.inference_mode():
            result = model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_audio_tokens,
                audio_temperature=1.0,
                audio_top_k=128,
                audio_repetition_penalty=1.1,
                audio_repetition_window=4096,
                return_dict_in_generate=True,
                audio_streamer=CancellationCheck(metrics),
            )
            if result.audio_codes is None or result.audio_lengths is None:
                raise PipelineError("Liquid produced no audio codes.")
            length = int(result.audio_lengths[0].item())
            if length <= 0:
                raise PipelineError("Liquid produced no speech frames.")
            if (
                result.sequences.shape[1] - inputs["input_ids"].shape[1]
                >= self.cfg.max_audio_tokens
            ):
                raise PipelineError(
                    "Speech hit --max-audio-tokens; increase it or lower --chunk-chars."
                )
            waveform = processor.audio_tokenizer.decode(
                result.audio_codes[:1, :, :length]
            ).audio_values[0]
        return waveform.detach().float().cpu().numpy().reshape(-1), 24000


class IndicASR:
    def __init__(self, runtime):
        self.rt, self.cfg = runtime, runtime.config

    def load(self, metrics):
        def loader():
            from huggingface_hub import snapshot_download

            # Use the publisher's checkpoint-local adapter. Installing the
            # Bodhan training package would pin an incompatible Transformers.
            directory = Path(
                snapshot_download(
                    self.cfg.indic_model,
                    local_files_only=self.cfg.offline,
                    ignore_patterns=["nemo/*", "*.nemo", "*.onnx", "*.gguf"],
                )
            )
            filename = directory / "indic_transcribe.py"
            if not filename.is_file():
                raise PipelineError(
                    "The Indic checkpoint has no indic_transcribe.py adapter. Check the model revision/access."
                )
            name = "stts_indic_" + directory.name.replace("-", "_")
            spec = importlib.util.spec_from_file_location(name, filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            sys.path.insert(0, str(directory))
            try:
                spec.loader.exec_module(module)
                factory = module.IndicTranscribe.from_pretrained
                parameters = inspect.signature(factory).parameters
                options = {}
                if "device" in parameters:
                    options["device"] = self.rt.device
                elif self.rt.device != "cpu":
                    raise PipelineError(
                        "This Indic wrapper exposes no explicit device control. Try --device cpu; MPS compatibility for this revision is unverified."
                    )
                if "dtype" in parameters:
                    options["dtype"] = self.rt.dtype
                return factory(str(directory), **options)
            finally:
                sys.path.remove(str(directory))

        return self.rt.get("indic_asr", loader, metrics)

    def transcribe(self, path, metrics):
        engine = self.load(metrics)
        with metrics.stage("asr", self.rt):
            options = {"lang": self.cfg.language}
            if self.cfg.mixed_script:
                options["itn"] = True
            if self.cfg.romanized:
                options["romanized"] = True
            value = engine.transcribe(str(path), **options)
        if not isinstance(value, str):
            raise PipelineError("Indic-Transcribe returned no transcript.")
        return value.strip()


class TransformerLLM:
    def __init__(self, runtime):
        self.rt, self.cfg = runtime, runtime.config

    def load(self, metrics):
        def loader():
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForMultimodalLM,
                AutoTokenizer,
            )

            kind = (
                AutoModelForMultimodalLM
                if "Qwen3.5" in self.cfg.llm_model
                else AutoModelForCausalLM
            )
            model = (
                kind.from_pretrained(
                    self.cfg.llm_model, dtype=self.rt.dtype, **self.rt.hub_options()
                )
                .to(self.rt.device)
                .eval()
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.llm_model, **self.rt.hub_options()
            )
            return model, tokenizer

        return self.rt.get("llm", loader, metrics)

    def answer(self, messages, metrics):
        model, tokenizer = self.load(metrics)
        with metrics.stage("llm", self.rt) as row:
            messages = list(messages)
            while True:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.rt.device)
                count = int(inputs["input_ids"].shape[-1])
                if count <= self.cfg.max_context_tokens:
                    break
                if len(messages) <= 2:
                    raise PipelineError(
                        "The current prompt exceeds --max-context-tokens; shorten it or raise the limit."
                    )
                del messages[1:3]  # Drop a whole old user/assistant pair.
            row["input_tokens"] = count
            clock = TokenClock(self.rt, row, metrics)
            options = {"do_sample": self.cfg.temperature > 0, "repetition_penalty": 1.1}
            if self.cfg.temperature > 0:
                options.update(temperature=self.cfg.temperature, top_k=50)
            with self.rt.torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    streamer=clock,
                    **options,
                )
            self.rt.sync()
            elapsed = time.perf_counter() - clock.started
            tokens = output[0, count:]
            row.update(
                ttft_s=clock.first,
                generation_s=elapsed,
                output_tokens=int(tokens.numel()),
                tokens_per_s=tokens.numel() / elapsed,
                token_limit_reached=tokens.numel() >= self.cfg.max_new_tokens,
            )
            reply = spoken_text(tokenizer.decode(tokens, skip_special_tokens=True))
            if row["token_limit_reached"]:
                from .text import complete_prefix

                metrics.data["reply_truncated"] = True
                reply = complete_prefix(reply)
                if not reply:
                    raise PipelineError(
                        "The token limit was reached before a complete sentence. Increase the reply token budget."
                    )
            return reply


class OllamaLLM:
    def __init__(self, config):
        self.cfg = config

    def answer(self, messages, metrics):
        request = urllib.request.Request(
            self.cfg.ollama_url.rstrip("/") + "/api/chat",
            data=json.dumps(
                {
                    "model": self.cfg.llm_model,
                    "messages": messages,
                    "stream": True,
                    "think": False,
                    "keep_alive": "10m",
                    "options": {
                        "num_ctx": self.cfg.max_context_tokens,
                        "num_predict": self.cfg.max_new_tokens,
                        "temperature": self.cfg.temperature,
                        "seed": self.cfg.seed,
                    },
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        chunks, complete = [], False
        with metrics.stage("llm") as row:
            started = time.perf_counter()
            row["ttft_s"] = None
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    for raw in response:
                        CancellationCheck(metrics).put(None)
                        data = json.loads(raw)
                        if data.get("error"):
                            raise PipelineError(str(data["error"]))
                        content = data.get("message", {}).get("content", "")
                        if content:
                            if row["ttft_s"] is None:
                                row["ttft_s"] = time.perf_counter() - started
                            chunks.append(content)
                        if data.get("done"):
                            complete = True
                            evaluation = data.get("eval_duration", 0) / 1e9
                            row.update(
                                output_tokens=data.get("eval_count"),
                                input_tokens=data.get("prompt_eval_count"),
                                server_load_s=data.get("load_duration", 0) / 1e9,
                                tokens_per_s=data.get("eval_count", 0) / evaluation
                                if evaluation
                                else None,
                                server_eval_s=evaluation,
                            )
            except (urllib.error.URLError, TimeoutError) as exc:
                raise PipelineError(
                    f"Ollama request failed: {exc}. Start Ollama and choose an installed model name."
                ) from exc
            if not complete:
                raise PipelineError(
                    "Ollama disconnected before completing its response."
                )
        return spoken_text("".join(chunks))


class OmniTTS:
    def __init__(self, runtime):
        self.rt, self.cfg = runtime, runtime.config

    def load(self, metrics):
        def loader():
            try:
                from omnivoice import OmniVoice
            except ImportError as exc:
                raise PipelineError(
                    "Install requirements-voice.txt to enable OmniVoice."
                ) from exc
            # Explicitly disable its optional automatic reference ASR.
            model_path = "k2-fsa/OmniVoice"
            if self.cfg.offline:
                from huggingface_hub import snapshot_download

                model_path = snapshot_download(model_path, local_files_only=True)
                if not (Path(model_path) / "audio_tokenizer").is_dir():
                    raise PipelineError(
                        "Offline OmniVoice needs its bundled audio_tokenizer cached. Run this backend once online first."
                    )
            return OmniVoice.from_pretrained(
                model_path,
                device_map=self.rt.device,
                dtype=self.rt.dtype,
                load_asr=False,
            )

        return self.rt.get("omni_tts", loader, metrics)

    def prepare_voice(self, metrics):
        model = self.load(metrics)
        if self.cfg.reference_audio and "voice_prompt" not in self.rt.cache:
            with metrics.stage("voice_conditioning", self.rt):
                self.rt.cache["voice_prompt"] = model.create_voice_clone_prompt(
                    ref_audio=self.cfg.reference_audio, ref_text=self.cfg.reference_text
                )
        return model

    def synthesize(self, text, metrics):
        model = self.prepare_voice(metrics)
        options = {
            "text": text,
            "language": self.cfg.language,
            "speed": self.cfg.speed,
            "num_step": self.cfg.omni_steps,
        }
        if self.cfg.reference_audio:
            options["voice_clone_prompt"] = self.rt.cache["voice_prompt"]
        else:
            options["instruct"] = (
                self.cfg.voice_description or f"{self.cfg.gender}, medium pitch"
            )
        with self.rt.torch.inference_mode():
            waveforms = model.generate(**options)
        if not waveforms:
            raise PipelineError("OmniVoice returned no speech.")
        return waveforms[0], int(model.sampling_rate)


class MioTTS:
    """Indic-Mio's named emotion tags plus a reference-conditioned MioCodec.

    The publisher's short model-card example mixes codec APIs/sample rates;
    this adapter uses the actual MioCodecModel API and reads its sample rate.
    The codec conservatively runs on CPU on Apple Silicon; its complex/iSTFT
    operators have version-dependent MPS support.
    """

    def __init__(self, runtime):
        self.rt, self.cfg = runtime, runtime.config

    def load(self, metrics):
        def loader():
            try:
                from miocodec import MioCodecModel
            except ImportError as exc:
                raise PipelineError(
                    "Install requirements-emotion.txt to use Indic-Mio."
                ) from exc
            from transformers import AutoModelForCausalLM, AutoTokenizer

            name = "SPRINGLab/Indic-Mio"
            model = (
                AutoModelForCausalLM.from_pretrained(
                    name, dtype=self.rt.dtype, **self.rt.hub_options()
                )
                .to(self.rt.device)
                .eval()
            )
            tokenizer = AutoTokenizer.from_pretrained(name, **self.rt.hub_options())
            codec_device = "cpu" if self.rt.device == "mps" else self.rt.device
            from huggingface_hub import hf_hub_download

            codec_name = "Aratako/MioCodec-25Hz-44.1kHz-v2"
            config_path = hf_hub_download(
                codec_name, "config.yaml", local_files_only=self.cfg.offline
            )
            weights_path = hf_hub_download(
                codec_name, "model.safetensors", local_files_only=self.cfg.offline
            )
            with offline_torch_downloads(self.rt):
                codec = (
                    MioCodecModel.from_pretrained(
                        config_path=config_path, weights_path=weights_path
                    )
                    .to(codec_device)
                    .eval()
                )
            return model, tokenizer, codec

        return self.rt.get("mio_tts", loader, metrics)

    def prepare_voice(self, metrics):
        model, tokenizer, codec = self.load(metrics)
        if "mio_voice" not in self.rt.cache:
            from miocodec import load_audio

            with metrics.stage("voice_conditioning", self.rt):
                reference = load_audio(
                    self.cfg.reference_audio, sample_rate=codec.config.sample_rate
                ).to(next(codec.parameters()).device)
                with self.rt.torch.inference_mode():
                    self.rt.cache["mio_voice"] = codec.encode(
                        reference, return_content=False, return_global=True
                    ).global_embedding
        return model, tokenizer, codec

    def synthesize(self, text, metrics):
        model, tokenizer, codec = self.prepare_voice(metrics)
        if self.cfg.emotion != "neutral":
            text += f" <{self.cfg.emotion}>"
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.rt.device)
        with self.rt.torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                max_new_tokens=self.cfg.max_audio_tokens,
            )
            tokens = output[0, inputs["input_ids"].shape[-1] :]
            if tokens.numel() >= self.cfg.max_audio_tokens:
                raise PipelineError(
                    "Indic-Mio hit --max-audio-tokens; increase it or shorten speech chunks."
                )
            offset, vocabulary = 151669, 12800
            codes = tokens[(tokens >= offset) & (tokens < offset + vocabulary)] - offset
            if not codes.numel():
                raise PipelineError("Indic-Mio produced no valid speech tokens.")
            codes = codes.to(next(codec.parameters()).device)
            waveform = codec.decode(
                content_token_indices=codes, global_embedding=self.rt.cache["mio_voice"]
            )
        return waveform.detach().float().cpu().numpy().reshape(-1), int(
            codec.config.sample_rate
        )
