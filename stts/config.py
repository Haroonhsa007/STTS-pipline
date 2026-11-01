from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path


class PipelineError(Exception):
    """An actionable input, dependency or inference error."""


@dataclass
class Config:
    engine: str = "python"
    profile: str = "liquid"
    language: str = "en"
    device: str = "auto"
    dtype: str = "auto"
    threads: int = 4
    llm_backend: str = "transformers"
    llm_model: str = "LiquidAI/LFM3-1.7B"
    audio_model: str = "LiquidAI/LFM3-Audio-2B"
    tts_backend: str = "liquid"
    indic_model: str = "bodhan-ai/indic-transcribe-core"
    mixed_script: bool = False
    romanized: bool = False
    ollama_url: str = "http://127.0.0.1:11434"
    max_new_tokens: int = 256
    max_audio_tokens: int = 2048
    max_context_tokens: int = 2048
    history_turns: int = 4
    temperature: float = 0.1
    gender: str = "female"
    locale: str = "American English"
    voice_description: str | None = None
    reference_audio: str | None = None
    reference_audio_id: str | None = None
    reference_text: str | None = None
    expression: str = "neutral"
    emotion: str = "neutral"
    speed: float = 1.0
    omni_steps: int = 32
    chunk_chars: int = 220
    play: bool = False
    release_between_stages: bool = False
    offline: bool = False
    seed: int = 42
    max_input_seconds: float = 30.0
    metrics_file: str = "outputs/metrics.jsonl"
    input_device: int | None = None
    output_device: int | None = None
    silence_seconds: float = 0.8
    silence_threshold: float = 0.012
    wait_seconds: float = 8.0

    @classmethod
    def from_args(cls, args):
        fields = cls.__dataclass_fields__
        values = {k: v for k, v in vars(args).items() if k in fields}
        profile = values.get(
            "profile", "urdu" if values.get("language") == "ur" else "liquid"
        )
        values["profile"] = profile
        if profile == "urdu":
            values.setdefault("language", "ur")
            values.setdefault("llm_model", "Qwen/Qwen3.5-2B")
            values.setdefault("tts_backend", "omnivoice")
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        if self.engine not in ("python", "native", "bilingual"):
            raise PipelineError("Choose the python, native or bilingual engine.")
        if self.engine == "native" and (
            self.language != "en" or self.tts_backend != "liquid"
        ):
            raise PipelineError(
                "Native conversation currently supports English with its default voice. Choose Python for Urdu or expressive voices."
            )
        for key in (
            "speed",
            "temperature",
            "max_input_seconds",
            "silence_seconds",
            "silence_threshold",
            "wait_seconds",
        ):
            if not isfinite(getattr(self, key)):
                raise PipelineError(f"--{key.replace('_', '-')} must be finite.")
        if (
            self.engine == "python"
            and self.profile == "liquid"
            and self.language != "en"
        ):
            raise PipelineError(
                "Liquid's selected models do not list Urdu. Use --profile urdu."
            )
        if self.language == "auto" and self.engine != "bilingual":
            raise PipelineError(
                "Automatic English/Urdu switching requires the bilingual engine."
            )
        if self.language == "ur" and self.tts_backend == "liquid":
            raise PipelineError(
                "Liquid TTS does not list Urdu. Use --tts-backend omnivoice."
            )
        if self.tts_backend != "indic-mio" and bool(self.reference_audio) != bool(
            self.reference_text
        ):
            raise PipelineError(
                "Provide both --reference-audio and its exact --reference-text; auto-ASR is disabled."
            )
        if self.reference_audio and not Path(self.reference_audio).is_file():
            raise PipelineError(
                f"Reference audio does not exist: {self.reference_audio}"
            )
        if self.reference_text is not None and not self.reference_text.strip():
            raise PipelineError("The reference transcript must not be empty.")
        if self.reference_audio and self.voice_description:
            raise PipelineError(
                "Choose reference voice cloning OR --voice-description."
            )
        custom_voice = (
            self.reference_audio
            or self.voice_description
            or self.expression != "neutral"
            or self.speed != 1
            or self.emotion != "neutral"
        )
        if self.tts_backend == "liquid" and custom_voice:
            raise PipelineError(
                "Cloning, descriptions, expression tags and speed require --tts-backend omnivoice."
            )
        if self.emotion != "neutral" and self.tts_backend != "indic-mio":
            raise PipelineError(
                "Named emotion tags require --tts-backend indic-mio. OmniVoice supports reference style and --expression vocal events instead."
            )
        if self.tts_backend == "indic-mio":
            if not self.reference_audio:
                raise PipelineError(
                    "Indic-Mio requires --reference-audio to supply a speaker embedding."
                )
            if (
                self.voice_description
                or self.expression != "neutral"
                or self.speed != 1
            ):
                raise PipelineError(
                    "Indic-Mio supports reference voices and --emotion, not description, expression or speed controls."
                )
            supported = (
                {"neutral", "happy", "sad", "angry", "disgust", "fear", "surprise"}
                if self.language == "ur"
                else {
                    "neutral",
                    "happy",
                    "sad",
                    "enunciated",
                    "confused",
                    "angry",
                    "whisper",
                }
            )
            if self.emotion not in supported:
                raise PipelineError(
                    f"This language supports these Indic-Mio emotion tags: {', '.join(sorted(supported))}"
                )
        if self.mixed_script and self.romanized:
            raise PipelineError("Choose --mixed-script OR --romanized.")
        if (self.mixed_script or self.romanized) and not self.indic_model.endswith(
            "-flex"
        ):
            raise PipelineError(
                "Output formatting modes require --indic-model bodhan-ai/indic-transcribe-flex."
            )
        if (
            self.llm_backend == "ollama"
            and "/" in self.llm_model
            and not self.llm_model.startswith("hf.co/")
        ):
            raise PipelineError(
                "For Ollama, pass the exact installed Ollama model name with --llm-model."
            )
        if not 0.5 <= self.speed <= 2.0:
            raise PipelineError("--speed must be between 0.5 and 2.0.")
        if not 4 <= self.omni_steps <= 128:
            raise PipelineError(
                "--omni-steps must be between 4 and 128 (32 is the quality default)."
            )
        if not 40 <= self.chunk_chars <= 1000:
            raise PipelineError("--chunk-chars must be between 40 and 1000.")
        for key in (
            "threads",
            "max_new_tokens",
            "max_audio_tokens",
            "max_context_tokens",
        ):
            if getattr(self, key) <= 0:
                raise PipelineError(f"--{key.replace('_', '-')} must be positive.")
        if self.history_turns < 0 or self.temperature < 0:
            raise PipelineError("History turns and temperature must not be negative.")
        if not 1 <= self.max_input_seconds <= 30:
            raise PipelineError(
                "--max-input-seconds must be 1–30; split longer recordings into turns."
            )
        if (
            not 0 < self.silence_seconds <= 5
            or not 0 < self.silence_threshold < 1
            or self.wait_seconds <= 0
        ):
            raise PipelineError("Invalid microphone silence threshold or timeout.")

    def public_dict(self):
        values = asdict(self)
        if self.reference_audio_id:
            values["reference_audio"] = self.reference_audio_id
        return values
