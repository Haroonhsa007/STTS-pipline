"""Local browser UI. Models stay lazy and inference runs outside the HTTP loop."""

import asyncio
import copy
import json
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import psutil
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Config, PipelineError

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
MAX_UPLOAD = 12 * 1024 * 1024


def prune_generated_audio(folder, keep=32):
    """Bound app-generated UUID recordings across restarts; leave named files alone."""
    candidates = []
    for path in folder.glob("*.wav"):
        try:
            if str(uuid.UUID(path.stem)) != path.stem:
                continue
            candidates.append(path)
        except ValueError:
            continue
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[
        keep:
    ]:
        path.unlink(missing_ok=True)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: Literal["auto", "en", "ur"] = "auto"
    engine: Literal["python", "native", "bilingual"] = "bilingual"
    tts_backend: Literal["liquid", "omnivoice", "indic-mio"] = "omnivoice"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    gender: Literal["female", "male"] = "female"
    locale: Literal["American English", "British English"] = "American English"
    emotion: str = Field(default="neutral", max_length=24)
    expression: Literal["neutral", "laughter", "sigh"] = "neutral"
    speed: float = Field(default=1, ge=0.5, le=2)
    voice_description: str = Field(default="", max_length=500)
    reference_text: str = Field(default="", max_length=2000)
    max_new_tokens: int = Field(default=384, ge=16, le=1024)
    omni_steps: int = Field(default=16, ge=4, le=64)
    offline: bool = True
    verify_tts: bool = False

    def config(self, reference_path=None):
        values = self.model_dump(exclude={"verify_tts"})
        values.update(
            profile="urdu" if self.language == "ur" else "liquid",
            llm_model="Qwen/Qwen3.5-2B"
            if self.language == "ur"
            else "LiquidAI/LFM3-1.7B",
            reference_audio=str(reference_path) if reference_path else None,
            reference_text=self.reference_text.strip() or None,
            voice_description=self.voice_description.strip() or None,
            metrics_file=str(OUTPUTS / "metrics.jsonl"),
        )
        if self.engine == "native":
            values.update(
                llm_model="LiquidAI/LFM3-Audio-2B-GGUF",
                audio_model="LiquidAI/LFM3-Audio-2B-GGUF",
                history_turns=16,
            )
        elif self.engine == "bilingual":
            values.update(
                profile="urdu",
                llm_model="LiquidAI/LFM3-Audio-2B-GGUF",
                audio_model="Mvkd2k06/indic-transcribe-flex-onnx",
                history_turns=16,
            )
        config = Config(**values)
        if reference_path:
            import hashlib

            config.reference_audio_id = (
                "sha256:" + hashlib.sha256(reference_path.read_bytes()).hexdigest()
            )
        config.validate()
        return config


class Job:
    def __init__(self, identifier, owner):
        self.id, self.owner = identifier, owner
        self.started = time.perf_counter()
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.deltas = []
        self.data = {
            "id": identifier,
            "status": "running",
            "phase": "Preparing runtime",
            "stages": {},
        }
        self.output = None

    def event(self, event):
        with self.lock:
            kind = event["kind"]
            if kind == "stage":
                self.data["stages"][event["name"]] = event["row"]
                if event["row"]["status"] == "running":
                    self.data["phase"] = event["row"].get("stage_name", event["name"])
            elif kind in ("transcript", "reply"):
                self.data[kind] = event["text"]
            elif kind == "reply_delta":
                self.data["reply"] = event["text"]
            elif kind == "audio_chunk":
                self.deltas.append(
                    {
                        "seq": len(self.deltas) + 1,
                        "kind": kind,
                        "pcm": event["pcm"],
                        "sample_rate": event["sample_rate"],
                    }
                )
            elif kind == "audio_ready":
                self.data["response_first_audio_s"] = event["response_first_audio_s"]
            elif kind == "finished":
                self.data.update(
                    {
                        key: copy.deepcopy(value)
                        for key, value in event["result"].items()
                        if key != "status"
                    }
                )
            # The pipeline's finish event occurs before it flushes metrics to disk.
            # Only the worker marks the job complete, once its result is available.

    def snapshot(self, after=0):
        with self.lock:
            result = copy.deepcopy(self.data)
            result["deltas"] = [
                dict(delta) for delta in self.deltas if delta["seq"] > after
            ]
        if result["status"] == "running":
            result["total_wall_s"] = time.perf_counter() - self.started
        return result


def system_metrics():
    process = psutil.Process()
    memory = psutil.virtual_memory()
    try:
        children_rss = sum(
            child.memory_info().rss for child in process.children(recursive=True)
        )
    except psutil.Error:
        children_rss = 0
    return {
        "system_cpu_percent": psutil.cpu_percent(),
        "process_rss_mb": process.memory_info().rss / 2**20,
        "native_rss_mb": children_rss / 2**20,
        "disk_free_mb": shutil.disk_usage(ROOT).free / 2**20,
        "system_available_mb": memory.available / 2**20,
        "system_total_mb": memory.total / 2**20,
        "swap_used_mb": psutil.swap_memory().used / 2**20,
        "cpu_count": psutil.cpu_count(),
    }


class Manager:
    def __init__(self, pipeline_factory=None):
        self.factory = pipeline_factory
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="stts-inference"
        )
        self.guard = threading.Lock()
        self.jobs = {}
        self.pipeline = None
        self.config_key = None
        self.histories = {}

    def run(
        self,
        job,
        config,
        mode,
        text,
        source,
        reference,
        scratch,
        verify,
        expected_transcript,
    ):
        with job.lock:
            job.data["config"] = config.public_dict()
        try:
            from .pipeline import VoicePipeline

            # Public config uses a stable content fingerprint for reference audio.
            key_values = config.public_dict()
            if config.engine in ("native", "bilingual"):
                key_values = {
                    key: key_values[key] for key in ("engine", "device", "threads")
                }
            key = json.dumps(key_values, sort_keys=True)
            if self.pipeline is None or key != self.config_key:
                if self.pipeline:
                    self.pipeline.close()
                    self.pipeline = None
                if config.engine == "bilingual" and not self.factory:
                    from .bilingual import BilingualPipeline

                    factory = BilingualPipeline
                elif config.engine == "native" and not self.factory:
                    from .native import NativePipeline

                    factory = NativePipeline
                else:
                    factory = self.factory or VoicePipeline
                self.pipeline = factory(config, event_sink=job.event)
                self.config_key = key
            self.pipeline.event_sink = job.event
            if config.engine in ("native", "bilingual"):
                self.pipeline.config = config
                if hasattr(self.pipeline, "runtime"):
                    self.pipeline.runtime.config = config
                if hasattr(self.pipeline, "voice"):
                    self.pipeline.voice.config = config
            self.pipeline.cancel_event = job.cancel
            self.pipeline.config.reference_audio = str(reference) if reference else None
            self.pipeline.history = list(self.histories.get(job.owner, []))
            result = self.pipeline.turn(
                mode,
                input_path=source,
                text=text,
                output=OUTPUTS / "web" / f"{job.id}.wav",
                verify_tts=verify,
                reference=expected_transcript,
            )
            self.histories[job.owner] = list(self.pipeline.history)
            while len(self.histories) > 32:
                del self.histories[next(iter(self.histories))]
            with job.lock:
                job.output = Path(result["output"]) if result.get("output") else None
                job.data.update(result)
                job.data["pipeline_wall_s"] = result.get("total_wall_s")
                job.data["total_wall_s"] = time.perf_counter() - job.started
                job.data["phase"] = "Complete"
                if job.output:
                    job.data["audio_url"] = (
                        f"/api/jobs/{job.id}/audio?session={job.owner}"
                    )
            if job.output:
                try:
                    prune_generated_audio(OUTPUTS / "web")
                except OSError as exc:
                    with job.lock:
                        job.data["storage_warning"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - worker boundary serializes third-party model failures.
            with job.lock:
                job.data.update(
                    status="cancelled" if job.cancel.is_set() else "error",
                    error=str(exc),
                    phase="Interrupted" if job.cancel.is_set() else "Failed",
                    total_wall_s=time.perf_counter() - job.started,
                )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
            self.guard.release()

    def close(self):
        self.executor.shutdown(wait=True)
        if self.pipeline:
            self.pipeline.close()


async def save_upload(upload, path):
    size = 0
    try:
        with path.open("wb") as handle:
            while chunk := await upload.read(65536):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(
                        413,
                        "Audio upload exceeds 12 MiB. Use a clip of 30 seconds or less.",
                    )
                handle.write(chunk)
    finally:
        await upload.close()
    if not size:
        raise HTTPException(422, "The audio file is empty.")
    return path


def valid_session(value):
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError):
        raise HTTPException(422, "Invalid browser session.") from None


def create_app(pipeline_factory=None):
    manager = Manager(pipeline_factory)

    @asynccontextmanager
    async def lifespan(app):
        prune_generated_audio(OUTPUTS / "web")
        yield
        await asyncio.to_thread(manager.close)

    app = FastAPI(title="STTS Voice Studio", lifespan=lifespan)
    app.state.manager = manager
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            from fastapi.responses import JSONResponse

            return JSONResponse(
                {"detail": "Use this page from its own localhost address."},
                status_code=403,
            )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def get_job(identifier, session):
        job = manager.jobs.get(identifier)
        if not job or job.owner != valid_session(session):
            raise HTTPException(404, "Turn not found. It may have expired.")
        return job

    @app.get("/")
    async def home():
        return FileResponse(ROOT / "web/index.html")

    @app.get("/api/system")
    async def system():
        from .bilingual import available as bilingual_available
        from .native import available

        return {
            **system_metrics(),
            "busy": manager.guard.locked(),
            "native_available": available(),
            "bilingual_available": bilingual_available(),
        }

    @app.post("/api/turn", status_code=202)
    async def turn(
        session: str = Form(...),
        settings: str = Form("{}", max_length=6000),
        mode: Literal["run", "ask", "speak", "transcribe", "warmup"] = Form("run"),
        text: str = Form("", max_length=8000),
        expected_transcript: str = Form("", max_length=8000),
        audio: Annotated[UploadFile | None, File()] = None,
        voice: Annotated[UploadFile | None, File()] = None,
    ):
        owner = valid_session(session)
        try:
            options = Settings.model_validate_json(settings)
        except ValidationError as exc:
            raise HTTPException(422, str(exc)) from None
        if mode != "warmup" and bool(text.strip()) == bool(audio):
            raise HTTPException(422, "Provide either a message or an audio recording.")
        if mode == "transcribe" and not audio:
            raise HTTPException(422, "Transcription needs an audio recording.")
        if mode == "speak" and audio:
            raise HTTPException(422, "Voice preview needs text.")
        if mode == "warmup" and options.engine not in ("native", "bilingual"):
            raise HTTPException(422, "Model warmup is available for the native engine.")
        if mode in ("ask", "transcribe"):
            # These operations never synthesize speech, so do not require TTS assets.
            options = options.model_copy(
                update={
                    "tts_backend": "omnivoice"
                    if options.language == "ur" or options.engine == "bilingual"
                    else "liquid",
                    "emotion": "neutral",
                    "expression": "neutral",
                    "speed": 1,
                    "voice_description": "",
                    "reference_text": "",
                    "verify_tts": False,
                }
            )
            if voice:
                await voice.close()
                voice = None
        if not manager.guard.acquire(blocking=False):
            raise HTTPException(
                409, "A turn is already running. Wait for it to finish."
            )
        identifier = str(uuid.uuid4())
        job = Job(identifier, owner)
        scratch = Path(tempfile.mkdtemp(prefix="stts-web-"))
        try:
            source = (
                await save_upload(audio, scratch / "input.audio") if audio else None
            )
            reference = (
                await save_upload(voice, scratch / "voice.audio") if voice else None
            )
            # Decode before model initialization. Reject invalid audio and excessive duration.
            if source or reference:
                from .audio import read_audio, save_wave

                for path in (source, reference):
                    if path:
                        samples, rate = await asyncio.to_thread(read_audio, path, 30)
                        normalized = path.with_suffix(".wav")
                        await asyncio.to_thread(save_wave, normalized, samples, rate)
                        if path == source:
                            source = normalized
                        else:
                            reference = normalized
            config = options.config(reference)
            manager.jobs[identifier] = job
            while len(manager.jobs) > 32:
                expired = manager.jobs.pop(next(iter(manager.jobs)))
                if expired.output:
                    expired.output.unlink(missing_ok=True)
            manager.executor.submit(
                manager.run,
                job,
                config,
                mode,
                text.strip(),
                source,
                reference,
                scratch,
                options.verify_tts,
                expected_transcript.strip() or None,
            )
            return {"id": identifier}
        except BaseException as exc:
            shutil.rmtree(scratch, ignore_errors=True)
            manager.guard.release()
            if isinstance(exc, (PipelineError, RuntimeError, ValueError)):
                raise HTTPException(422, str(exc)) from None
            raise

    @app.get("/api/jobs/{identifier}")
    async def status(identifier: str, session: str, after: int = 0):
        return get_job(identifier, session).snapshot(max(0, after))

    @app.post("/api/jobs/{identifier}/cancel")
    async def cancel(identifier: str, session: str):
        job = get_job(identifier, session)
        job.cancel.set()
        return {"status": "cancelling"}

    @app.get("/api/jobs/{identifier}/events")
    async def events(identifier: str, session: str, request: Request):
        job = get_job(identifier, session)

        async def stream():
            cursor = 0
            while not await request.is_disconnected():
                snapshot = job.snapshot(cursor)
                if snapshot["deltas"]:
                    cursor = snapshot["deltas"][-1]["seq"]
                snapshot["system"] = system_metrics()
                yield (
                    "data: "
                    + json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
                    + "\n\n"
                )
                if snapshot["status"] != "running":
                    break
                await asyncio.sleep(0.3)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/jobs/{identifier}/audio")
    async def result_audio(identifier: str, session: str):
        job = get_job(identifier, session)
        if not job.output or not job.output.is_file():
            raise HTTPException(404, "No generated audio for this turn.")
        return FileResponse(
            job.output, media_type="audio/wav", filename=f"reply-{job.id[:8]}.wav"
        )

    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
    return app


app = create_app()
