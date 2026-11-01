"""Measured timings and reference-based error rates; no invented quality scores."""

import json
import math
import os
import threading
import time
import unicodedata
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def normalize(text):
    # Preserve letters, diacritics and script distinctions; remove punctuation.
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join(
        "".join(
            " " if unicodedata.category(c).startswith("P") else c for c in text
        ).split()
    )


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = row
    return previous[-1]


def error_rates(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError("Reference transcript must contain text.")
    words, chars = ref.split(), list(ref.replace(" ", ""))
    return {
        "wer": edit_distance(words, hyp.split()) / len(words),
        "cer": edit_distance(chars, list(hyp.replace(" ", ""))) / len(chars),
        "normalization": "NFC + casefold + punctuation removal + whitespace collapse; CER excludes spaces",
    }


def percentile(values, p):
    ordered = sorted(values)
    at = (len(ordered) - 1) * p
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


class Metrics:
    def __init__(self, operation, config, event_sink=None):
        self.started = time.perf_counter()
        self.event_sink = event_sink
        self.data = {
            "schema_version": 1,
            "time_utc": datetime.now(UTC).isoformat(),
            "operation": operation,
            "status": "running",
            "config": config.public_dict(),
            "stages": {},
            "quality": {},
        }

    def emit(self, kind, **values):
        if self.event_sink:
            self.event_sink(
                {
                    "kind": kind,
                    "elapsed_s": time.perf_counter() - self.started,
                    **values,
                }
            )

    @contextmanager
    def stage(self, name, runtime=None):
        if runtime:
            runtime.sync()
        started = time.perf_counter()
        row = {"status": "running"}
        event_name, suffix = name, 2
        while event_name in self.data["stages"]:
            event_name = f"{name}__{suffix}"
            suffix += 1
        row["stage_name"] = name
        self.data["stages"][event_name] = row
        self.emit("stage", name=event_name, row=dict(row))
        stop = threading.Event()
        peaks = {"process_rss_peak_mb": None}
        try:
            import psutil

            process = psutil.Process(os.getpid())
            cpu_start = sum(process.cpu_times()[:2])
        except ImportError:
            process = None

        def sample():
            if process:
                rss = process.memory_info().rss / 2**20
                peaks["process_rss_peak_mb"] = max(
                    peaks["process_rss_peak_mb"] or 0, rss
                )

        def monitor():
            last_emit = 0.0
            while not stop.wait(0.05):
                sample()
                if self.event_sink and time.perf_counter() - last_emit >= 0.3:
                    last_emit = time.perf_counter()
                    self.emit(
                        "stage",
                        name=event_name,
                        row={**row, **peaks, "wall_s": time.perf_counter() - started},
                    )

        sample()
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            yield row
            row["status"] = "ok"
        except BaseException as exc:
            row["status"] = "error"
            row["error_type"] = type(exc).__name__
            raise
        finally:
            try:
                if runtime:
                    runtime.sync()
            finally:
                stop.set()
                thread.join()
                sample()
                row["wall_s"] = time.perf_counter() - started
                row.update(peaks)
                if process:
                    cpu_s = max(0, sum(process.cpu_times()[:2]) - cpu_start)
                    row.update(
                        process_cpu_s=cpu_s,
                        cpu_percent_one_core=100 * cpu_s / row["wall_s"],
                        system_available_end_mb=psutil.virtual_memory().available
                        / 2**20,
                        system_swap_used_end_mb=psutil.swap_memory().used / 2**20,
                    )
                if runtime:
                    row.update(runtime.memory())
                self.emit("stage", name=event_name, row=dict(row))

    def finish(self, status="ok", error=None):
        self.data["status"] = status
        self.data["total_wall_s"] = time.perf_counter() - self.started
        if error:
            self.data["error"] = str(error)
        self.emit(
            "finished",
            status=status,
            error=str(error) if error else None,
            result=self.data,
        )
        return self.data

    def append(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # One write per record. This CLI has one producer per process.
        line = json.dumps(self.data, ensure_ascii=False, allow_nan=False) + "\n"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line)


def summarize(path):
    records, malformed = [], 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
    # Compare like-for-like configurations and operations, never pool different models.
    groups = {}
    for record in records:
        config = record.get("config", {})
        cold = record.get("cold_in_process", False)
        identity = {
            "operation": record.get("operation"),
            "config": config,
            "cold_in_process": cold,
            "run_options": record.get("run_options", {}),
            "device": record.get("device"),
            "dtype": record.get("dtype"),
            "detected_language": record.get("detected_language"),
        }
        key = json.dumps(identity, sort_keys=True)
        group = groups.setdefault(
            key, {**identity, "runs": 0, "failures": 0, "measurements": {}}
        )
        group["runs"] += 1
        if record.get("status") != "ok":
            group["failures"] += 1
            continue
        measurements = {"total_wall_s": record.get("total_wall_s")}
        for name, row in record.get("stages", {}).items():
            for metric in (
                "wall_s",
                "ttft_s",
                "tokens_per_s",
                "rtf",
                "first_audio_s",
                "process_rss_peak_mb",
                "model_inference_s",
                "audio_seconds",
                "input_tokens",
                "output_tokens",
                "cpu_percent_one_core",
            ):
                measurements[f"{name}.{metric}"] = row.get(metric)
        for task in ("asr", "tts_asr_proxy"):
            for metric in ("wer", "cer"):
                measurements[f"quality.{task}.{metric}"] = (
                    record.get("quality", {}).get(task, {}).get(metric)
                )
        measurements["response_first_audio_s"] = record.get("response_first_audio_s")
        for metric, value in measurements.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                group["measurements"].setdefault(metric, []).append(value)
    for group in groups.values():
        stats = {}
        for metric, values in group.pop("measurements").items():
            stats[metric] = {
                "n": len(values),
                "mean": sum(values) / len(values),
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
            }
        group["statistics"] = stats
        recommendations = []
        loads = sum(
            v["mean"]
            for k, v in stats.items()
            if k.startswith("load_") and k.endswith(".wall_s")
        )
        total = stats.get("total_wall_s", {}).get("mean", 0)
        if total and loads / total > 0.25:
            recommendations.append(
                "Model loading dominates: compare warmed turns in listen mode; keep models resident when memory permits."
            )
        if stats.get("tts.rtf", {}).get("mean", 0) > 1:
            recommendations.append(
                "TTS generates slower than playback: try a smaller backend or fewer OmniVoice steps, then re-check voice quality."
            )
        if stats.get("asr.rtf", {}).get("mean", 0) > 1:
            recommendations.append(
                "ASR is slower than the recording duration: test shorter turns, quantized inference or a faster device."
            )
        if group["config"].get("release_between_stages"):
            recommendations.append(
                "Releasing models saves memory but reloads weights between stages; benchmark with it disabled on the M4."
            )
        group["suggestions"] = recommendations
    return {
        "records": len(records),
        "malformed_lines": malformed,
        "groups": list(groups.values()),
        "note": "p95 from fewer than 20 measurements is unstable. Warm/cold status is recorded on each load event. Quality requires reference transcripts or human evaluation.",
    }
