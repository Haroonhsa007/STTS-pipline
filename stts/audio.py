import math
import sys
import tempfile
import time
from pathlib import Path

from .config import PipelineError

SAMPLE_RATE = 16000


def read_audio(path, max_seconds=30):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    source = Path(path)
    if not source.is_file():
        raise PipelineError(f"Audio file not found: {path}")
    info = sf.info(str(source))
    if not info.frames or not info.samplerate or info.duration > max_seconds:
        raise PipelineError(
            f"Input must contain 0–{max_seconds:g} seconds of audio; got {info.duration:.2f}s."
        )
    samples, rate = sf.read(str(source), dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if not np.isfinite(samples).all():
        raise PipelineError("Audio contains invalid samples.")
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        samples = resample_poly(
            samples, SAMPLE_RATE // divisor, rate // divisor
        ).astype("float32")
    if float(np.max(np.abs(samples))) < 1e-5:
        raise PipelineError("Audio is silent.")
    return samples, SAMPLE_RATE


def input_device(config):
    import sounddevice as sd

    try:
        info = sd.query_devices(config.input_device, kind="input")
    except (ValueError, sd.PortAudioError) as exc:
        raise PipelineError(
            "No usable input device selected. Run python STTS.py devices and choose --input-device ID."
        ) from exc
    if info["max_input_channels"] < 1:
        raise PipelineError(
            "The selected device is output-only. Run python STTS.py devices and select a microphone."
        )
    return info


def no_speech_message(max_rms, threshold):
    if max_rms < 1e-5:
        return "Microphone returned silence. Check the input device, physical mute/input volume, and microphone permission for Terminal (or the app running Python). Run python STTS.py mic-test."
    return (
        f"Audio was received, but it was below the speech threshold: max RMS {max_rms:.5f}, threshold {threshold:.5f}. "
        "Speak AFTER 'Listening on ...', or run python STTS.py mic-test to tune --silence-threshold."
    )


def record(config, diagnostics=None):
    """Energy endpointing: bounded recording, trailing silence and wait timeout."""
    import numpy as np
    import sounddevice as sd

    info = input_device(config)
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.update(
        input_name=info["name"],
        threshold=config.silence_threshold,
        max_rms=0.0,
        signal_peak=0.0,
        speech_detected=False,
    )
    blocks, speech, quiet = [], False, 0.0
    frame_size = 512
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=frame_size,
        device=config.input_device,
    ) as stream:
        print(
            f"Listening on {info['name']} — speak aloud now; pause to finish.",
            file=sys.stderr,
            flush=True,
        )
        started = time.perf_counter()
        while True:
            block, overflow = stream.read(frame_size)
            if overflow:
                raise PipelineError(
                    "Microphone overflow: close other audio applications or use a file."
                )
            blocks.append(block[:, 0].copy())
            duration = len(blocks) * frame_size / SAMPLE_RATE
            level = float(np.sqrt(np.mean(block**2)))
            diagnostics.update(
                max_rms=max(diagnostics["max_rms"], level),
                signal_peak=max(
                    diagnostics["signal_peak"], float(np.max(np.abs(block)))
                ),
                captured_seconds=duration,
            )
            if level >= config.silence_threshold:
                speech, quiet = True, 0.0
                diagnostics["speech_detected"] = True
            elif speech:
                quiet += frame_size / SAMPLE_RATE
            if speech and quiet >= config.silence_seconds:
                break
            if not speech and time.perf_counter() - started >= config.wait_seconds:
                raise PipelineError(
                    no_speech_message(diagnostics["max_rms"], config.silence_threshold)
                )
            if duration >= config.max_input_seconds:
                break
    if not speech:
        raise PipelineError(
            no_speech_message(diagnostics["max_rms"], config.silence_threshold)
        )
    return np.concatenate(blocks), SAMPLE_RATE


def microphone_test(config, seconds=5, output=None, force=False):
    """Explicit user-run capture diagnostic; no model loads or downloads."""
    import numpy as np
    import sounddevice as sd

    if not math.isfinite(seconds) or not 1 <= seconds <= 30:
        raise PipelineError("--seconds must be between 1 and 30.")
    if output and Path(output).exists() and not force:
        raise PipelineError(
            "Diagnostic output exists; choose another path or use --force."
        )
    info = input_device(config)
    print(
        f"Microphone: {info['name']}. Speak aloud for {seconds:g} seconds. No model is loaded.",
        file=sys.stderr,
        flush=True,
    )
    blocks, levels, peak = [], [], 0.0
    frames_left = round(seconds * SAMPLE_RATE)
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=512,
        device=config.input_device,
    ) as stream:
        while frames_left:
            count = min(512, frames_left)
            block, overflow = stream.read(count)
            if overflow:
                raise PipelineError(
                    "Microphone overflow during test; close other audio applications and retry."
                )
            blocks.append(block[:, 0].copy())
            level = float(np.sqrt(np.mean(block**2)))
            levels.append(level)
            peak = max(peak, float(np.max(np.abs(block))))
            if len(levels) % 6 == 0 or count == frames_left:
                bars = min(30, int(level * 600))
                print(
                    f"\rMic [{'#' * bars:<30}] RMS {level:.5f} | trigger {config.silence_threshold:.5f}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
            frames_left -= count
    print(file=sys.stderr)
    samples = np.concatenate(blocks)
    maximum = max(levels)
    detected = maximum >= config.silence_threshold
    report = {
        "input_name": info["name"],
        "sample_rate": SAMPLE_RATE,
        "seconds": len(samples) / SAMPLE_RATE,
        "max_rms": maximum,
        "signal_peak": peak,
        "speech_threshold": config.silence_threshold,
        "threshold_crossed": detected,
    }
    if maximum < 1e-5:
        report["result"] = no_speech_message(maximum, config.silence_threshold)
    elif not detected:
        report["suggested_threshold"] = maximum * 0.4
        report["result"] = (
            "Signal received, but below the current threshold. If you spoke during this test, try the suggested threshold; it is not automatically applied."
        )
    else:
        report["result"] = (
            "Signal crossed the threshold. This confirms audio input, not speech recognition accuracy."
        )
    if output:
        report["output"] = str(save_wave(output, samples, SAMPLE_RATE, force).resolve())
    if config.play:
        play(samples, SAMPLE_RATE, config.output_device)
    return report


def save_wave(path, samples, sample_rate, force=False):
    import numpy as np
    import soundfile as sf

    target = Path(path)
    if target.exists() and not force:
        raise PipelineError(
            f"Output exists: {path}. Choose another path or pass --force."
        )
    samples = np.asarray(samples, dtype="float32").reshape(-1)
    if not samples.size or not np.isfinite(samples).all():
        raise PipelineError("The TTS model produced empty or invalid audio.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".wav", dir=target.parent, delete=False
    ) as handle:
        scratch = Path(handle.name)
    try:
        sf.write(str(scratch), np.clip(samples, -1, 1), sample_rate, subtype="PCM_16")
        if force:
            scratch.replace(target)
        else:
            # Atomic no-clobber publication, including concurrent CLI processes.
            import os

            os.link(scratch, target)
            scratch.unlink()
    finally:
        scratch.unlink(missing_ok=True)
    return target


def play(samples, sample_rate, device=None):
    import sounddevice as sd

    sd.play(samples, sample_rate, device=device)
    sd.wait()
