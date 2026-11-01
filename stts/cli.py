import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from .config import Config, PipelineError
from .metrics import summarize


def parser():
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("--profile", choices=["liquid", "urdu"])
    common.add_argument("--language", choices=["en", "ur"])
    common.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"])
    common.add_argument("--dtype", choices=["auto", "float16", "float32", "bfloat16"])
    common.add_argument("--threads", type=int)
    common.add_argument("--llm-backend", choices=["transformers", "ollama"])
    common.add_argument("--llm-model")
    common.add_argument("--audio-model")
    common.add_argument("--tts-backend", choices=["liquid", "omnivoice", "indic-mio"])
    common.add_argument(
        "--indic-model",
        choices=["bodhan-ai/indic-transcribe-core", "bodhan-ai/indic-transcribe-flex"],
    )
    common.add_argument("--mixed-script", action="store_true")
    common.add_argument("--romanized", action="store_true")
    common.add_argument("--ollama-url")
    common.add_argument("--max-new-tokens", type=int)
    common.add_argument("--max-audio-tokens", type=int)
    common.add_argument("--max-context-tokens", type=int)
    common.add_argument("--history-turns", type=int)
    common.add_argument("--temperature", type=float)
    common.add_argument("--gender", choices=["female", "male"])
    common.add_argument("--locale", choices=["American English", "British English"])
    common.add_argument("--voice-description")
    common.add_argument("--reference-audio")
    common.add_argument(
        "--reference-text",
        help="Exact transcript of the reference voice clip; prevents automatic ASR.",
    )
    common.add_argument(
        "--expression",
        choices=["neutral", "laughter", "sigh"],
        help="OmniVoice vocal-event prefix, not a guaranteed emotion label.",
    )
    common.add_argument(
        "--emotion",
        choices=[
            "neutral",
            "happy",
            "sad",
            "angry",
            "disgust",
            "fear",
            "surprise",
            "enunciated",
            "confused",
            "whisper",
        ],
        help="Indic-Mio only; named emotion/style tag, language dependent.",
    )
    common.add_argument("--speed", type=float)
    common.add_argument("--omni-steps", type=int)
    common.add_argument("--chunk-chars", type=int)
    common.add_argument("--play", action="store_true")
    common.add_argument("--release-between-stages", action="store_true")
    common.add_argument("--offline", action="store_true")
    common.add_argument("--seed", type=int)
    common.add_argument("--max-input-seconds", type=float)
    common.add_argument("--metrics-file")
    common.add_argument("--input-device", type=int)
    common.add_argument("--output-device", type=int)
    common.add_argument("--silence-seconds", type=float)
    common.add_argument("--silence-threshold", type=float)
    common.add_argument("--wait-seconds", type=float)
    root = argparse.ArgumentParser(
        description="Local English/Urdu voice pipeline. Global options work before or after a command.",
        parents=[common],
    )
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor",
        parents=[common],
        help="Check dependencies and hardware without downloading weights.",
    )
    commands.add_parser(
        "devices", parents=[common], help="List microphone and playback device numbers."
    )
    mic = commands.add_parser(
        "mic-test",
        parents=[common],
        help="Show microphone levels without loading models.",
    )
    mic.add_argument("--seconds", type=float, default=5)
    mic.add_argument(
        "--output", help="Optionally save the diagnostic recording as WAV."
    )
    mic.add_argument("--force", action="store_true")
    commands.add_parser(
        "stats",
        parents=[common],
        help="Summarize the metrics JSONL file by configuration and cold/warm state.",
    )
    for name in ("transcribe", "ask", "speak", "run", "listen", "benchmark"):
        sub = commands.add_parser(name, parents=[common])
        if name == "transcribe":
            sub.add_argument("--input", required=True)
        elif name in ("ask", "speak"):
            sub.add_argument("--text", required=True)
        elif name in ("run", "benchmark"):
            source = sub.add_mutually_exclusive_group(required=True)
            source.add_argument("--input")
            source.add_argument("--text")
        if name in ("transcribe", "run", "listen", "benchmark"):
            sub.add_argument(
                "--reference-transcript", help="Human ground truth for ASR WER/CER."
            )
        if name in ("run", "speak", "listen", "benchmark"):
            sub.add_argument(
                "--output",
                help="WAV output. Otherwise generates a unique file in outputs/.",
            )
            sub.add_argument("--force", action="store_true")
            sub.add_argument(
                "--verify-tts",
                action="store_true",
                help="Extra non-Whisper ASR pass over output for a text-fidelity proxy (<=30s).",
            )
        if name == "listen":
            sub.add_argument(
                "--turns", type=int, default=0, help="0 = keep listening until /quit."
            )
            sub.add_argument(
                "--json",
                action="store_true",
                help="Print full metrics after each turn instead of a readable conversation.",
            )
        if name == "benchmark":
            sub.add_argument(
                "--runs",
                type=int,
                default=5,
                help="Repeated identical inputs, one cold turn then warm turns; history reset each time.",
            )
    return root


def doctor(config):
    report = {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "config": config.public_dict(),
        "dependencies": {},
        "notes": [],
    }
    for package in (
        "torch",
        "transformers",
        "soundfile",
        "sounddevice",
        "librosa",
        "psutil",
        "omnivoice",
        "bodhan-genai",
    ):
        try:
            report["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][package] = None
    try:
        import torch

        report["mps_available"] = torch.backends.mps.is_available()
        report["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        report["notes"].append("Install requirements.txt to enable inference.")
    try:
        import psutil

        report["system_ram_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
        report["system_available_ram_gb"] = round(
            psutil.virtual_memory().available / 2**30, 2
        )
        if report["system_ram_gb"] < 12:
            report["notes"].append(
                "The 8GB CPU server is memory constrained for the full Liquid stack. CPU float32 audio weights alone approach 8GB; use reduced precision and/or separate services. Real-time performance is not guaranteed."
            )
    except ImportError:
        pass
    report["notes"].extend(
        [
            "doctor does not download or execute checkpoint code.",
            "Microphone and speaker hardware are accessed only by devices/listen/--play.",
            "Cold loading includes model downloads on first use; subsequent listen turns reuse models.",
            "The CLI is turn-based, with sentence-at-a-time TTS. It does not implement full-duplex interruption or token-streaming ASR.",
        ]
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main(argv=None):
    args = parser().parse_args(argv)
    pipeline = None
    try:
        config = Config.from_args(args)
        if config.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        if args.command == "doctor":
            doctor(config)
            return 0
        if args.command == "stats":
            print(
                json.dumps(summarize(config.metrics_file), indent=2, ensure_ascii=False)
            )
            return 0
        if args.command == "devices":
            import sounddevice as sd

            print(sd.query_devices())
            return 0
        if args.command == "mic-test":
            from .audio import microphone_test

            print(
                json.dumps(
                    microphone_test(config, args.seconds, args.output, args.force),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        # Reject conflicting outputs before downloading any model weights.
        output = getattr(args, "output", None)
        if output and Path(output).exists() and not args.force:
            raise PipelineError(
                "Output exists. Choose another --output or use --force."
            )
        if (
            output
            and getattr(args, "input", None)
            and Path(output).resolve() == Path(args.input).resolve()
        ):
            raise PipelineError("Input and output audio paths must be different.")
        if args.command == "listen" and args.turns < 0:
            raise PipelineError("--turns must be nonnegative.")
        if args.command == "listen" and output and args.turns != 1:
            raise PipelineError(
                "Use --turns 1 with --output, or omit --output for unique files per turn."
            )
        if args.command == "benchmark" and (args.runs < 1 or output):
            raise PipelineError(
                "Benchmark needs --runs >= 1 and generates unique outputs; omit --output."
            )
        reference = getattr(args, "reference_transcript", None)
        if reference is not None and not reference.strip():
            raise PipelineError("--reference-transcript must not be empty.")
        if reference is not None:
            from .metrics import error_rates

            error_rates(reference, reference)  # Validate before loading weights.
        from .pipeline import VoicePipeline

        pipeline = VoicePipeline(config)
        kwargs = {
            "output": output,
            "reference": reference,
            "verify_tts": getattr(args, "verify_tts", False),
            "force": getattr(args, "force", False),
        }
        if args.command == "listen":
            count = 0
            while not args.turns or count < args.turns:
                action = input(
                    "Type a message, or press Enter then speak aloud (/reset, /quit): "
                ).strip()
                if action == "/quit":
                    break
                if action == "/reset":
                    pipeline.history.clear()
                    continue
                if action.startswith("/"):
                    print(
                        "Commands: /reset clears history; /quit stops. Otherwise type a message or press Enter for microphone input.",
                        file=sys.stderr,
                    )
                    continue
                try:
                    result = pipeline.turn(
                        "run",
                        text=action or None,
                        microphone=not bool(action),
                        **kwargs,
                    )
                except PipelineError as exc:
                    print(f"Turn failed: {exc}", file=sys.stderr)
                    count += 1
                    continue
                if args.json:
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                else:
                    print(f"You: {result.get('transcript', action)}")
                    print(f"Assistant: {result['reply']}")
                    print(
                        f"Audio: {result.get('output', '')}\nMetrics: {config.metrics_file}"
                    )
                count += 1
        elif args.command == "benchmark":
            results = []
            for index in range(args.runs):
                pipeline.history.clear()
                print(
                    f"Benchmark turn {index + 1}/{args.runs}",
                    file=sys.stderr,
                    flush=True,
                )
                results.append(
                    pipeline.turn(
                        "run",
                        input_path=getattr(args, "input", None),
                        text=getattr(args, "text", None),
                        **kwargs,
                    )
                )
            print(json.dumps({"runs": results}, indent=2, ensure_ascii=False))
        else:
            result = pipeline.turn(
                args.command,
                input_path=getattr(args, "input", None),
                text=getattr(args, "text", None),
                **kwargs,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (KeyboardInterrupt, EOFError):
        print("Stopped.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports third-party model errors.
        if isinstance(exc, ModuleNotFoundError):
            message = f"Missing dependency: {exc.name}. Install requirements.txt and the profile's optional requirements."
        else:
            message = str(exc)
        print(f"Error: {message}", file=sys.stderr)
        return 1
    finally:
        if pipeline:
            pipeline.close()
