"""Evaluate community JSONL cases against an already-running local STTS server.

No server, model download or microphone is started by this command.
"""

import argparse
import json
import math
import time
import uuid
from pathlib import Path

import httpx

MODES = {"transcribe", "ask", "speak", "run"}


def load_cases(path):
    path = Path(path)
    cases = []
    identifiers = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict):
            raise TypeError(f"Line {number}: each case must be an object.")
        unknown = set(case) - {"id", "mode", "language", "audio", "text", "reference"}
        if unknown:
            raise ValueError(f"Line {number}: unknown fields {sorted(unknown)}.")
        case.setdefault("id", f"case-{number}")
        case.setdefault("mode", "run")
        case.setdefault("language", "auto")
        if (
            not isinstance(case["id"], str)
            or not case["id"]
            or case["id"] in identifiers
        ):
            raise ValueError(f"Line {number}: id must be a unique nonempty string.")
        identifiers.add(case["id"])
        if case["mode"] not in MODES or case["language"] not in {"auto", "en", "ur"}:
            raise ValueError(f"Line {number}: unsupported mode or language.")
        for field in ("text", "audio", "reference"):
            if field in case and not isinstance(case[field], str):
                raise ValueError(f"Line {number}: {field} must be text.")
        has_text = bool(case.get("text", "").strip())
        has_audio = bool(case.get("audio"))
        if has_text == has_audio:
            raise ValueError(f"Line {number}: provide exactly one of text or audio.")
        if case["mode"] == "transcribe" and not has_audio:
            raise ValueError(f"Line {number}: transcribe requires audio.")
        if case["mode"] in {"ask", "speak"} and has_audio:
            raise ValueError(f"Line {number}: ask/speak require text.")
        if has_audio:
            audio = (path.parent / case["audio"]).resolve()
            if not audio.is_file():
                raise ValueError(f"Line {number}: audio file not found: {audio}")
            if audio.stat().st_size > 12 * 1024 * 1024:
                raise ValueError(
                    f"Line {number}: audio exceeds the 12 MiB upload limit."
                )
            case["audio"] = str(audio)
        cases.append(case)
    if not cases:
        raise ValueError("The manifest contains no cases.")
    return cases


def run_case(client, case, settings, owner, timeout):
    settings = {**settings, "language": case["language"]}
    fields = {"session": owner, "mode": case["mode"], "settings": json.dumps(settings)}
    if case.get("reference"):
        fields["expected_transcript"] = case["reference"]
    if case.get("audio"):
        audio = Path(case["audio"])
        with audio.open("rb") as handle:
            response = client.post(
                "/api/turn",
                data=fields,
                files={"audio": (audio.name, handle, "application/octet-stream")},
            )
    else:
        fields["text"] = case["text"]
        response = client.post("/api/turn", data=fields)
    response.raise_for_status()
    identifier = response.json()["id"]
    deadline = time.monotonic() + timeout
    finished = False
    try:
        while time.monotonic() < deadline:
            response = client.get(
                f"/api/jobs/{identifier}", params={"session": owner, "after": 2**31 - 1}
            )
            response.raise_for_status()
            result = response.json()
            result.pop("deltas", None)
            if result["status"] != "running":
                finished = True
                return result
            time.sleep(0.15)
        raise TimeoutError(f"Case {case['id']} exceeded {timeout:g} seconds.")
    finally:
        if not finished:
            # Cancel only the job submitted by this evaluator, including Ctrl-C.
            try:
                client.post(f"/api/jobs/{identifier}/cancel", params={"session": owner})
            except httpx.HTTPError:
                pass


def stats(values):
    values = sorted(
        value
        for value in values
        if isinstance(value, (float, int)) and math.isfinite(value)
    )
    if not values:
        return None

    def percentile(fraction):
        at = fraction * (len(values) - 1)
        low, high = math.floor(at), math.ceil(at)
        return values[low] + (values[high] - values[low]) * (at - low)

    return {"n": len(values), "p50": percentile(0.5), "p95": percentile(0.95)}


def summarize(records):
    groups = {}
    for record in records:
        result = record["result"]
        config = {
            key: value
            for key, value in result.get("config", {}).items()
            if key != "metrics_file"
        }
        identity = {
            "mode": record["case"]["mode"],
            "language": result.get("detected_language", record["case"]["language"]),
            "input_kind": "audio" if record["case"].get("audio") else "text",
            "cold_in_process": result.get("cold_in_process"),
            "run_options": result.get("run_options", {}),
            "config": config,
        }
        key = json.dumps(identity, sort_keys=True)
        group = groups.setdefault(
            key, {"identity": identity, "runs": 0, "failures": 0, "values": {}}
        )
        group["runs"] += 1
        if result.get("status") != "ok":
            group["failures"] += 1
            continue
        measurements = {
            "total_wall_s": result.get("total_wall_s"),
            "server_first_audio_s": result.get("response_first_audio_s"),
        }
        for name, stage in result.get("stages", {}).items():
            for metric in ("wall_s", "ttft_s", "tokens_per_s", "rtf"):
                measurements[f"{name}.{metric}"] = stage.get(metric)
        for task in ("asr", "tts_asr_proxy"):
            for metric in ("wer", "cer"):
                measurements[f"{task}.{metric}"] = (
                    result.get("quality", {}).get(task, {}).get(metric)
                )
        for name, value in measurements.items():
            if isinstance(value, (float, int)) and math.isfinite(value):
                group["values"].setdefault(name, []).append(value)
    for group in groups.values():
        group["statistics"] = {
            name: stats(values) for name, values in group.pop("values").items()
        }
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--engine", choices=("bilingual", "native", "python"), default="bilingual"
    )
    parser.add_argument(
        "--settings", type=Path, help="Optional JSON object of supported web settings."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--conversation",
        action="store_true",
        help="Share one conversation history across all cases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate cases without contacting a server or loading models.",
    )
    args = parser.parse_args()
    try:
        cases = load_cases(args.manifest)
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("--timeout must be a positive finite number.")
        if args.engine == "native" and any(case["language"] != "en" for case in cases):
            raise ValueError("Native English requires language=en on every case.")
        if args.engine == "native" and any(case["mode"] == "speak" for case in cases):
            raise ValueError(
                "Native English generates conversational speech through run; use bilingual for verbatim speak tests."
            )
        overrides = json.loads(args.settings.read_text()) if args.settings else {}
        if not isinstance(overrides, dict):
            raise TypeError("--settings must contain a JSON object.")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    settings = {
        "tts_backend": "liquid" if args.engine == "native" else "omnivoice",
        "offline": True,
        **overrides,
        "engine": args.engine,
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "valid_cases": len(cases),
                    "engine": args.engine,
                    "server_contacted": False,
                }
            )
        )
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    shared_owner = str(uuid.uuid4())
    with httpx.Client(base_url=args.url, timeout=10) as client:
        for case in cases:
            client_failure = False
            owner = shared_owner if args.conversation else str(uuid.uuid4())
            try:
                result = run_case(client, case, settings, owner, args.timeout)
            except (httpx.HTTPError, TimeoutError) as exc:
                result = {"status": "error", "error": str(exc)}
                client_failure = True
            records.append({"case": case, "result": result})
            report = {
                "schema_version": 1,
                "engine": args.engine,
                "requested_settings": settings,
                "scope": "Local server measurements. Excludes microphone endpoint delay and actual playback. WER/CER require supplied references; no automatic factual accuracy, MOS or emotion score.",
                "conversation": args.conversation,
                "planned_cases": len(cases),
                "attempted_cases": len(records),
                "records": records,
                "groups": summarize(records),
            }
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"{case['id']}: {result['status']}", flush=True)
            if client_failure:
                break  # Avoid submitting more jobs after uncertain server/network state.
    return int(any(record["result"].get("status") != "ok" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
