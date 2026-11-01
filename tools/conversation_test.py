"""Replay 1000+ generated text tokens through real speech recognition and synthesis.

Uses macOS's installed Samantha voice for repeatable user speech, never the mic.
No model downloads. Results are scoped to this synthetic speech replay.
"""

import argparse
import json
import statistics
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import httpx

PROMPTS = [
    "I'm setting up a quiet home office. My budget is six hundred dollars and my desk is one hundred twenty centimeters wide.",
    "I live in Regina and work as a software developer. I have video calls every day. What should I buy first?",
    "I already have a good laptop. Please split the budget between a chair, microphone, and lighting.",
    "I like warm light and I don't want a big microphone in front of my face. What would you suggest?",
    "Before we go further, remind me of my budget and desk width.",
    "A friend suggested a headset. What are the advantages compared with a separate microphone?",
    "I sometimes work late at night. How can I keep calls quiet without sounding muffled?",
    "Would you put the desk near the window or against a wall? Explain your reasoning.",
    "Actually, I can increase the budget to seven hundred dollars. Keep one hundred dollars unspent for later.",
    "Given that change, what should I improve first, and why?",
    "What should I test before buying anything else? I care about clear speech more than appearances.",
    "Let's talk about the chair. Which adjustments matter most for long programming sessions?",
    "Can you give me a simple daily routine so I don't sit still all day?",
    "I tend to forget breaks when I am debugging. Make that routine easy to remember.",
    "How should I position the screen and keyboard on a narrow desk?",
    "Quick memory check: where do I live, what do I do, and what lighting do I prefer?",
    "Explain one common mistake with microphone placement and how to fix it.",
    "If the room echoes, what can I try using things I already own?",
    "My video calls are usually in the morning. How could the window affect the lighting?",
    "I also want the desk easy to clean. Suggest a practical way to manage cables.",
    "What should I check during a short test call with a friend?",
    "If they hear keyboard noise, what changes would you make in order?",
    "Compare the headset and separate microphone options one last time for my needs.",
    "Summarize our current plan. Remember the updated budget and the money I want to keep unspent.",
    "What would be your first three actions tomorrow morning, and why?",
    "How can I tell whether a change has actually improved my calls rather than just sounding different?",
    "Tell me how to keep this setup comfortable in winter without buying more equipment.",
    "Give me a concise final recommendation that fits everything I told you.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--min-tokens", type=int, default=1000)
    parser.add_argument(
        "--engine", choices=("native", "bilingual"), default="bilingual"
    )
    parser.add_argument("--max-turns", type=int, default=56)
    parser.add_argument("--output", default="outputs/conversation-test.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    owner = str(uuid.uuid4())
    records = []
    total = 0
    settings = json.dumps(
        {
            "engine": args.engine,
            "language": "auto" if args.engine == "bilingual" else "en",
            "tts_backend": "omnivoice" if args.engine == "bilingual" else "liquid",
            "offline": True,
            "max_new_tokens": 1024,
        }
    )
    with (
        httpx.Client(base_url=args.url, timeout=150) as client,
        tempfile.TemporaryDirectory(prefix="stts-replay-") as folder,
    ):
        for index in range(1, args.max_turns + 1):
            prompt = PROMPTS[(index - 1) % len(PROMPTS)]
            if total >= args.min_tokens and index > 24:
                break
            path = Path(folder) / "user.aiff"
            subprocess.run(
                ["say", "-v", "Samantha", "-r", "175", "-o", str(path), prompt],
                check=True,
                capture_output=True,
            )
            started = time.perf_counter()
            with path.open("rb") as audio:
                response = client.post(
                    "/api/turn",
                    data={
                        "session": owner,
                        "settings": settings,
                        "mode": "run",
                        "expected_transcript": prompt,
                    },
                    files={"audio": ("user.aiff", audio, "audio/aiff")},
                )
            response.raise_for_status()
            identifier = response.json()["id"]
            first_received = None
            chunk_count = 0
            last_seq = 0
            result = None
            with client.stream(
                "GET", f"/api/jobs/{identifier}/events", params={"session": owner}
            ) as stream:
                stream.raise_for_status()
                for line in stream.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    for delta in data.pop("deltas", []):
                        if delta["seq"] <= last_seq:
                            continue
                        last_seq = delta["seq"]
                        chunk_count += 1
                        if first_received is None:
                            first_received = time.perf_counter() - started
                    if data["status"] != "running":
                        result = data
                        break
            if result is None:
                raise RuntimeError("Stream ended without a result")
            if result["status"] != "ok":
                raise RuntimeError(result.get("error", result["status"]))
            total += result.get("text_output_tokens", 0)
            row = {
                "turn": index,
                "prompt": prompt,
                "reply": result.get("reply"),
                "transcript": result.get("transcript"),
                "text_output_tokens": result.get("text_output_tokens"),
                "cumulative_text_tokens": total,
                "server_first_audio_s": result.get("response_first_audio_s"),
                "http_first_audio_s": first_received,
                "total_wall_s": result.get("total_wall_s"),
                "chunks": chunk_count,
                "reply_truncated": result.get("reply_truncated"),
                "quality": result.get("quality"),
                "stages": result.get("stages"),
                "system": result.get("system"),
            }
            records.append(row)
            report = {
                "engine": args.engine,
                "scope": "Synthetic English speech via installed macOS Samantha voice; all STT and replies use real local model weights. HTTP chunk receipt excludes actual speaker playback.",
                "target_text_tokens": args.min_tokens,
                "generated_text_tokens": total,
                "turns": records,
            }
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            print(
                f"turn {index}: text_tokens={result.get('text_output_tokens')} total_tokens={total} first_audio={first_received:.3f}s duration={result.get('total_wall_s'):.3f}s truncated={result.get('reply_truncated')}",
                flush=True,
            )
            if result.get("reply_truncated"):
                raise RuntimeError(f"Turn {index} was truncated")
    if total < args.min_tokens:
        raise RuntimeError(
            f"Only {total} generated text tokens; increase the replay prompts"
        )
    latencies = sorted(row["http_first_audio_s"] for row in records)
    report["summary"] = {
        "turns": len(records),
        "generated_text_tokens": total,
        "http_first_audio_p50_s": statistics.median(latencies),
        "http_first_audio_p95_s": latencies[
            min(len(latencies) - 1, int(0.95 * (len(latencies) - 1)))
        ],
        "truncated": sum(bool(row["reply_truncated"]) for row in records),
        "mean_wer": statistics.mean(row["quality"]["asr"]["wer"] for row in records),
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
