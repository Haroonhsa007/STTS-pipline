import json

import httpx
import pytest

from tools.evaluate import load_cases, run_case, stats, summarize


def test_urdu_manifest_and_relative_audio_paths(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fixture")
    manifest = tmp_path / "cases.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "urdu",
                "mode": "transcribe",
                "language": "ur",
                "audio": "clip.wav",
                "reference": "السلام علیکم",
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    cases = load_cases(manifest)
    assert cases[0]["audio"] == str(audio)
    assert cases[0]["reference"] == "السلام علیکم"


@pytest.mark.parametrize(
    "case",
    [
        {"id": "bad", "mode": "transcribe", "text": "No audio"},
        {"id": "bad", "mode": "ask", "text": "Two inputs", "audio": "clip.wav"},
        {"id": "bad", "mode": "unknown", "text": "Unknown mode"},
        {
            "id": "bad",
            "mode": "ask",
            "text": "Typo",
            "reference_transcript": "Wrong field",
        },
    ],
)
def test_invalid_manifest_fails_before_contacting_server(tmp_path, case):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError):
        load_cases(path)


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = tmp_path / "cases.jsonl"
    case = json.dumps({"id": "same", "mode": "ask", "text": "Hello"})
    path.write_text(case + "\n" + case)
    with pytest.raises(ValueError, match="unique"):
        load_cases(path)


def test_successful_evaluation_strips_audio_deltas_and_does_not_cancel():
    paths = []

    def handle(request):
        paths.append((request.method, request.url.path))
        if request.url.path == "/api/turn":
            return httpx.Response(202, json={"id": "job"})
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "reply": "Hello",
                "deltas": [{"pcm": "not-exported"}],
            },
        )

    with httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(handle)
    ) as client:
        result = run_case(
            client,
            {"id": "one", "mode": "ask", "language": "en", "text": "Hello"},
            {"engine": "bilingual"},
            "owner",
            1,
        )
    assert result["reply"] == "Hello"
    assert "deltas" not in result
    assert not any(path.endswith("/cancel") for _, path in paths)


def test_timeout_cancels_only_owned_job():
    paths = []

    def handle(request):
        paths.append(str(request.url))
        return httpx.Response(202, json={"id": "owned-job"})

    with (
        httpx.Client(
            base_url="http://testserver", transport=httpx.MockTransport(handle)
        ) as client,
        pytest.raises(TimeoutError),
    ):
        run_case(
            client,
            {"id": "one", "mode": "ask", "language": "en", "text": "Hello"},
            {},
            "owned-session",
            0,
        )
    assert (
        paths[-1] == "http://testserver/api/jobs/owned-job/cancel?session=owned-session"
    )


def test_statistics_keep_language_cold_and_verification_separate():
    rows = []
    for language, cold, verify in [
        ("en", False, False),
        ("ur", False, False),
        ("en", True, False),
        ("en", False, True),
    ]:
        rows.append(
            {
                "case": {"mode": "run", "language": language, "text": "Sample"},
                "result": {
                    "status": "ok",
                    "detected_language": language,
                    "cold_in_process": cold,
                    "run_options": {"verify_tts": verify},
                    "total_wall_s": 2,
                },
            }
        )
    assert len(summarize(rows)) == 4
    assert stats([1, 3, 5]) == {"n": 3, "p50": 3, "p95": 4.8}
