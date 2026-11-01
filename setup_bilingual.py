"""Install the pinned local English/Urdu models without changing system packages."""

import hashlib
import json
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASR_REVISION = "57c4a1472cf25a03fd09ce03db0fd4eca1052bb3"
VOICE_REVISION = "874fcb41ececda20bcbc0782d6b4ea7f7d069ea8"
TRANSLATOR_REVISION = "302d78f00e6fdb50a1064059df7c392b735e9d05"
AUDIO_ARCHIVE = "audio-v0.9.3-bin-macos-arm64-metal.tar.gz"
AUDIO_SHA256 = "70a84c2b61ce5ef99547a12affab9b0d40daf08666baf7da4f940d5c7b722819"


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise SystemExit(
            "This tested bilingual installer targets your M4 Mac. Server deployment has not been validated."
        )
    from stts.bilingual import available, paths

    if not available() and shutil.disk_usage(ROOT).free < 5 * 2**30:
        raise SystemExit(
            "Free at least 5 GiB before a new bilingual installation. No existing files were removed."
        )
    from huggingface_hub import hf_hub_download, snapshot_download

    from setup_native import main as native_setup

    native_setup()
    base = ROOT / "models/bilingual"
    asr = base / "asr-flex"
    if not paths():
        archive = Path(
            hf_hub_download(
                "Mvkd2k06/indic-transcribe-flex-onnx",
                "variants/int8/indic-transcribe-flex-int8.tar",
                revision=ASR_REVISION,
                local_dir=asr,
            )
        )
        with tarfile.open(archive) as package:
            package.extractall(asr, filter="data")
        archive.unlink()
    hf_hub_download(
        "audio-cpp/audio.cpp-gguf",
        "OmniVoice-GGUF/omnivoice-q8_0.gguf",
        revision=VOICE_REVISION,
        local_dir=base / "tts",
    )
    snapshot_download(
        "JustFrederik/nllb-200-distilled-600M-ct2-int8",
        revision=TRANSLATOR_REVISION,
        local_dir=base / "translator-ct2",
    )
    snapshot_download(
        "facebook/nllb-200-distilled-600M",
        revision="f8d333a098d19b4fd9a8b18f94170487ad3f821d",
        local_dir=base / "translator",
        allow_patterns=["*.json", "*.model"],
    )
    binary = ROOT / "native/audio-v0.9.3/audiocpp_server"
    if not binary.is_file():
        with tempfile.TemporaryDirectory(prefix="stts-voice-install-") as folder:
            archive = Path(folder) / AUDIO_ARCHIVE
            urllib.request.urlretrieve(
                "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.3/"
                + AUDIO_ARCHIVE,
                archive,
            )
            if hashlib.sha256(archive.read_bytes()).hexdigest() != AUDIO_SHA256:
                raise SystemExit("Voice runtime checksum mismatch.")
            with tarfile.open(archive) as package:
                package.extractall(binary.parent, filter="data")
    metadata = {
        "asr": ASR_REVISION,
        "voice": VOICE_REVISION,
        "translation": TRANSLATOR_REVISION,
    }
    (base / "revisions.json").write_text(json.dumps(metadata, indent=2))
    print(
        "Installed. Run python web.py, choose Live English ↔ Urdu / Auto, and Start conversation."
    )


if __name__ == "__main__":
    main()
