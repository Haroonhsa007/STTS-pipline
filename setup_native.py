"""Install the pinned 4-bit English runtime into this project (no system changes)."""

import hashlib
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TAG = "b11960"
REVISION = "02355c1a141e464a064f2ab2c276638dbcb7f38a"
PACKAGES = {
    ("Darwin", "arm64"): (
        "macos-arm64",
        "58e5a5e1d7c2bf66cedd9c29e0323f3e570660a70a1e2156d32eed08f4cc1e3c",
    ),
    ("Linux", "x86_64"): (
        "ubuntu-x64",
        "31cc3a6bc638ac6c48dc4b1b74888f1eaadf3d14227c1e5f67daf9a49541d97a",
    ),
    ("Linux", "aarch64"): (
        "ubuntu-arm64",
        "0065ecfbffe8a790bb0d28059a3320cdc8f723b7f3f248cdc8bfc04790927715",
    ),
}


def main():
    key = (platform.system(), platform.machine())
    if key not in PACKAGES:
        raise SystemExit(
            "Native setup supports Apple Silicon macOS and Linux arm64/x64. Use the Python engine on other systems."
        )
    binary = ROOT / f"native/llama-{TAG}/llama-server"
    destination = ROOT / "models/native"
    destination.mkdir(parents=True, exist_ok=True)
    missing = not all(
        (destination / name).is_file()
        for name in ("LFM3-Audio-2B-Q4_K_M.gguf", "mmproj-LFM3-Audio-2B-Q8_0.gguf")
    )
    if missing and shutil.disk_usage(ROOT).free < 2 * 2**30:
        raise SystemExit(
            "Free at least 2 GiB before downloading the native model. No files were removed."
        )
    if not binary.is_file():
        package, expected = PACKAGES[key]
        name = f"llama-{TAG}-bin-{package}.tar.gz"
        with tempfile.TemporaryDirectory(prefix="stts-setup-") as folder:
            archive = Path(folder) / name
            urllib.request.urlretrieve(
                f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{name}",
                archive,
            )
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if digest != expected:
                raise SystemExit(
                    "Runtime checksum does not match the published pinned release."
                )
            with tarfile.open(archive) as tar:
                tar.extractall(ROOT / "native", filter="data")
    # HTTP mode avoids Xet reconstruction staging competing with a nearly-full disk.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    for filename in ("LFM3-Audio-2B-Q4_K_M.gguf", "mmproj-LFM3-Audio-2B-Q8_0.gguf"):
        path = hf_hub_download(
            "LiquidAI/LFM3-Audio-2B-GGUF",
            filename,
            revision=REVISION,
            local_dir=destination,
        )
        print(path)
    (destination / "revision.txt").write_text(REVISION)
    print(
        "Installed. Start python web.py and select Native 4-bit. Use CPU device on the server."
    )


if __name__ == "__main__":
    main()
