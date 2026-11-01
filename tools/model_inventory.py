"""List project model files and missing assets without loading/downloading models."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "Liquid LFM3 Audio Q4 + Q8 projector": [
        "models/native/LFM3-Audio-2B-Q4_K_M.gguf",
        "models/native/mmproj-LFM3-Audio-2B-Q8_0.gguf",
    ],
    "Indic Flex int8": [
        "models/bilingual/asr-flex/indic-transcribe-flex-int8/encoder.int8.onnx",
        "models/bilingual/asr-flex/indic-transcribe-flex-int8/encoder.int8.onnx.data",
        "models/bilingual/asr-flex/indic-transcribe-flex-int8/decoder.int8.onnx",
        "models/bilingual/asr-flex/indic-transcribe-flex-int8/decoder.int8.onnx.data",
        "models/bilingual/asr-flex/indic-transcribe-flex-int8/tokenizer.json",
    ],
    "OmniVoice Q8": ["models/bilingual/tts/OmniVoice-GGUF/omnivoice-q8_0.gguf"],
    "NLLB600M CT2 int8 + tokenizer": [
        "models/bilingual/translator-ct2/model.bin",
        "models/bilingual/translator-ct2/config.json",
        "models/bilingual/translator-ct2/shared_vocabulary.txt",
        "models/bilingual/translator/tokenizer.json",
    ],
}


def inventory(root=ROOT):
    assets = {}
    for name, files in EXPECTED.items():
        missing = [name for name in files if not (root / name).is_file()]
        assets[name] = {"files_present": not missing, "missing": missing}
    weights = []
    for path in sorted((root / "models").rglob("*")):
        if path.is_file() and path.suffix in {
            ".gguf",
            ".onnx",
            ".data",
            ".bin",
            ".safetensors",
            ".pt",
            ".pth",
        }:
            weights.append(
                {"path": str(path.relative_to(root)), "bytes": path.stat().st_size}
            )
    total = sum(item["bytes"] for item in weights)
    return {
        "scope": "Project-local file presence and logical byte sizes only; not a checksum or inference validation. No models loaded or downloaded.",
        "assets": assets,
        "weight_files": weights,
        "weight_bytes": total,
        "weight_gib": round(total / 2**30, 3),
    }


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2, ensure_ascii=False))
