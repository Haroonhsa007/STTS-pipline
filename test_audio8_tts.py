"""
Quick test script for Edge0/Audio8-TTS-Realistic-0.6b

Models are downloaded into a "models" folder that sits next to this file.
Generated audio goes into an "outputs" folder next to this file.

Install:
    pip install "torch>=2.5.0" "torchaudio>=2.5.0" "transformers>=4.57.0,<5" \
                "soundfile>=0.12" "safetensors>=0.4" huggingface_hub

Usage:
    # 1) Plain generation (no reference voice)
    python test_audio8_tts.py

    # 2) Custom text
    python test_audio8_tts.py --text "Hello, this is a test."

    # 3) Zero-shot voice cloning (transcript must match the recording exactly)
    python test_audio8_tts.py --ref-audio reference.wav --ref-text "Exact words spoken in reference.wav" \
                              --text "Now I am speaking with the cloned voice."
"""

import argparse
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths: everything lives next to this file
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Make sure any other Hugging Face downloads/caches (remote code, etc.) also
# stay inside ./models. Must be set BEFORE importing transformers/huggingface_hub.
os.environ["HF_HOME"] = str(MODELS_DIR / "_hf_home")
os.environ["HF_HUB_CACHE"] = str(MODELS_DIR / "_hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR / "_hf_cache")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

MODEL_ID = "Edge0/Audio8-TTS-Realistic-0.6b"
LOCAL_MODEL_DIR = MODELS_DIR / "Audio8-TTS-Realistic-0.6b"

DEFAULT_TEXT = "Welcome to Audio8 TTS. It is wonderful to hear from you."


def download_model() -> Path:
    """Download the full repo into ./models/Audio8-TTS-Realistic-0.6b (skips if present)."""
    print(f"[1/3] Checking model files in: {LOCAL_MODEL_DIR}")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(LOCAL_MODEL_DIR),
    )
    return LOCAL_MODEL_DIR


def load_model(model_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[2/3] Loading model on {device} ({dtype}) ...")

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            dtype=dtype,
        )
        .eval()
        .to(device)
    )
    return processor, model, device


def synthesize(processor, model, device, text, ref_audio=None, ref_text=None, out_path=None):
    if ref_audio:
        if not ref_text:
            raise ValueError("--ref-text is required when using --ref-audio (must match the audio exactly).")
        inputs = processor(
            text=[text],
            reference_audio=[ref_audio],
            reference_text=[ref_text],
            return_tensors="pt",
        )
    else:
        inputs = processor(text=[text], return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            return_dict_in_generate=True,
        )
        waveforms, waveform_lengths = model.decode_audio(output.codes)
    elapsed = time.time() - start

    audio = waveforms[0, : int(waveform_lengths[0])].float().cpu().numpy()
    sr = model.config.codec_sample_rate
    sf.write(str(out_path), audio, sr)

    duration = len(audio) / sr
    print(f"      Saved {out_path.name}: {duration:.2f}s of audio in {elapsed:.2f}s "
          f"(real-time factor {elapsed / max(duration, 1e-6):.2f})")


def main():
    parser = argparse.ArgumentParser(description="Test Audio8-TTS-Realistic-0.6b")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to speak")
    parser.add_argument("--ref-audio", default=None, help="Reference .wav for voice cloning")
    parser.add_argument("--ref-text", default=None, help="Exact transcript of the reference audio")
    parser.add_argument("--out", default=None, help="Output file name (saved in ./outputs)")
    args = parser.parse_args()

    model_path = download_model()
    processor, model, device = load_model(model_path)

    out_name = args.out or ("clone_output.wav" if args.ref_audio else "output.wav")
    out_path = OUTPUT_DIR / out_name

    print("[3/3] Generating speech ...")
    synthesize(
        processor,
        model,
        device,
        text=args.text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        out_path=out_path,
    )
    print(f"Done. Output: {out_path}")


if __name__ == "__main__":
    main()