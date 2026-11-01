# Downloaded model inventory

Snapshot of this project's local assets on October 31, 2026. These are models
actually present, not a list of every candidate considered or every model in the
user's global caches. Logical sizes use 1 GiB = 2^30 bytes. Tokenizer/config files
are additional. No new model weights were downloaded for this documentation pass.

| Model asset | Local file(s) | Weight size | Source revision |
| --- | --- | ---: | --- |
| LFM3-Audio-2B backbone Q4_K_M | `models/native/LFM3-Audio-2B-Q4_K_M.gguf` | 1,099,626,016 B / 1.024 GiB | `02355c1a141e464a064f2ab2c276638dbcb7f38a` |
| LFM3 audio projector Q8_0 | `models/native/mmproj-LFM3-Audio-2B-Q8_0.gguf` | 303,811,840 B / 0.283 GiB | Same Liquid revision |
| Indic Flex int8 encoder | `models/bilingual/asr-flex/indic-transcribe-flex-int8/encoder.int8.onnx` and `.onnx.data` | 1,273,575,690 B | `57c4a1472cf25a03fd09ce03db0fd4eca1052bb3` |
| Indic Flex int8 decoder | Same directory, `decoder.int8.onnx` and `.onnx.data` | 432,048,625 B | Same ONNX revision |
| NLLB600M CT2 int8 | `models/bilingual/translator-ct2/model.bin` | 622,595,991 B / 0.580 GiB | `302d78f00e6fdb50a1064059df7c392b735e9d05` |
| OmniVoice Q8 | `models/bilingual/tts/OmniVoice-GGUF/omnivoice-q8_0.gguf` | 1,350,296,096 B / 1.258 GiB | `874fcb41ececda20bcbc0782d6b4ea7f7d069ea8` |

**Total model-weight files: 5,081,954,258 bytes / 4.733 GiB.**
The native model runs with llama.cpp `b11960`; the bilingual voice runs with
audio.cpp `v0.9.3`. Their local distributions occupy about 40 MiB and 87 MiB.
The source installers pin the revisions and runtime archive checksums.

NLLB tokenizer/config assets are also present in `models/bilingual/translator/`,
from `facebook/nllb-200-distilled-600M`, revision
`f8d333a098d19b4fd9a8b18f94170487ad3f821d`. They are not a second full NLLB model.
The cached `models/_hf_home` Audio8 Python modules and `_hf_cache` metadata are
not a complete downloaded Audio8 weight checkpoint. They are not selected by
the default pipeline and were left intact.

The discarded experimental Qwen variants, rejected ASR conversion and earlier
Liquid FP16 caches are not current installed presets. Legacy Python adapters may
need additional downloads; their source files do not mean their weights exist.

Check your own checkout without loading a model or using the network:

```bash
python tools/model_inventory.py
```

This checks file presence and byte sizes, not file integrity or model quality.
For a fresh checkout, run `python setup_bilingual.py` after installing the Python
dependencies. Weights and runtimes are excluded from the source repository/archive.

Primary model sources:

- [Liquid GGUF](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF)
- [Indic Flex ONNX conversion](https://huggingface.co/Mvkd2k06/indic-transcribe-flex-onnx), derived from [Bodhan Indic Flex](https://huggingface.co/bodhan-ai/indic-transcribe-flex)
- [NLLB CT2 conversion](https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8), based on [Meta NLLB600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [OmniVoice GGUF bundle](https://huggingface.co/audio-cpp/audio.cpp-gguf), based on [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
