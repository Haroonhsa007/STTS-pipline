# Source license and third-party assets

Original STTS Pipeline source code and documentation are licensed under
[Apache-2.0](LICENSE); see [NOTICE](NOTICE) for attribution. This section records
third-party provenance and does not replace publishers' actual license terms.

Model weights are downloaded separately and excluded from the source archive.
The pipeline's Apache-2.0 license does not relicense those weights, codecs,
conversions, upstream source or native runtimes.

| Asset | Publisher license information |
| --- | --- |
| Liquid LFM3-Audio-2B | Model card identifies the custom `lfm1.0` license: [publisher](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF) |
| Indic Flex ONNX conversion | Its bundled `Bodhan_AI_Open_Model_License.md` and `NOTICE.md` cover provenance and derivative terms; retain both. [conversion](https://huggingface.co/Mvkd2k06/indic-transcribe-flex-onnx), [upstream](https://huggingface.co/bodhan-ai/indic-transcribe-flex) |
| NLLB600M / CT2 conversion | Upstream model card identifies CC-BY-NC-4.0: [Meta](https://huggingface.co/facebook/nllb-200-distilled-600M). Check the conversion's notices too: [CT2 repository](https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8) |
| OmniVoice / GGUF bundle | Publisher distinguishes Apache-2.0 code from CC-BY-NC pretrained weights: [OmniVoice license section](https://huggingface.co/k2-fsa/OmniVoice#license). The GGUF bundle also contains audio components; inspect [converter/runtime documentation](https://github.com/0xShug0/audio.cpp/blob/main/docs/models/omnivoice.md) and retained notices. |

Native software is separate: [llama.cpp](https://github.com/ggml-org/llama.cpp)
and [audio.cpp](https://github.com/0xShug0/audio.cpp). Python dependencies also keep
their own licenses. Retain the notices supplied with any downloaded distribution.

Before distributing bundled weights or modified conversions, read the actual
licenses for those assets. The phrase "open-source pipeline" describes the
licensed pipeline source; it does not mean all selected models have permissive
or commercial-use terms.
