# ComfyUI-BlueMagpie-TTS

ComfyUI nodes for [OpenFormosa/BlueMagpie-TTS](https://huggingface.co/OpenFormosa/BlueMagpie-TTS), a Taiwanese Mandarin and Mandarin/English code-switching TTS model.

## Nodes

- **BlueMagpie Model Loader** loads a local model or downloads the public Hugging Face release on first use.
- **BlueMagpie TTS** generates a standard ComfyUI `AUDIO` object using either a built-in speaker or optional reference audio.

The reference `AUDIO` input takes precedence over `reference_audio_path`, and either reference input takes precedence over the built-in speaker selection. Reference audio does not require a transcript; a clean clip of at least three seconds is recommended by the model authors.

For reliable cloning, use a clean, single-speaker clip around **3 to 10 seconds**. Long recordings containing multiple utterances, pauses, music, or other speakers can reduce content accuracy. The node reports a warning in `generation_info` and the ComfyUI log when a reference is longer than 15 seconds.

## Installation

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/endman100/ComfyUI-BlueMagpie-TTS
python -m pip install -r ComfyUI-BlueMagpie-TTS\requirements.txt
```

Restart ComfyUI after installation. The package supports Python 3.10 through 3.12. The upstream project documents Linux and macOS; Windows support should be treated as experimental.

This release was exercised on Windows with ComfyUI 0.27.0, Python 3.12.8, PyTorch 2.11.0+cu130, and an RTX 5090. CUDA, Apple Silicon MPS, and CPU device selections are exposed. See [TESTING.md](TESTING.md) for the verification record.

## Model location

The first execution of **BlueMagpie Model Loader** downloads about 8 GB to:

```text
ComfyUI/models/bluemagpie/OpenFormosa/BlueMagpie-TTS/
```

Local models are discovered under the registered `bluemagpie`, `llm`, and `LLM` model folders, including paths supplied through `extra_model_paths.yaml`.

Required model files are:

- `config.json`
- `pytorch_model.bin`
- `audiovae.pth`
- `tokenizer.json`
- `checkpoints/speaker_centroids.pt` for built-in speakers

## Suggested settings

- `cfg_value`: `2.0` to `2.8`
- `inference_timesteps`: `10`
- `retry_badcase`: enabled by default to reject obvious duration-ratio failures
- built-in speakers: `hung_yi_lee`, `female_voice`

The node returns 48 kHz audio for the current official checkpoint. Long passages should be split into punctuation-aware chunks before this node.

`reference_audio_path` accepts formats supported by the upstream `librosa` loader, including WAV, FLAC, and MP3 in a standard installation. Connected ComfyUI `AUDIO` input is converted to a temporary WAV and deleted after generation.

## Current scope

This package currently exposes general TTS, the two bundled speaker centroids, and transcript-free reference-audio cloning. Upstream speech continuation, arbitrary custom centroid input/extraction, streaming, and continuous batching are not yet exposed as ComfyUI nodes.

BlueMagpie-TTS is probabilistic and can mispronounce, omit, substitute, or add words even when automatic retry is enabled. Review important output before use.

## Voice rights and output review

Only use reference recordings and speaker embeddings you have permission to synthesize. Generated speech can be incorrect; review important outputs before publication or real-world use.

## Development test

```powershell
python -m pytest -q
```

The test suite mocks the 8 GB model and checks model discovery, required files, reclaimable loader caching, CUDA/MPS/CPU selection, built-in speaker conditioning, reference-audio validation and precedence, long-reference warnings, temporary-file cleanup, and ComfyUI `AUDIO` tensor shape.
