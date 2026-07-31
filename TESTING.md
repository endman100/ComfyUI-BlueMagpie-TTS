# Verification record

Verified on 2026-07-31 with:

- Windows / ComfyUI 0.27.0
- Python 3.12.8
- PyTorch 2.11.0+cu130
- NVIDIA GeForce RTX 5090 (32 GB)
- OpenFormosa/BlueMagpie-TTS model files stored under `D:/ComfyUI/models/bluemagpie/`

## Automated checks

```text
python -m pytest -q
11 passed
```

The live ComfyUI API registered `BlueMagpieModelLoader` and `BlueMagpieTTS` with the expected sockets.

## Live inference

Three live API workflows completed successfully:

1. Built-in `hung_yi_lee` speaker, Mandarin/English code-switching text, saved as 48 kHz mono FLAC.
2. Reference-audio conditioning using the first 5.12-second output as the reference, saved as 48 kHz mono FLAC.
3. Built-in `female_voice` final smoke test, saved as 48 kHz mono FLAC.

The first output was transcribed through `ComfyUI-WhisperLargeV3-Repack`. It recovered the Chinese sentence, `TTS`, `ComfyUI`, `API`, and the full English phrase; the only observed lexical difference was `Magpie` transcribed as `Mapie`.

## Reference-audio findings

A 60.37-second MP3 containing multiple spoken phrases and pauses completed inference but produced poor content accuracy. Using a clean 4.05-second single-speaker segment from the same file produced three 48 kHz mono FLAC samples whose intended short test sentence was substantially recovered by unprompted Whisper Large V3 transcription.

This is why the node recommends a clean 3-to-10-second reference and emits a warning for references longer than 15 seconds. The warning is guidance rather than a hard limit because upstream accepts longer inputs.
