from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import uuid
import weakref
from dataclasses import dataclass, field
from typing import Any

import soundfile as sf
import torch

try:
    import folder_paths
except ModuleNotFoundError:  # Lets unit tests import the module outside ComfyUI.
    folder_paths = None

try:
    from comfy.utils import ProgressBar
except ModuleNotFoundError:  # Lets unit tests import the module outside ComfyUI.
    ProgressBar = None


DEFAULT_MODEL_ID = "OpenFormosa/BlueMagpie-TTS"
MODEL_TYPE = "BLUE_MAGPIE_MODEL"
MODEL_FOLDER = "bluemagpie"
MODEL_FOLDER_NAMES = (MODEL_FOLDER, "llm", "LLM")
MODEL_REQUIRED_FILES = (
    "config.json",
    "pytorch_model.bin",
    "audiovae.pth",
    "tokenizer.json",
    os.path.join("checkpoints", "speaker_centroids.pt"),
)
BUILT_IN_SPEAKERS = ("hung_yi_lee", "female_voice")
REFERENCE_RECOMMENDED_MAX_SECONDS = 15.0
LONG_REFERENCE_WARNING = (
    "Long reference audio can reduce content accuracy. Use a clean, single-speaker clip around 3 to 10 seconds."
)
_LOGGER = logging.getLogger(__name__)
_MODEL_CACHE: weakref.WeakValueDictionary[tuple[str, str], BlueMagpieModelBundle] = weakref.WeakValueDictionary()


@dataclass
class BlueMagpieModelBundle:
    model: Any
    model_dir: str
    device: str
    speaker_ids: tuple[str, ...]
    speaker_centroids: torch.Tensor
    sample_rate: int
    lock: threading.Lock = field(default_factory=threading.Lock)

    def speaker_centroid(self, speaker_id: str) -> torch.Tensor:
        try:
            index = self.speaker_ids.index(speaker_id)
        except ValueError as exc:
            available = ", ".join(self.speaker_ids) or "none"
            raise ValueError(f"Speaker {speaker_id!r} is unavailable. Available speakers: {available}.") from exc
        return self.speaker_centroids[index]


def _register_model_folder() -> None:
    if folder_paths is None:
        return
    default_dir = os.path.join(folder_paths.models_dir, MODEL_FOLDER)
    try:
        folder_paths.add_model_folder_path(MODEL_FOLDER, default_dir, is_default=True)
    except TypeError:
        folder_paths.add_model_folder_path(MODEL_FOLDER, default_dir)


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if not path:
            continue
        normalized = os.path.normcase(os.path.abspath(os.path.expanduser(path)))
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(path)
    return result


def _folder_paths_for(folder_name: str) -> list[str]:
    if folder_paths is None:
        return []
    try:
        return list(folder_paths.get_folder_paths(folder_name))
    except KeyError:
        return []


def _model_roots() -> list[str]:
    roots: list[str] = []
    if folder_paths is not None:
        for folder_name in MODEL_FOLDER_NAMES:
            roots.extend(_folder_paths_for(folder_name))
        for folder_name in MODEL_FOLDER_NAMES:
            roots.append(os.path.join(folder_paths.models_dir, folder_name))
    return _dedupe_paths(roots)


def _is_model_dir(path: str) -> bool:
    return os.path.isdir(path) and all(os.path.isfile(os.path.join(path, name)) for name in MODEL_REQUIRED_FILES)


def _path_depth(root: str, path: str) -> int:
    relative = os.path.relpath(path, root)
    return 0 if relative == "." else len(relative.split(os.sep))


def _iter_model_dirs() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for root in _model_roots():
        if not os.path.isdir(root):
            continue
        for directory, dirnames, _filenames in os.walk(root, followlinks=True):
            dirnames[:] = [name for name in dirnames if name not in {".cache", ".git", "__pycache__"}]
            if _path_depth(root, directory) > 3:
                dirnames[:] = []
                continue
            if not _is_model_dir(directory):
                continue
            normalized = os.path.normcase(os.path.abspath(directory))
            if normalized not in seen:
                display_name = os.path.relpath(directory, root).replace(os.sep, "/")
                found.append((display_name if display_name != "." else os.path.basename(directory), directory))
                seen.add(normalized)
            dirnames[:] = []
    return found


def _model_choices() -> list[str]:
    choices = [display_name for display_name, _path in _iter_model_dirs()]
    if DEFAULT_MODEL_ID not in choices:
        choices.append(DEFAULT_MODEL_ID)
    return choices


def _local_model_path(selection: str) -> str | None:
    expanded = os.path.abspath(os.path.expanduser(selection))
    if _is_model_dir(expanded):
        return expanded

    normalized = selection.replace("\\", "/").strip("/")
    aliases = [normalized]
    if normalized == DEFAULT_MODEL_ID:
        aliases.append(os.path.basename(DEFAULT_MODEL_ID))

    for root in _model_roots():
        for alias in aliases:
            candidate = os.path.join(root, *alias.split("/"))
            if _is_model_dir(candidate):
                return candidate

    for display_name, path in _iter_model_dirs():
        if display_name == normalized or os.path.basename(path) == normalized:
            return path
    return None


def _default_download_dir(repo_id: str) -> str:
    roots = _folder_paths_for(MODEL_FOLDER)
    root = roots[0] if roots else os.path.join(folder_paths.models_dir, MODEL_FOLDER)
    return os.path.join(root, *repo_id.split("/"))


def _download_model(repo_id: str) -> str:
    if folder_paths is None:
        raise RuntimeError("ComfyUI folder_paths is required to download BlueMagpie-TTS.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "BlueMagpie Model Loader requires huggingface_hub. Install this node's requirements.txt and restart ComfyUI."
        ) from exc

    target = _default_download_dir(repo_id)
    os.makedirs(target, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=target)
    if not _is_model_dir(target):
        missing = [name for name in MODEL_REQUIRED_FILES if not os.path.isfile(os.path.join(target, name))]
        raise RuntimeError(f"BlueMagpie model download is incomplete. Missing files: {', '.join(missing)}.")
    return target


def _resolve_model_path(selection: str) -> str:
    selected = (selection or DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    local_path = _local_model_path(selected)
    return local_path if local_path is not None else _download_model(selected)


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but PyTorch cannot access a CUDA device.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was selected, but PyTorch cannot access Apple Silicon acceleration.")
    return device


def _load_speaker_table(model_dir: str) -> tuple[tuple[str, ...], torch.Tensor]:
    path = os.path.join(model_dir, "checkpoints", "speaker_centroids.pt")
    if not os.path.isfile(path):
        raise RuntimeError(f"BlueMagpie built-in speaker table is missing: {path}")
    table = torch.load(path, map_location="cpu", weights_only=True)
    speaker_ids = tuple(str(value) for value in table.get("speaker_ids", ()))
    centroids = table.get("centroids")
    if not speaker_ids or not isinstance(centroids, torch.Tensor) or centroids.ndim != 2:
        raise RuntimeError("BlueMagpie speaker_centroids.pt has an invalid format.")
    if centroids.shape[0] != len(speaker_ids) or not torch.isfinite(centroids).all():
        raise RuntimeError("BlueMagpie speaker_centroids.pt contains invalid speaker vectors.")
    return speaker_ids, centroids.detach().float().cpu()


def _load_model_bundle(model_dir: str, device: str) -> BlueMagpieModelBundle:
    try:
        from bluemagpie import BlueMagpieModel
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise ImportError(
            "BlueMagpie Model Loader requires the official bluemagpie-tts package. "
            "Install this node's requirements.txt and restart ComfyUI."
        ) from exc

    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    model = BlueMagpieModel.from_local(model_dir, tokenizer=tokenizer, training=False, device=device)
    speaker_ids, centroids = _load_speaker_table(model_dir)
    return BlueMagpieModelBundle(
        model=model,
        model_dir=model_dir,
        device=device,
        speaker_ids=speaker_ids,
        speaker_centroids=centroids,
        sample_rate=int(model.sample_rate),
    )


def _audio_to_wav_path(audio: dict[str, Any]) -> str:
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("reference_audio must be a ComfyUI AUDIO object.")
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    if waveform.ndim != 3 or waveform.shape[0] < 1 or waveform.shape[1] < 1 or waveform.shape[-1] < 1:
        raise ValueError(f"Expected reference waveform shape [B, C, T], got {tuple(waveform.shape)}.")
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise ValueError("reference_audio sample_rate must be positive.")

    waveform = waveform[0].detach().cpu().float()
    if not torch.isfinite(waveform).all():
        raise ValueError("reference_audio contains non-finite waveform samples.")
    samples = waveform.transpose(0, 1).numpy()
    if samples.shape[1] == 1:
        samples = samples[:, 0]
    temp_dir = folder_paths.get_temp_directory() if folder_paths is not None else tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, f"bluemagpie_reference_{uuid.uuid4().hex}.wav")
    sf.write(path, samples, sample_rate, format="WAV", subtype="PCM_16")
    return path


def _validated_reference_path(path: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(path.strip()))
    if not os.path.isfile(resolved):
        raise ValueError(f"Reference audio file does not exist: {resolved}")
    return resolved


def _reference_duration_seconds(path: str) -> float | None:
    try:
        duration = float(sf.info(path).duration)
    except (OSError, RuntimeError, ValueError):
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def _to_comfy_audio(waveform: torch.Tensor | Any, sample_rate: int) -> dict[str, Any]:
    audio = waveform if isinstance(waveform, torch.Tensor) else torch.as_tensor(waveform)
    audio = audio.detach().cpu().float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.ndim == 2:
        audio = audio.unsqueeze(0)
    elif audio.ndim != 3:
        raise ValueError(f"BlueMagpie returned an unsupported waveform shape: {tuple(audio.shape)}.")
    if audio.numel() == 0:
        raise ValueError("BlueMagpie returned an empty waveform.")
    if not torch.isfinite(audio).all():
        raise ValueError("BlueMagpie returned non-finite waveform samples.")
    return {"waveform": audio.contiguous(), "sample_rate": int(sample_rate)}


_register_model_folder()


class BlueMagpieModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = _model_choices()
        return {
            "required": {
                "model_path": (
                    model_choices,
                    {
                        "default": model_choices[0],
                        "tooltip": "Uses a local ComfyUI model folder first; downloads the public Hugging Face model when absent.",
                    },
                ),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = (MODEL_TYPE, "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load_model"
    CATEGORY = "audio/BlueMagpie TTS"
    DESCRIPTION = "Load OpenFormosa BlueMagpie-TTS for Taiwanese Mandarin and Mandarin/English code-switching."

    def load_model(self, model_path: str, device: str):
        resolved_path = _resolve_model_path(model_path)
        resolved_device = _resolve_device(device)
        key = (os.path.normcase(os.path.abspath(resolved_path)), resolved_device)
        bundle = _MODEL_CACHE.get(key)
        if bundle is None:
            bundle = _load_model_bundle(resolved_path, resolved_device)
            _MODEL_CACHE[key] = bundle

        info = {
            "model_path": model_path,
            "resolved_model_path": resolved_path,
            "device": resolved_device,
            "sample_rate": bundle.sample_rate,
            "speakers": list(bundle.speaker_ids),
        }
        return (bundle, json.dumps(info, ensure_ascii=False, indent=2))


class BlueMagpieTTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "你好，這是 BlueMagpie TTS 的 ComfyUI 測試。",
                    },
                ),
                "speaker": (list(BUILT_IN_SPEAKERS), {"default": "hung_yi_lee"}),
                "cfg_value": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "inference_timesteps": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "max_len": ("INT", {"default": 2000, "min": 16, "max": 8192, "step": 16}),
                "retry_badcase": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "reference_audio": (
                    "AUDIO",
                    {
                        "tooltip": "Optional rights-cleared voice reference. Use one clean speaker for about 3 to 10 seconds; it overrides speaker selection."
                    },
                ),
                "reference_audio_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional local reference audio path (for example WAV, FLAC, or MP3). Use a clean 3 to 10 second clip. Ignored when reference_audio is connected.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")
    FUNCTION = "generate"
    CATEGORY = "audio/BlueMagpie TTS"
    DESCRIPTION = "Generate 48 kHz Taiwanese Mandarin or Mandarin/English speech with a built-in or rights-cleared reference voice."

    def generate(
        self,
        model: BlueMagpieModelBundle,
        text: str,
        speaker: str,
        cfg_value: float,
        inference_timesteps: int,
        max_len: int,
        retry_badcase: bool,
        reference_audio: dict[str, Any] | None = None,
        reference_audio_path: str = "",
    ):
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")
        if not isinstance(model, BlueMagpieModelBundle):
            raise TypeError("model must come from BlueMagpie Model Loader.")

        temp_reference = None
        resolved_reference = ""
        if reference_audio is not None:
            temp_reference = _audio_to_wav_path(reference_audio)
            resolved_reference = temp_reference
        elif reference_audio_path and reference_audio_path.strip():
            resolved_reference = _validated_reference_path(reference_audio_path)

        reference_duration = _reference_duration_seconds(resolved_reference) if resolved_reference else None
        generation_warnings = []
        if reference_duration is not None and reference_duration > REFERENCE_RECOMMENDED_MAX_SECONDS:
            _LOGGER.warning(LONG_REFERENCE_WARNING)
            generation_warnings.append(LONG_REFERENCE_WARNING)

        pbar = ProgressBar(2) if ProgressBar is not None else None
        if pbar:
            pbar.update_absolute(1, 2)

        generation_args: dict[str, Any] = {
            "target_text": text.strip(),
            "cfg_value": float(cfg_value),
            "inference_timesteps": int(inference_timesteps),
            "max_len": int(max_len),
            "retry_badcase": bool(retry_badcase),
            "retry_badcase_ratio_threshold": 6.0,
        }
        conditioning = f"built_in:{speaker}"
        if resolved_reference:
            generation_args["reference_wav_path"] = resolved_reference
            conditioning = "reference_audio"
        else:
            generation_args["speaker_centroid"] = model.speaker_centroid(speaker)

        try:
            with model.lock:
                waveform = model.model.generate(**generation_args)
            audio = _to_comfy_audio(waveform, model.sample_rate)
        finally:
            if temp_reference:
                try:
                    os.remove(temp_reference)
                except OSError:
                    pass

        if pbar:
            pbar.update_absolute(2, 2)

        info = {
            "conditioning": conditioning,
            "cfg_value": float(cfg_value),
            "inference_timesteps": int(inference_timesteps),
            "max_len": int(max_len),
            "retry_badcase": bool(retry_badcase),
            "sample_rate": model.sample_rate,
            "samples": int(audio["waveform"].shape[-1]),
            "reference_duration_seconds": reference_duration,
            "warnings": generation_warnings,
        }
        return (audio, json.dumps(info, ensure_ascii=False, indent=2))
